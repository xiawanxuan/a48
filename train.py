import os
import sys
import yaml
import random
import logging
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F_t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.backbone import build_detection_model, build_optimizer, build_lr_scheduler
from modules.callbacks import CallbackManager
from modules.augmentation import MosaicAugment, CutMixAugment, TrainTransform, ValTransform
from modules.evaluation import DefectDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MosaicCutMixDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mosaic_aug, cutmix_aug, train_transform):
        self.base_dataset = base_dataset
        self.mosaic_aug = mosaic_aug
        self.cutmix_aug = cutmix_aug
        self.train_transform = train_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        img_pil = F_t.to_pil_image(img)
        boxes = target["boxes"]
        labels = target["labels"]

        if self.mosaic_aug and random.random() < self.mosaic_aug.prob:
            indices = random.sample(range(len(self.base_dataset)), min(4, len(self.base_dataset)))
            samples = []
            for i in indices:
                s_img, s_target = self.base_dataset[i]
                s_img_pil = F_t.to_pil_image(s_img)
                samples.append((s_img_pil, s_target["boxes"], s_target["labels"]))

            if len(samples) >= 2:
                while len(samples) < 4:
                    samples.append(samples[0])
                img_pil, boxes, labels = self.mosaic_aug(samples)

        if self.cutmix_aug and random.random() < self.cutmix_aug.prob and len(self.base_dataset) > 1:
            other_idx = random.randint(0, len(self.base_dataset) - 1)
            if other_idx != idx:
                other_img, other_target = self.base_dataset[other_idx]
                other_img_pil = F_t.to_pil_image(other_img)
                img_pil, boxes, labels = self.cutmix_aug(
                    img_pil, boxes, labels,
                    other_img_pil, other_target["boxes"], other_target["labels"],
                )

        img_tensor, boxes, labels = self.train_transform(img_pil, boxes, labels)

        target = {
            "boxes": boxes,
            "labels": labels.long(),
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target


def train_one_epoch(model, optimizer, data_loader, device, epoch, callback_mgr):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        global_step = epoch * len(data_loader) + i
        if callback_mgr:
            callback_mgr.on_train_step(epoch, losses.item(), global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, data_loader, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train defect detection model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Config YAML path")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config loaded from {args.config}")

    device_cfg = config.get("device", "auto")
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["classes"]

    model, device = build_detection_model(num_classes, pretrained=True, device=device_cfg)
    logger.info(f"Model built: Faster R-CNN ResNet50-FPN, num_classes={num_classes}, device={device}")

    optimizer = build_optimizer(model, config)
    scheduler = build_lr_scheduler(optimizer, config)
    callback_mgr = CallbackManager(config)

    dataset_root = config["dataset"]["root"]
    splits_dir = os.path.join(dataset_root, "splits")

    train_transform = TrainTransform(
        image_size=config.get("inference", {}).get("image_size", 800),
        hflip=config.get("augmentation", {}).get("hflip", True),
        color_jitter=config.get("augmentation", {}).get("color_jitter", True),
    )
    val_transform = ValTransform(
        image_size=config.get("inference", {}).get("image_size", 800),
    )

    train_split_path = os.path.join(splits_dir, "train.json")
    val_split_path = os.path.join(splits_dir, "val.json")

    if not os.path.exists(train_split_path):
        logger.error(f"Train split not found: {train_split_path}")
        logger.error("Run 'python data/split_dataset.py --root <dataset_root>' first")
        return

    base_train_dataset = DefectDataset(train_split_path, class_names, transform=train_transform)
    val_dataset = DefectDataset(val_split_path, class_names, transform=val_transform)

    aug_cfg = config.get("augmentation", {})
    mosaic_aug = None
    cutmix_aug = None

    if aug_cfg.get("mosaic", {}).get("enabled", True):
        mosaic_aug = MosaicAugment(
            size=config.get("inference", {}).get("image_size", 800),
            prob=aug_cfg.get("mosaic", {}).get("prob", 0.5),
        )
    if aug_cfg.get("cutmix", {}).get("enabled", True):
        cutmix_aug = CutMixAugment(
            alpha=aug_cfg.get("cutmix", {}).get("alpha", 1.0),
            prob=aug_cfg.get("cutmix", {}).get("prob", 0.3),
        )

    train_dataset = MosaicCutMixDataset(base_train_dataset, mosaic_aug, cutmix_aug, train_transform)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    epochs = config["training"]["epochs"]
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, callback_mgr)
        val_loss = validate(model, val_loader, device)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | LR: {current_lr:.8f}"
        )

        should_stop = False
        if callback_mgr:
            should_stop = callback_mgr.on_epoch_end(model, epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if should_stop:
            logger.info("Training stopped early by callback")
            break

    if callback_mgr:
        callback_mgr.on_train_end()

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
