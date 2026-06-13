import os
import json
import logging
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from utils.metrics import (
    compute_precision,
    compute_recall,
    compute_f1,
    compute_class_ap,
    compute_map,
    match_predictions,
)

logger = logging.getLogger(__name__)


class DefectDataset(Dataset):
    def __init__(self, split_json_path, class_names, transform=None):
        with open(split_json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.class_names = class_names
        self.class_to_idx = {name: idx + 1 for idx, name in enumerate(class_names)}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img = Image.open(entry["image"]).convert("RGB")

        boxes = []
        labels = []
        for obj in entry.get("objects", []):
            name = obj["name"]
            if name in self.class_to_idx:
                boxes.append(obj["bbox"])
                labels.append(self.class_to_idx[name])

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


class DetectionEvaluator:
    def __init__(self, class_names, iou_threshold=0.5):
        self.class_names = class_names
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.all_pred_boxes = []
        self.all_pred_labels = []
        self.all_pred_scores = []
        self.all_gt_boxes = []
        self.all_gt_labels = []
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def update(self, predictions, targets):
        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get("boxes", torch.zeros((0, 4))).cpu().numpy().tolist()
            pred_labels = pred.get("labels", torch.zeros((0,), dtype=torch.int64)).cpu().numpy().tolist()
            pred_scores = pred.get("scores", torch.zeros((0,))).cpu().numpy().tolist()

            gt_boxes = target.get("boxes", torch.zeros((0, 4))).cpu().numpy().tolist()
            gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.int64)).cpu().numpy().tolist()

            self.all_pred_boxes.append(pred_boxes)
            self.all_pred_labels.append(pred_labels)
            self.all_pred_scores.append(pred_scores)
            self.all_gt_boxes.append(gt_boxes)
            self.all_gt_labels.append(gt_labels)

            tp, fp, fn, _ = match_predictions(
                pred_boxes, pred_labels, pred_scores,
                gt_boxes, gt_labels,
                self.iou_threshold,
            )
            self.total_tp += tp
            self.total_fp += fp
            self.total_fn += fn

    def compute(self):
        precision = compute_precision(self.total_tp, self.total_fp)
        recall = compute_recall(self.total_tp, self.total_fn)
        f1 = compute_f1(precision, recall)

        class_aps = {}
        num_classes = len(self.class_names)
        for class_idx in range(num_classes):
            class_id = class_idx + 1
            ap = compute_class_ap(
                self.all_pred_boxes,
                self.all_pred_labels,
                self.all_pred_scores,
                self.all_gt_boxes,
                self.all_gt_labels,
                class_id,
                self.iou_threshold,
            )
            class_aps[self.class_names[class_idx]] = ap

        mAP = compute_map(list(class_aps.values()))

        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP": mAP,
            "class_AP": class_aps,
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
        }
        return results

    def print_results(self, results=None):
        if results is None:
            results = self.compute()

        logger.info("=" * 50)
        logger.info("Detection Evaluation Results")
        logger.info("=" * 50)
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall:    {results['recall']:.4f}")
        logger.info(f"F1 Score:  {results['f1']:.4f}")
        logger.info(f"mAP:       {results['mAP']:.4f}")
        logger.info("-" * 50)
        for cls_name, ap in results["class_AP"].items():
            logger.info(f"  {cls_name}: AP = {ap:.4f}")
        logger.info("=" * 50)

        return results
