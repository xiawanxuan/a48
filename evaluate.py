import os
import sys
import yaml
import json
import argparse
import logging
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.backbone import build_detection_model
from modules.evaluation import DefectDataset, collate_fn, DetectionEvaluator
from utils.metrics import compute_precision, compute_recall, compute_f1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate defect detection model")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Config YAML path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--iou_threshold", type=float, default=None, help="IoU threshold override")
    parser.add_argument("--score_threshold", type=float, default=None, help="Score threshold override")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["classes"]
    device_cfg = config.get("device", "auto")
    inf_cfg = config.get("inference", {})
    eval_cfg = config.get("evaluation", {})

    iou_threshold = args.iou_threshold or eval_cfg.get("iou_threshold", 0.5)
    score_threshold = args.score_threshold or inf_cfg.get("score_threshold", 0.5)

    model, device = build_detection_model(num_classes, pretrained=False, device=device_cfg)

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    from modules.augmentation import ValTransform
    dataset_root = config["dataset"]["root"]
    split_path = os.path.join(dataset_root, "splits", f"{args.split}.json")

    if not os.path.exists(split_path):
        logger.error(f"Split file not found: {split_path}")
        return

    dataset = DefectDataset(split_path, class_names, transform=ValTransform(
        image_size=inf_cfg.get("image_size", 800),
    ))

    batch_size = config["training"].get("batch_size", 4)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    evaluator = DetectionEvaluator(class_names, iou_threshold=iou_threshold)

    logger.info(f"Evaluating on {args.split} split ({len(dataset)} images)...")
    logger.info(f"IoU threshold: {iou_threshold}, Score threshold: {score_threshold}")

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            filtered_outputs = []
            for output in outputs:
                keep = output["scores"] >= score_threshold
                filtered_outputs.append({
                    "boxes": output["boxes"][keep],
                    "labels": output["labels"][keep],
                    "scores": output["scores"][keep],
                })

            evaluator.update(filtered_outputs, targets)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {(i+1) * batch_size} images...")

    results = evaluator.print_results()

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, f"eval_{args.split}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {results_path}")


if __name__ == "__main__":
    main()
