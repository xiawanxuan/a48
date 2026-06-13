import os
import sys
import json
import yaml
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.inference import Detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Local image inference for defect detection")
    parser.add_argument("--image", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Config YAML path")
    parser.add_argument("--output", type=str, default="./inference_results", help="Output directory for results")
    parser.add_argument("--score_threshold", type=float, default=None, help="Score threshold override")
    parser.add_argument("--visualize", action="store_true", help="Save visualized detection results")
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["classes"]
    inf_cfg = config.get("inference", {})

    score_threshold = args.score_threshold or inf_cfg.get("score_threshold", 0.5)
    nms_threshold = inf_cfg.get("nms_threshold", 0.5)

    detector = Detector(
        model_path=args.model,
        num_classes=num_classes,
        device="auto",
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.image):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
    else:
        image_paths = [args.image]

    all_results = []

    for img_path in image_paths:
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue

        logger.info(f"Processing: {img_path}")
        detections = detector.detect(img_path)

        result = {
            "image": img_path,
            "detections": detections,
            "num_defects": len(detections),
        }
        all_results.append(result)

        logger.info(f"  Found {len(detections)} defects")
        for det in detections:
            cls_name = class_names[det["label"] - 1] if det["label"] <= len(class_names) else f"class_{det['label']}"
            logger.info(f"    {cls_name}: score={det['score']:.4f}, bbox={det['bbox']}")

        if args.visualize:
            vis_path = os.path.join(
                args.output,
                os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg",
            )
            detector.visualize(img_path, detections, class_names, vis_path)

    results_path = os.path.join(args.output, "detection_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
