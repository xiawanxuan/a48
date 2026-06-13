import os
import sys
import yaml
import argparse
import logging
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.http_api import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Start defect detection HTTP API server")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Config YAML path")
    parser.add_argument("--host", type=str, default=None, help="Host override")
    parser.add_argument("--port", type=int, default=None, help="Port override")
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["classes"]
    device_cfg = config.get("device", "auto")
    inf_cfg = config.get("inference", {})

    http_cfg = config.get("http_api", {})
    host = args.host or http_cfg.get("host", "0.0.0.0")
    port = args.port or http_cfg.get("port", 8000)

    app = create_app(
        model_path=args.model,
        num_classes=num_classes,
        class_names=class_names,
        device=device_cfg,
        score_threshold=inf_cfg.get("score_threshold", 0.5),
        nms_threshold=inf_cfg.get("nms_threshold", 0.5),
    )

    logger.info(f"Starting API server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
