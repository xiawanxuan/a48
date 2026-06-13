import math

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detection_model(num_classes, pretrained=True, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)
    return model, device


def build_optimizer(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    lr = config.get("training", {}).get("lr", 0.001)
    momentum = config.get("training", {}).get("momentum", 0.9)
    weight_decay = config.get("training", {}).get("weight_decay", 0.0005)

    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    return optimizer


def build_lr_scheduler(optimizer, config):
    scheduler_cfg = config.get("training", {}).get("lr_scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "cosine")
    epochs = config.get("training", {}).get("epochs", 50)

    if scheduler_type == "cosine":
        warmup_epochs = scheduler_cfg.get("warmup_epochs", 3)
        min_lr = scheduler_cfg.get("min_lr", 1e-6)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "step":
        step_size = scheduler_cfg.get("step_size", 10)
        gamma = scheduler_cfg.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

    return scheduler
