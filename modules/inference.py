import torch
import logging
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, model_path, num_classes, device="auto", score_threshold=0.5, nms_threshold=0.5):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.num_classes = num_classes

        from modules.backbone import build_detection_model
        self.model, _ = build_detection_model(num_classes, pretrained=False, device=self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def preprocess(self, img, image_size=800):
        try:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            orig_w, orig_h = img.size

            if orig_w <= 0 or orig_h <= 0:
                raise ValueError(f"Invalid image size: {orig_w}x{orig_h}")

            min_size = 32
            if orig_w < min_size or orig_h < min_size:
                scale = max(min_size / orig_w, min_size / orig_h)
                new_w = max(min_size, int(orig_w * scale))
                new_h = max(min_size, int(orig_h * scale))
                img = img.resize((new_w, new_h), Image.BICUBIC)
                orig_w, orig_h = new_w, new_h
                logger.warning(f"Image too small, upscaled to {orig_w}x{orig_h}")

            ratio = min(image_size / orig_w, image_size / orig_h)
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
            new_w = max(1, new_w)
            new_h = max(1, new_h)

            img_resized = F.resize(img, (new_h, new_w))

            pad_left = (image_size - new_w) // 2
            pad_top = (image_size - new_h) // 2
            pad_right = image_size - new_w - pad_left
            pad_bottom = image_size - new_h - pad_top

            img_padded = ImageOps.expand(
                img_resized,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=(114, 114, 114),
            )

            img_tensor = F.to_tensor(img_padded)
            img_tensor = F.normalize(
                img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            scale = ratio
            pad = (pad_left, pad_top)

            return img_tensor, scale, pad, (orig_w, orig_h)

        except Exception as e:
            logger.error(f"Preprocess failed: {str(e)}")
            blank = Image.new("RGB", (image_size, image_size), (114, 114, 114))
            img_tensor = F.to_tensor(blank)
            img_tensor = F.normalize(
                img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            return img_tensor, 1.0, (0, 0), (image_size, image_size)

    @torch.no_grad()
    def detect(self, img, image_size=800):
        img_tensor, scale, (pad_left, pad_top), (orig_w, orig_h) = self.preprocess(
            img, image_size
        )
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        predictions = self.model(img_tensor)[0]

        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
        boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
        boxes[:, 3] = (boxes[:, 3] - pad_top) / scale

        boxes[:, 0] = boxes[:, 0].clip(min=0, max=orig_w)
        boxes[:, 1] = boxes[:, 1].clip(min=0, max=orig_h)
        boxes[:, 2] = boxes[:, 2].clip(min=0, max=orig_w)
        boxes[:, 3] = boxes[:, 3].clip(min=0, max=orig_h)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        labels = labels[valid]
        scores = scores[valid]

        results = []
        for box, label, score in zip(boxes, labels, scores):
            results.append({
                "bbox": [float(x) for x in box],
                "label": int(label),
                "score": float(score),
            })

        return results

    def visualize(self, img, detections, class_names=None, output_path=None):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        draw = ImageDraw.Draw(img)

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for det in detections:
            bbox = det["bbox"]
            label = det["label"]
            score = det["score"]

            color = colors[(label - 1) % len(colors)]

            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            if class_names and 0 < label <= len(class_names):
                name = class_names[label - 1]
            else:
                name = f"class_{label}"

            text = f"{name}: {score:.2f}"
            draw.text((x1, y1 - 10), text, fill=color)

        if output_path:
            img.save(output_path)
            print(f"Visualization saved to {output_path}")

        return img
