import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F


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
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        img_resized = F.resize(img, (image_size, image_size))

        img_tensor = F.to_tensor(img_resized)
        img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        scale_x = orig_w / image_size
        scale_y = orig_h / image_size

        return img_tensor, scale_x, scale_y

    @torch.no_grad()
    def detect(self, img, image_size=800):
        img_tensor, scale_x, scale_y = self.preprocess(img, image_size)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        predictions = self.model(img_tensor)[0]

        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

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
