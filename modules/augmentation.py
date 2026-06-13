import random
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


class MosaicAugment:
    def __init__(self, size=800, prob=0.5):
        self.size = size
        self.prob = prob

    def __call__(self, samples):
        if random.random() > self.prob:
            return samples[0]

        imgs, bboxes_list, labels_list = [], [], []
        for img, bboxes, labels in samples:
            imgs.append(img)
            bboxes_list.append(bboxes)
            labels_list.append(labels)

        s = self.size
        mosaic_img = Image.new("RGB", (s * 2, s * 2), (114, 114, 114))
        merged_bboxes = []
        merged_labels = []

        xc = random.randint(s // 2, s * 3 // 2)
        yc = random.randint(s // 2, s * 3 // 2)

        positions = [
            (0, 0, xc, yc),
            (xc, 0, s * 2, yc),
            (0, yc, xc, s * 2),
            (xc, yc, s * 2, s * 2),
        ]

        for i, (img, bboxes, labels) in enumerate(zip(imgs, bboxes_list, labels_list)):
            w, h = img.size
            x1, y1, x2, y2 = positions[i]
            target_w = x2 - x1
            target_h = y2 - y1

            scale_x = target_w / w
            scale_y = target_h / h
            scale = min(scale_x, scale_y)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = F.resize(img, (new_h, new_w))
            paste_x = x1 + (target_w - new_w) // 2
            paste_y = y1 + (target_h - new_h) // 2
            mosaic_img.paste(resized, (paste_x, paste_y))

            if bboxes.numel() > 0:
                scaled_bboxes = bboxes.clone().float()
                scaled_bboxes[:, [0, 2]] = scaled_bboxes[:, [0, 2]] * scale + paste_x
                scaled_bboxes[:, [1, 3]] = scaled_bboxes[:, [1, 3]] * scale + paste_y
                merged_bboxes.append(scaled_bboxes)
                merged_labels.append(labels)

        if merged_bboxes:
            merged_bboxes = torch.cat(merged_bboxes, dim=0)
            merged_labels = torch.cat(merged_labels, dim=0)
        else:
            merged_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            merged_labels = torch.zeros((0,), dtype=torch.int64)

        merged_bboxes[:, 0] = merged_bboxes[:, 0].clamp(min=0)
        merged_bboxes[:, 1] = merged_bboxes[:, 1].clamp(min=0)
        merged_bboxes[:, 2] = merged_bboxes[:, 2].clamp(max=s * 2)
        merged_bboxes[:, 3] = merged_bboxes[:, 3].clamp(max=s * 2)

        valid = (merged_bboxes[:, 2] > merged_bboxes[:, 0]) & (
            merged_bboxes[:, 3] > merged_bboxes[:, 1]
        )
        merged_bboxes = merged_bboxes[valid]
        merged_labels = merged_labels[valid]

        mosaic_img = F.resize(mosaic_img, (self.size, self.size))
        if merged_bboxes.numel() > 0:
            scale = self.size / (s * 2)
            merged_bboxes = merged_bboxes.float() * scale

        return mosaic_img, merged_bboxes, merged_labels


class CutMixAugment:
    def __init__(self, alpha=1.0, prob=0.3):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, img1, bboxes1, labels1, img2, bboxes2, labels2):
        if random.random() > self.prob:
            return img1, bboxes1, labels1

        lam = np.random.beta(self.alpha, self.alpha)
        w1, h1 = img1.size
        w2, h2 = img2.size

        target_w = min(w1, w2)
        target_h = min(h1, h2)

        cut_w = int(target_w * math.sqrt(1 - lam))
        cut_h = int(target_h * math.sqrt(1 - lam))

        cx = random.randint(0, target_w - 1)
        cy = random.randint(0, target_h - 1)

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, target_w)
        y2 = min(cy + cut_h // 2, target_h)

        img1_arr = np.array(img1)
        img2_arr = np.array(img2)

        img2_resized = np.array(F.resize(img2, (h1, w1)))
        img1_arr[y1:y2, x1:x2] = img2_resized[y1:y2, x1:x2]
        result_img = Image.fromarray(img1_arr)

        cut_box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

        if bboxes2.numel() > 0:
            scaled_b2 = bboxes2.clone().float()
            scale_x = w1 / w2
            scale_y = h1 / h2
            scaled_b2[:, [0, 2]] *= scale_x
            scaled_b2[:, [1, 3]] *= scale_y

            inside = self._boxes_inside_region(scaled_b2, x1, y1, x2, y2)
            clipped = self._clip_boxes(scaled_b2[inside], x1, y1, x2, y2)
            clipped_labels = labels2[inside]

            if clipped.numel() > 0:
                merged_bboxes = torch.cat([bboxes1, clipped], dim=0)
                merged_labels = torch.cat([labels1, clipped_labels], dim=0)
            else:
                merged_bboxes = bboxes1
                merged_labels = labels1
        else:
            merged_bboxes = bboxes1
            merged_labels = labels1

        return result_img, merged_bboxes, merged_labels

    @staticmethod
    def _boxes_inside_region(boxes, x1, y1, x2, y2):
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        return (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)

    @staticmethod
    def _clip_boxes(boxes, x1, y1, x2, y2):
        if boxes.numel() == 0:
            return boxes
        clipped = boxes.clone()
        clipped[:, 0] = clipped[:, 0].clamp(min=x1)
        clipped[:, 1] = clipped[:, 1].clamp(min=y1)
        clipped[:, 2] = clipped[:, 2].clamp(max=x2)
        clipped[:, 3] = clipped[:, 3].clamp(max=y2)
        return clipped


class TrainTransform:
    def __init__(self, image_size=800, hflip=True, color_jitter=True):
        self.image_size = image_size
        self.hflip = hflip
        self.color_jitter = color_jitter

    def __call__(self, img, bboxes, labels):
        if self.color_jitter and random.random() < 0.5:
            img = F.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = F.adjust_contrast(img, random.uniform(0.8, 1.2))
            img = F.adjust_saturation(img, random.uniform(0.8, 1.2))

        if self.hflip and random.random() < 0.5:
            img = F.hflip(img)
            if bboxes.numel() > 0:
                w = img.size[0]
                bboxes = bboxes.clone()
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        img = F.resize(img, (self.image_size, self.image_size))

        orig_w, orig_h = img.size
        if bboxes.numel() > 0 and (orig_w != self.image_size or orig_h != self.image_size):
            pass

        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, bboxes, labels


class ValTransform:
    def __init__(self, image_size=800):
        self.image_size = image_size

    def __call__(self, img, bboxes, labels):
        img = F.resize(img, (self.image_size, self.image_size))
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img, bboxes, labels
