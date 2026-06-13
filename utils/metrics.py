import torch
import numpy as np


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_precision(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def compute_recall(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def compute_map(all_aps):
    if len(all_aps) == 0:
        return 0.0
    return float(np.mean(all_aps))


def match_predictions(
    pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5
):
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    if num_preds == 0 and num_gts == 0:
        return 0, 0, 0, {}
    if num_preds == 0:
        return 0, 0, num_gts, {}
    if num_gts == 0:
        return 0, num_preds, 0, {}

    tp = 0
    fp = 0
    fn = 0

    gt_matched = [False] * num_gts
    class_stats = {}

    sorted_indices = sorted(range(num_preds), key=lambda i: -pred_scores[i])

    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_label = pred_labels[pred_idx]
        pred_score = pred_scores[pred_idx]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx in range(num_gts):
            if gt_matched[gt_idx]:
                continue
            if gt_labels[gt_idx] != pred_label:
                continue

            iou = compute_iou(pred_box, gt_boxes[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

        if pred_label not in class_stats:
            class_stats[pred_label] = {"tp": 0, "fp": 0, "scores": [], "matched_gt": 0}
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            class_stats[pred_label]["tp"] += 1
        else:
            class_stats[pred_label]["fp"] += 1
        class_stats[pred_label]["scores"].append(pred_score)

    fn = num_gts - sum(gt_matched)

    for gt_idx in range(num_gts):
        label = gt_labels[gt_idx]
        if label not in class_stats:
            class_stats[label] = {"tp": 0, "fp": 0, "scores": [], "matched_gt": 0}
        if not gt_matched[gt_idx]:
            class_stats[label]["matched_gt"] += 1

    return tp, fp, fn, class_stats


def compute_class_ap(
    all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels, class_id, iou_threshold=0.5
):
    all_scores = []
    all_tp = []
    total_gt = 0

    for pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels in zip(
        all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels
    ):
        class_gt_indices = [i for i, l in enumerate(gt_labels) if l == class_id]
        total_gt += len(class_gt_indices)

        class_pred_indices = [i for i, l in enumerate(pred_labels) if l == class_id]
        if len(class_pred_indices) == 0:
            continue

        sorted_indices = sorted(class_pred_indices, key=lambda i: -pred_scores[i])
        gt_matched = [False] * len(class_gt_indices)

        for pred_idx in sorted_indices:
            all_scores.append(pred_scores[pred_idx])
            pred_box = pred_boxes[pred_idx]

            best_iou = 0.0
            best_gt_local = -1

            for local_idx, gt_idx in enumerate(class_gt_indices):
                if gt_matched[local_idx]:
                    continue
                iou = compute_iou(pred_box, gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_local = local_idx

            if best_iou >= iou_threshold and best_gt_local >= 0:
                all_tp.append(1)
                gt_matched[best_gt_local] = True
            else:
                all_tp.append(0)

    if total_gt == 0:
        return 0.0

    if len(all_scores) == 0:
        return 0.0

    sorted_indices = np.argsort(-np.array(all_scores))
    tp_sorted = np.array(all_tp)[sorted_indices]

    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1 - tp_sorted)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    return compute_ap(recalls, precisions)
