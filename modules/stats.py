import os
import json
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class DefectStats:
    def __init__(self, class_names):
        self.class_names = class_names
        self.class_to_idx = {name: idx + 1 for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx + 1: name for idx, name in enumerate(class_names)}
        self.reset()

    def reset(self):
        self._image_stats = []
        self._total_counts = OrderedDict()
        for name in self.class_names:
            self._total_counts[name] = 0

    def update_single(self, detections, image_id=None):
        counts = OrderedDict()
        for name in self.class_names:
            counts[name] = 0

        for det in detections:
            label = det.get("label", 0)
            if label in self.idx_to_class:
                name = self.idx_to_class[label]
                counts[name] += 1
            else:
                key = f"class_{label}"
                if key not in counts:
                    counts[key] = 0
                counts[key] += 1
                if key not in self._total_counts:
                    self._total_counts[key] = 0
                self._total_counts[key] += 1

        total = sum(counts.values())
        entry = {
            "image_id": image_id,
            "counts": dict(counts),
            "total": total,
        }
        self._image_stats.append(entry)

        for name, cnt in counts.items():
            if name in self._total_counts:
                self._total_counts[name] += cnt

        return entry

    def update_batch(self, batch_results):
        for item in batch_results:
            image_id = item.get("image", item.get("image_id", None))
            detections = item.get("detections", [])
            self.update_single(detections, image_id)

    def get_single_summary(self, detections, image_id=None):
        entry = self.update_single(detections, image_id)
        total = entry["total"]
        rows = []
        for cls_name, cnt in entry["counts"].items():
            ratio = (cnt / total * 100) if total > 0 else 0.0
            rows.append({
                "class": cls_name,
                "count": cnt,
                "ratio": round(ratio, 2),
            })
        return {
            "image_id": image_id,
            "total": total,
            "statistics": rows,
        }

    def get_batch_summary(self):
        grand_total = sum(self._total_counts.values())
        rows = []
        for cls_name, cnt in self._total_counts.items():
            ratio = (cnt / grand_total * 100) if grand_total > 0 else 0.0
            rows.append({
                "class": cls_name,
                "count": cnt,
                "ratio": round(ratio, 2),
            })
        return {
            "num_images": len(self._image_stats),
            "grand_total": grand_total,
            "statistics": rows,
            "per_image": self._image_stats,
        }

    def format_table(self, summary, title="Defect Statistics"):
        lines = []
        sep_width = 56

        lines.append("=" * sep_width)
        lines.append(f"  {title}")
        lines.append("=" * sep_width)

        header = f"{'Class':<20s} {'Count':>8s} {'Ratio':>10s} {'Bar':>15s}"
        lines.append(header)
        lines.append("-" * sep_width)

        stats = summary.get("statistics", [])
        for row in stats:
            bar_len = int(row["ratio"] / 100 * 15)
            bar = "█" * bar_len + "░" * (15 - bar_len)
            lines.append(
                f"{row['class']:<20s} {row['count']:>8d} {row['ratio']:>9.2f}% {bar:>15s}"
            )

        lines.append("-" * sep_width)
        total_key = "grand_total" if "grand_total" in summary else "total"
        total_val = summary.get(total_key, 0)
        num_img = summary.get("num_images", None)
        if num_img is not None:
            lines.append(f"{'TOTAL':<20s} {total_val:>8d} {'100.00%':>10s}")
            lines.append(f"{'Images':<20s} {num_img:>8d}")
        else:
            lines.append(f"{'TOTAL':<20s} {total_val:>8d} {'100.00%':>10s}")

        lines.append("=" * sep_width)
        return "\n".join(lines)

    def save_json(self, summary, output_path):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to {output_path}")
