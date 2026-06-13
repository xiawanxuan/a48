import os
import json
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        objects.append({
            "name": name,
            "difficult": difficult,
            "bbox": [xmin, ymin, xmax, ymax],
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def split_dataset(
    dataset_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
):
    random.seed(seed)

    images_dir = os.path.join(dataset_root, "images")
    annotations_dir = os.path.join(dataset_root, "annotations")

    if not os.path.exists(images_dir):
        images_dir = os.path.join(dataset_root, "JPEGImages")
    if not os.path.exists(annotations_dir):
        annotations_dir = os.path.join(dataset_root, "Annotations")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []
    for f in os.listdir(images_dir):
        if Path(f).suffix.lower() in image_exts:
            image_files.append(f)

    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }

    output_dir = os.path.join(dataset_root, "splits")
    os.makedirs(output_dir, exist_ok=True)

    for split_name, file_list in splits.items():
        split_data = []
        for img_file in file_list:
            img_stem = Path(img_file).stem
            xml_path = os.path.join(annotations_dir, img_stem + ".xml")

            entry = {"image": os.path.join(images_dir, img_file)}

            if os.path.exists(xml_path):
                annotation = parse_voc_annotation(xml_path)
                entry["width"] = annotation["width"]
                entry["height"] = annotation["height"]
                entry["objects"] = annotation["objects"]
            else:
                entry["objects"] = []

            split_data.append(entry)

        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

        print(f"{split_name}: {len(file_list)} images -> {output_path}")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    split_dataset(
        args.root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
