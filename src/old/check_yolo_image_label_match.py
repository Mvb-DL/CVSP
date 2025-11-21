#!/usr/bin/env python3
"""
Check YOLO datasets for matching images and labels.

Prüft für jeden angegebenen Dataset-Root:
- images/train <-> labels/train
- images/val   <-> labels/val

und gibt aus:
- Anzahl Bilder
- Anzahl Labels
- Bilder MIT Label
- Bilder OHNE Label
- Labels OHNE Bild
"""

from pathlib import Path

# Basis-Pfad anpassen, falls nötig
DATA_ROOT = Path(r"D:\data")

DATASETS = [
    "VisdroneYOLO",
    "HERIDALYOLO",
    "SARDYOLO",
    "NTUT4KYOLO",
    "NTUT4KYOLO_reduced",  # optional, wird übersprungen, wenn nicht vorhanden
]

SPLITS = ["train", "val"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(img_dir: Path):
    """Alle Bild-Dateien (rekursiv) unterhalb von img_dir sammeln."""
    if not img_dir.exists():
        return []
    return [
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def list_labels(lbl_dir: Path):
    """Alle Label-Dateien (rekursiv) unterhalb von lbl_dir sammeln."""
    if not lbl_dir.exists():
        return []
    return [p for p in lbl_dir.rglob("*.txt") if p.is_file()]


def check_split(dataset_root: Path, split: str):
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    imgs = list_images(img_dir)
    lbls = list_labels(lbl_dir)

    # Maps für schnellen Lookup: name ohne Suffix / mit rel-Pfad
    # Wir nehmen relative Pfade ab images/ bzw. labels/, damit auch Subfolder funktionieren.
    img_keys = {}
    for p in imgs:
        rel = p.relative_to(img_dir)
        key = rel.with_suffix("")  # "sub/xyz"
        img_keys[key] = p

    lbl_keys = {}
    for p in lbls:
        rel = p.relative_to(lbl_dir)
        key = rel.with_suffix("")  # "sub/xyz"
        lbl_keys[key] = p

    # Bilder mit/ohne Label
    images_with_label = []
    images_without_label = []
    for key, img_path in img_keys.items():
        if key in lbl_keys:
            images_with_label.append(img_path)
        else:
            images_without_label.append(img_path)

    # Labels ohne Bild
    labels_without_image = []
    for key, lbl_path in lbl_keys.items():
        if key not in img_keys:
            labels_without_image.append(lbl_path)

    return {
        "split": split,
        "num_images": len(imgs),
        "num_labels": len(lbls),
        "num_images_with_label": len(images_with_label),
        "num_images_without_label": len(images_without_label),
        "num_labels_without_image": len(labels_without_image),
        "images_without_label": images_without_label,
        "labels_without_image": labels_without_image,
    }


def print_split_report(dataset_name: str, split_result: dict):
    split = split_result["split"]
    print(f"  [{split}]")
    print(f"    images total:              {split_result['num_images']:6d}")
    print(f"    labels total:              {split_result['num_labels']:6d}")
    print(f"    images WITH label:         {split_result['num_images_with_label']:6d}")
    print(f"    images WITHOUT label:      {split_result['num_images_without_label']:6d}")
    print(f"    labels WITHOUT image:      {split_result['num_labels_without_image']:6d}")

    # Wenn du detaillierte Pfade sehen willst, kannst du das einkommentieren:
    # if split_result["num_images_without_label"] > 0:
    #     print("    -> Images ohne Label:")
    #     for p in split_result["images_without_label"]:
    #         print(f"       - {p}")
    # if split_result["num_labels_without_image"] > 0:
    #     print("    -> Labels ohne Bild:")
    #     for p in split_result["labels_without_image"]:
    #         print(f"       - {p}")


def main():
    for ds_name in DATASETS:
        root = DATA_ROOT / ds_name
        if not root.exists():
            print(f"=== {ds_name} (übersprungen – Ordner existiert nicht) ===\n")
            continue

        print(f"=== Dataset: {ds_name} ===")
        total_images = 0
        total_labels = 0
        total_img_wo_label = 0
        total_lbl_wo_img = 0

        for split in SPLITS:
            res = check_split(root, split)
            print_split_report(ds_name, res)

            total_images += res["num_images"]
            total_labels += res["num_labels"]
            total_img_wo_label += res["num_images_without_label"]
            total_lbl_wo_img += res["num_labels_without_image"]

        print(f"  TOTAL images:               {total_images:6d}")
        print(f"  TOTAL labels:               {total_labels:6d}")
        print(f"  TOTAL images WITHOUT label: {total_img_wo_label:6d}")
        print(f"  TOTAL labels WITHOUT image: {total_lbl_wo_img:6d}")
        print()

    print("Fertig überprüft.")


if __name__ == "__main__":
    main()
