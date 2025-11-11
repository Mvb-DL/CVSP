# src/prepare_visdrone_yolo.py
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image 

from src.config import DATA_ROOT, VISDRONE_ROOT


VISDRONE_YOLO_ROOT = DATA_ROOT / "VisdroneYOLO"

def find_annotation_file(ann_dir: Path, img_path: Path) -> Optional[Path]:

    candidates = [
        ann_dir / f"{img_path.name}.json",      
        ann_dir / f"{img_path.stem}.json",     
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_image_size(meta: Dict[str, Any], img_path: Path) -> Tuple[int, int]:

    for w_key in ["width", "img_width", "image_width", "w"]:
        for h_key in ["height", "img_height", "image_height", "h"]:
            if w_key in meta and h_key in meta:
                return int(meta[w_key]), int(meta[h_key])

    if "image" in meta and isinstance(meta["image"], dict):
        img_meta = meta["image"]
        if "width" in img_meta and "height" in img_meta:
            return int(img_meta["width"]), int(img_meta["height"])

    with Image.open(img_path) as im:
        w, h = im.size
    return w, h


def extract_person_boxes(meta: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:

    person_boxes: List[Tuple[float, float, float, float]] = []

    candidates = meta.get("annotations") or meta.get("objects") or meta.get("labels") or []

    PERSON_CATEGORIES = {"person", "pedestrian"}
    PERSON_CATEGORY_IDS = {1} 

    for ann in candidates:
        if not isinstance(ann, dict):
            continue

        cat = (
            ann.get("category")
            or ann.get("label")
            or ann.get("classTitle")
            or ann.get("class_name")
            or ann.get("name")
        )
        cat_id = ann.get("category_id") or ann.get("class_id") or ann.get("tag_id")

        is_person = False
        if isinstance(cat, str) and cat.lower() in PERSON_CATEGORIES:
            is_person = True
        if cat_id is not None:
            try:
                if int(cat_id) in PERSON_CATEGORY_IDS:
                    is_person = True
            except Exception:
                pass

        if not is_person:
            continue

        bbox = ann.get("bbox")

        if bbox is None:
            if {"x", "y", "w", "h"} <= ann.keys():
                bbox = [ann["x"], ann["y"], ann["w"], ann["h"]]
            elif {"x_min", "y_min", "x_max", "y_max"} <= ann.keys():
                x_min = ann["x_min"]
                y_min = ann["y_min"]
                x_max = ann["x_max"]
                y_max = ann["y_max"]
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        if bbox is None:
            points = None
            if "points" in ann and isinstance(ann["points"], dict):
                points = ann["points"].get("exterior") or ann["points"].get("points")
            if points and len(points) >= 2:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        if bbox is None or len(bbox) < 4:
            continue

        x, y, w, h = bbox[:4]
        person_boxes.append((float(x), float(y), float(w), float(h)))

    return person_boxes


def convert_split(split: str):

    src_img_dir = VISDRONE_ROOT / split / "img"
    src_ann_dir = VISDRONE_ROOT / split / "ann"

    if not src_img_dir.exists() or not src_ann_dir.exists():
        print(f"[visdrone_yolo][WARN] Split '{split}' not found (img or ann missing).")
        print(f"  img dir: {src_img_dir}")
        print(f"  ann dir: {src_ann_dir}")
        return

    dst_img_dir = VISDRONE_YOLO_ROOT / "images" / split
    dst_lbl_dir = VISDRONE_YOLO_ROOT / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(src_img_dir.glob("*.jpg"))
    print(f"[visdrone_yolo] Split '{split}': found {len(img_paths)} images.")

    for img_path in img_paths:
        dst_img_path = dst_img_dir / img_path.name
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)

        ann_json = find_annotation_file(src_ann_dir, img_path)
        if ann_json is None:
            print(f"[visdrone_yolo][WARN] No annotation JSON for {img_path.name}, skipping.")
            continue

        try:
            with ann_json.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[visdrone_yolo][WARN] Could not read JSON {ann_json.name}: {e}")
            continue

        img_w, img_h = extract_image_size(meta, img_path)
        boxes = extract_person_boxes(meta)

        dst_lbl_path = dst_lbl_dir / f"{img_path.stem}.txt"

        if not boxes:

            dst_lbl_path.touch()
            continue

        with dst_lbl_path.open("w", encoding="utf-8") as f:
            for (x, y, w, h) in boxes:

                x_c = (x + w / 2.0) / img_w
                y_c = (y + h / 2.0) / img_h
                w_r = w / img_w
                h_r = h / img_h

                f.write(f"0 {x_c:.6f} {y_c:.6f} {w_r:.6f} {h_r:.6f}\n")

    print(f"[visdrone_yolo] Finished split '{split}'. Labels in: {dst_lbl_dir}")


def main():
    print(f"[visdrone_yolo] VISDRONE_ROOT: {VISDRONE_ROOT}")
    print(f"[visdrone_yolo] YOLO output root: {VISDRONE_YOLO_ROOT}")

    convert_split("train")
    convert_split("val")

    print("[visdrone_yolo] Done.")


if __name__ == "__main__":
    main()
