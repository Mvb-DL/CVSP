# src/inspect_datasets.py
from pathlib import Path
from collections import Counter
import re

from src.config import EXTERNAL_ROOT

CANDIDATES = {
    "zenodo":  EXTERNAL_ROOT / "zenodo7740081_yolo",
    "sard":    EXTERNAL_ROOT / "sard_yolo",
    "heridal": EXTERNAL_ROOT / "heridal_yolo",
    "ntut4k":  EXTERNAL_ROOT / "ntut4k_drone_photos",  # negatives only
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _count_yolo(ds_root: Path):
    out = {}
    for split in ["train", "val", "test"]:
        imgs = ds_root / "images" / split
        lbls = ds_root / "labels" / split
        if not imgs.exists():
            continue
        n_img = sum(1 for p in imgs.rglob("*") if p.suffix.lower() in IMG_EXTS)
        n_lbl = 0
        empty_lbl = 0
        classes = Counter()
        if lbls.exists():
            for t in lbls.rglob("*.txt"):
                txt = t.read_text(encoding="utf-8", errors="ignore").strip()
                if not txt:
                    empty_lbl += 1
                else:
                    n_lbl += 1
                    for line in txt.splitlines():
                        m = re.match(r"^\s*(\d+)\s+", line)
                        if m:
                            classes[int(m.group(1))] += 1
        out[split] = {
            "images": n_img, "labels_nonempty": n_lbl,
            "labels_empty": empty_lbl, "class_hist": dict(classes)
        }
    return out

def _count_negative_folder(folder: Path):
    if not folder.exists():
        return {"images": 0}
    n_img = sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return {"images": n_img}

def main():
    print("== Dataset inspection (external @ D:/data/external) ==\n")
    for name, root in CANDIDATES.items():
        print(f"[{name}] → {root}")
        if not root.exists():
            print("  ⚠️  Not found\n"); continue
        if (root / "images").exists() and (root / "labels").exists():
            stats = _count_yolo(root)
            for split, s in stats.items():
                print(f"  {split:5s}: images={s['images']:6d} | labels≠∅={s['labels_nonempty']:6d} | labels=∅={s['labels_empty']:6d} | classes={s['class_hist']}")
            print()
        else:
            neg = _count_negative_folder(root)
            print(f"  (negative-only images) count={neg['images']}\n")

if __name__ == "__main__":
    main()
