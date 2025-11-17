# src/datasets/visdron.py
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterator, Optional


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int
    cls: int
    score: int
    truncation: int
    occlusion: int


@dataclass
class VisdronSample:
    image_path: Path
    ann_path: Optional[Path]
    bboxes: List[BBox]
    split: str  # 'train','val','test-dev','test-challenge'


def _split_line(line: str):
    line = line.strip()
    if not line:
        return []
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
    else:
        parts = line.split()
    return parts


def _parse_annotation_file(ann_path: Path) -> List[BBox]:
    """Parst eine VisDrone-Annotation (.txt) mit 8 Spalten."""
    bboxes: List[BBox] = []
    if not ann_path.exists():
        return bboxes

    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = _split_line(line)
            if len(parts) < 8:
                continue
            try:
                x = int(float(parts[0]))
                y = int(float(parts[1]))
                w = int(float(parts[2]))
                h = int(float(parts[3]))
                score = int(float(parts[4]))
                cls_id = int(float(parts[5]))
                trunc = int(float(parts[6]))
                occ = int(float(parts[7]))
            except ValueError:
                continue

            bboxes.append(
                BBox(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    cls=cls_id,
                    score=score,
                    truncation=trunc,
                    occlusion=occ,
                )
            )
    return bboxes


def iter_visdron_split(root: Path, split: str = "train") -> Iterator[VisdronSample]:
    """
    root: Pfad zu 'data/Visdron'
    split:
      'train'          -> 'train/img', 'train/ann'
      'val'            -> 'val/img', 'val/ann'
      'test-dev'       -> 'test dev/img', 'test dev/ann'
      'test-challenge' -> 'test challenge/img', 'test challenge/ann'
    """
    split_map = {
        "train": "train",
        "val": "val",
        "test": "test dev",
        "test-dev": "test dev",
        "dev": "test dev",
        "test-challenge": "test challenge",
        "challenge": "test challenge",
    }
    split_folder = split_map.get(split, split)
    split_root = root / split_folder

    img_dir = split_root / "img"
    ann_dir = split_root / "ann"

    if not img_dir.exists():
        raise FileNotFoundError(f"Bildordner nicht gefunden: {img_dir}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for img_path in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts):
        ann_path = ann_dir / (img_path.stem + ".txt")
        bboxes = _parse_annotation_file(ann_path) if ann_dir.exists() else []
        yield VisdronSample(
            image_path=img_path,
            ann_path=ann_path if ann_path.exists() else None,
            bboxes=bboxes,
            split=split,
        )


def get_visdron_sample_images(root: Path, split: str = "val", max_images: int = 50):
    """Gibt Pfade zu den ersten max_images Bildern zurück (für Basistests)."""
    images: List[Path] = []
    for sample in iter_visdron_split(root, split):
        images.append(sample.image_path)
        if len(images) >= max_images:
            break
    return images
