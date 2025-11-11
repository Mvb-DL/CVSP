# src/datasets/uavdt.py
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterator


@dataclass
class UAVDTFrame:
    image_path: Path
    split: str  # 'train' oder 'test'


def iter_uavdt_split(root: Path, split: str = "train") -> Iterator[UAVDTFrame]:
    """
    root: Pfad zu 'data/UAVDT'
    Struktur:
      root/train/ann, root/train/img, root/train/meta
      root/test/ann,  root/test/img,  root/test/meta
    """
    if split not in {"train", "test"}:
        raise ValueError("split muss 'train' oder 'test' sein.")

    split_root = root / split
    img_dir = split_root / "img"

    if not img_dir.exists():
        raise FileNotFoundError(f"Bildordner nicht gefunden: {img_dir}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for img_path in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts):
        yield UAVDTFrame(image_path=img_path, split=split)


def get_uavdt_sample_images(root: Path, split: str = "train", max_images: int = 50) -> List[Path]:
    images: List[Path] = []
    for frame in iter_uavdt_split(root, split):
        images.append(frame.image_path)
        if len(images) >= max_images:
            break
    return images
