from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ------------------------
# Logging
# ------------------------

def make_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"prepare_external_{ts}.log"

    logger = logging.getLogger("prep")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging to {log_file}")
    return logger


# ------------------------
# Utilities
# ------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PERSON_SYNONYMS = {"person", "human", "rescuer", "pedestrian", "people", "man", "woman"}

def gather_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def split_train_val(items: List[Path], val_ratio=0.2, seed=42) -> Tuple[List[Path], List[Path]]:
    r = random.Random(seed)
    items = list(items)
    r.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]

def im_size_pil(path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image  # optional
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return (0, 0)

def yolo_line_from_xywh(xc, yc, w, h, W, H):
    return f"0 {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}\n"

def safe_copy(src: Path, dst: Path, logger: logging.Logger, mode: str = "copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "symlink":
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        elif mode == "hardlink":
            try:
                os.link(src, dst)
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
    except Exception as e:
        logger.warning(f"Copy fallback for {src} -> {dst} due to: {e}")
        try:
            shutil.copy2(src, dst)
        except Exception as e2:
            logger.error(f"Failed to copy {src} -> {dst}: {e2}")

def write_yaml(yaml_path: Path, dataset_root: Path):
    """
    Write a classic YOLO data YAML that points to a single dataset root (absolute).
    """
    content = (
        f"path: {dataset_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: person\n"
    )
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(content, encoding="utf-8")

def ensure_unique_name(dst_dir: Path, name: str) -> str:
    stem = Path(name).stem
    ext = Path(name).suffix
    candidate = name
    i = 1
    while (dst_dir / candidate).exists():
        candidate = f"{stem}_{i}{ext}"
        i += 1
    return candidate


# ------------------------
# Format detection
# ------------------------

def detect_format(root: Path, logger: logging.Logger) -> str:
    """
    Returns one of: 'roboflow_yolo', 'plain_yolo', 'coco', 'voc', 'negative'
    """
    data_yaml = list(root.glob("**/data.yaml"))
    if data_yaml:
        try:
            y = _read_yaml_like(data_yaml[0])
            if "train" in y or "val" in y:
                logger.info("Detected Roboflow-style YOLO (data.yaml present).")
                return "roboflow_yolo"
        except Exception:
            pass

    has_yolo_labels = any("labels" in p.parts and p.suffix.lower() == ".txt" for p in root.rglob("*.txt"))
    if has_yolo_labels:
        logger.info("Detected plain YOLO labels.")
        return "plain_yolo"

    jsons = list(root.rglob("*.json"))
    for j in jsons:
        try:
            obj = json.loads(j.read_text(encoding="utf-8", errors="ignore"))
            if all(k in obj for k in ("images", "annotations", "categories")):
                logger.info(f"Detected COCO annotations: {j}")
                return "coco"
        except Exception:
            continue

    if any(p.suffix.lower() == ".xml" for p in root.rglob("*.xml")):
        logger.info("Detected VOC annotations (XML).")
        return "voc"

    logger.info("No annotations found — treating as negative-only.")
    return "negative"


def _read_yaml_like(path: Path) -> Dict:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


# ------------------------
# Converters
# ------------------------

@dataclass
class ConvertConfig:
    person_name: str = "person"
    val_ratio: float = 0.2
    copy_mode: str = "copy"  # 'copy' | 'hardlink' | 'symlink'
    yaml_dir: Path | None = None  # where to place YAMLs (e.g., PROJECT_ROOT/cfg)

def convert_roboflow_yolo(src: Path, out: Path, cfg: ConvertConfig, logger: logging.Logger):
    logger.info(f"[roboflow] Source: {src}")
    data_yaml = next(iter(src.glob("**/data.yaml")), None)
    class_idx = 0
    if data_yaml and data_yaml.exists():
        y = _read_yaml_like(data_yaml)
        names_line = y.get("names", "")
        if names_line:
            if "person" in names_line.lower():
                class_idx = 0
    logger.info(f"[roboflow] Using person class index: {class_idx}")

    def find_split_dirs(name: str) -> Tuple[Optional[Path], Optional[Path]]:
        imgs = next(iter(src.glob(f"**/{name}/images")), None)
        lbls = next(iter(src.glob(f"**/{name}/labels")), None)
        return imgs, lbls

    splits = {"train": find_split_dirs("train"),
              "val": find_split_dirs("valid") or find_split_dirs("val"),
              "test": find_split_dirs("test")}

    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def copy_split(split: str, src_imgs: Optional[Path], src_lbls: Optional[Path], dst_split: str):
        if not src_imgs:
            logger.warning(f"[roboflow] Split {split}: no images dir found.")
            return
        imgs = [p for p in src_imgs.rglob("*") if p.suffix.lower() in IMG_EXTS]
        logger.info(f"[roboflow] Split {split}: {len(imgs)} images")
        for img in imgs:
            rel_name = ensure_unique_name((out / "images" / dst_split), img.name)
            dst_img = out / "images" / dst_split / rel_name
            safe_copy(img, dst_img, logger, cfg.copy_mode)

            lbl_src = None
            if src_lbls:
                cand = (src_lbls / img.with_suffix(".txt").name)
                if cand.exists():
                    lbl_src = cand

            dst_lbl = out / "labels" / dst_split / (Path(rel_name).with_suffix(".txt").name)
            if lbl_src and lbl_src.exists():
                lines = lbl_src.read_text(encoding="utf-8", errors="ignore").splitlines()
                kept = []
                for ln in lines:
                    pp = ln.strip().split()
                    if not pp:
                        continue
                    try:
                        cid = int(pp[0])
                    except ValueError:
                        continue
                    if cid == class_idx:
                        kept.append(" ".join(["0"] + pp[1:]) + "\n")
                dst_lbl.write_text("".join(kept), encoding="utf-8")
            else:
                dst_lbl.write_text("", encoding="utf-8")

    if splits["train"][0] and splits["val"][0]:
        copy_split("train", *splits["train"], "train")
        copy_split("val", *splits["val"], "val")
    else:
        logger.info("[roboflow] No explicit val split found -> creating 80/20 split from train.")
        imgs = [p for p in (splits["train"][0] or src).rglob("*") if p.suffix.lower() in IMG_EXTS]
        train, val = split_train_val(imgs, cfg.val_ratio, seed=42)
        lbls = splits["train"][1]
        for img in train:
            rel = ensure_unique_name((out / "images" / "train"), img.name)
            dst_img = out / "images" / "train" / rel
            safe_copy(img, dst_img, logger, cfg.copy_mode)
            dst_lbl = out / "labels" / "train" / (Path(rel).with_suffix(".txt").name)
            src_lbl = (lbls / img.with_suffix(".txt").name) if lbls else None
            if src_lbl and src_lbl.exists():
                lines = src_lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
                kept = []
                for ln in lines:
                    pp = ln.strip().split()
                    if pp and pp[0].isdigit() and int(pp[0]) == class_idx:
                        kept.append(" ".join(["0"] + pp[1:]) + "\n")
                dst_lbl.write_text("".join(kept), encoding="utf-8")
            else:
                dst_lbl.write_text("", encoding="utf-8")
        for img in val:
            rel = ensure_unique_name((out / "images" / "val"), img.name)
            dst_img = out / "images" / "val" / rel
            safe_copy(img, dst_img, logger, cfg.copy_mode)
            dst_lbl = out / "labels" / "val" / (Path(rel).with_suffix(".txt").name)
            src_lbl = (lbls / img.with_suffix(".txt").name) if lbls else None
            if src_lbl and src_lbl.exists():
                lines = src_lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
                kept = []
                for ln in lines:
                    pp = ln.strip().split()
                    if pp and pp[0].isdigit() and int(pp[0]) == class_idx:
                        kept.append(" ".join(["0"] + pp[1:]) + "\n")
                dst_lbl.write_text("".join(kept), encoding="utf-8")
            else:
                dst_lbl.write_text("", encoding="utf-8")

    assert cfg.yaml_dir is not None, "cfg.yaml_dir must be set"
    yaml_path = cfg.yaml_dir / f"{out.name.lower()}.yaml"
    write_yaml(yaml_path, out)
    logger.info(f"[roboflow] Wrote {yaml_path}")


def convert_plain_yolo(src: Path, out: Path, cfg: ConvertConfig, logger: logging.Logger):
    logger.info(f"[yolo] Source: {src}")
    image_dirs = [p for p in src.rglob("images") if p.is_dir()]
    label_dirs = [p for p in src.rglob("labels") if p.is_dir()]
    if not image_dirs:
        image_dirs = [src / "images"] if (src / "images").exists() else []
    image_dirs = sorted(image_dirs, key=lambda p: len(gather_images(p)), reverse=True)
    img_root = image_dirs[0] if image_dirs else src
    lbl_root = None
    if label_dirs:
        best = None; best_common = -1
        for L in label_dirs:
            common = len(os.path.commonprefix([str(L), str(img_root)]))
            if common > best_common:
                best = L; best_common = common
        lbl_root = best

    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    imgs = gather_images(img_root)
    if (img_root / "train").exists():
        splits = {"train": img_root / "train", "val": (img_root / "val" if (img_root / "val").exists() else img_root / "valid")}
        for split, sdir in splits.items():
            if not sdir or not sdir.exists():
                continue
            these = gather_images(sdir)
            logger.info(f"[yolo] Split {split}: {len(these)} images")
            for img in these:
                rel = ensure_unique_name((out / "images" / split), img.name)
                dst_img = out / "images" / split / rel
                safe_copy(img, dst_img, logger, cfg.copy_mode)
                dst_lbl = out / "labels" / split / (Path(rel).with_suffix(".txt").name)
                src_lbl = (lbl_root / split / img.with_suffix(".txt").name) if lbl_root else None
                # fallback location
                alt = (img.parent.parent / "labels" / split / img.with_suffix(".txt").name)
                real_lbl = src_lbl if (src_lbl and src_lbl.exists()) else (alt if alt.exists() else None)
                if real_lbl:
                    lines = real_lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
                    kept = []
                    for ln in lines:
                        parts = ln.strip().split()
                        if parts and parts[0].isdigit() and int(parts[0]) == 0:
                            kept.append(" ".join(["0"] + parts[1:]) + "\n")
                    dst_lbl.write_text("".join(kept), encoding="utf-8")
                else:
                    dst_lbl.write_text("", encoding="utf-8")
    else:
        train, val = split_train_val(imgs, cfg.val_ratio, seed=42)
        logger.info(f"[yolo] No explicit split -> train={len(train)} val={len(val)}")
        for split, subset in [("train", train), ("val", val)]:
            for img in subset:
                rel = ensure_unique_name((out / "images" / split), img.name)
                dst_img = out / "images" / split / rel
                safe_copy(img, dst_img, logger, cfg.copy_mode)
                dst_lbl = out / "labels" / split / (Path(rel).with_suffix(".txt").name)
                src_lbl = (img.parent.parent / "labels" / img.with_suffix(".txt").name)
                if src_lbl.exists():
                    lines = src_lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
                    kept = []
                    for ln in lines:
                        parts = ln.strip().split()
                        if parts and parts[0].isdigit() and int(parts[0]) == 0:
                            kept.append(" ".join(["0"] + parts[1:]) + "\n")
                    dst_lbl.write_text("".join(kept), encoding="utf-8")
                else:
                    dst_lbl.write_text("", encoding="utf-8")

    assert cfg.yaml_dir is not None
    yaml_path = cfg.yaml_dir / f"{out.name.lower()}.yaml"
    write_yaml(yaml_path, out)
    logger.info(f"[yolo] Wrote {yaml_path}")


def convert_coco(src: Path, out: Path, cfg: ConvertConfig, logger: logging.Logger):
    logger.info(f"[coco] Source: {src}")
    jsons = sorted([p for p in src.rglob("*.json") if any(k in p.name.lower() for k in ["annot", "instance", "train", "val"])])
    if not jsons:
        jsons = sorted(src.rglob("*.json"))
    if not jsons:
        logger.error("[coco] No JSON found.")
        return

    coco = json.loads(jsons[0].read_text(encoding="utf-8", errors="ignore"))
    cat_to_person = {c["id"]: (0 if c.get("name","").lower() in PERSON_SYNONYMS else -1) for c in coco.get("categories", [])}
    id2img = {}
    id2wh = {}

    root = jsons[0].parent
    for im in coco.get("images", []):
        f = root / im.get("file_name", "")
        if not f.exists():
            alt = src / im.get("file_name", "")
            f = alt if alt.exists() else f
        id2img[im["id"]] = f
        W, H = im.get("width", 0), im.get("height", 0)
        if not (W and H) and f.exists():
            W, H = im_size_pil(f)
        id2wh[im["id"]] = (W, H)

    by_img: Dict[int, List[str]] = {}
    for a in coco.get("annotations", []):
        cid = a.get("category_id")
        if cat_to_person.get(cid, -1) != 0:
            continue
        img_id = a.get("image_id")
        x, y, w, h = a.get("bbox", [0,0,0,0])
        W, H = id2wh.get(img_id, (0,0))
        if W <= 0 or H <= 0:
            continue
        line = yolo_line_from_xywh(x + w/2, y + h/2, w, h, W, H)
        by_img.setdefault(img_id, []).append(line)

    imgs = [p for p in id2img.values() if p and p.exists()]
    train, val = split_train_val(imgs, cfg.val_ratio, seed=42)
    for split, subset in [("train", train), ("val", val)]:
        for img in subset:
            img_id = next((k for k,v in id2img.items() if v == img), None)
            rel = ensure_unique_name((out / "images" / split), img.name)
            dst_img = out / "images" / split / rel
            safe_copy(img, dst_img, logger, cfg.copy_mode)
            lbl = out / "labels" / split / (Path(rel).with_suffix(".txt").name)
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lines = by_img.get(img_id, [])
            lbl.write_text("".join(lines), encoding="utf-8")

    assert cfg.yaml_dir is not None
    yaml_path = cfg.yaml_dir / f"{out.name.lower()}.yaml"
    write_yaml(yaml_path, out)
    logger.info(f"[coco] Wrote {yaml_path}")


def convert_voc(src: Path, out: Path, cfg: ConvertConfig, logger: logging.Logger):
    logger.info(f"[voc] Source: {src}")
    imgs = gather_images(src)
    train, val = split_train_val(imgs, cfg.val_ratio, seed=42)

    for split, subset in [("train", train), ("val", val)]:
        for img in subset:
            rel = ensure_unique_name((out / "images" / split), img.name)
            dst_img = out / "images" / split / rel
            safe_copy(img, dst_img, logger, cfg.copy_mode)
            W,H = im_size_pil(img)
            xml = img.with_suffix(".xml")
            dst_lbl = out / "labels" / split / (Path(rel).with_suffix(".txt").name)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            if xml.exists():
                try:
                    root = ET.parse(xml).getroot()
                    for obj in root.findall("object"):
                        name = (obj.findtext("name") or "").lower()
                        if name not in PERSON_SYNONYMS:
                            continue
                        b = obj.find("bndbox")
                        if b is None:
                            continue
                        xmin = float(b.findtext("xmin", "0")); ymin = float(b.findtext("ymin", "0"))
                        xmax = float(b.findtext("xmax", "0")); ymax = float(b.findtext("ymax", "0"))
                        w = xmax - xmin; h = ymax - ymin
                        lines.append(yolo_line_from_xywh(xmin+w/2, ymin+h/2, w, h, W, H))
                except Exception as e:
                    logger.warning(f"[voc] Failed XML parse {xml}: {e}")
            dst_lbl.write_text("".join(lines), encoding="utf-8")

    assert cfg.yaml_dir is not None
    yaml_path = cfg.yaml_dir / f"{out.name.lower()}.yaml"
    write_yaml(yaml_path, out)
    logger.info(f"[voc] Wrote {yaml_path}")


def convert_negative(src: Path, out: Path, cfg: ConvertConfig, logger: logging.Logger):
    logger.info(f"[neg] Source: {src}")
    imgs = gather_images(src)
    if not imgs:
        logger.warning("[neg] No images found.")
        return
    train, val = split_train_val(imgs, cfg.val_ratio, seed=42)
    for split, subset in [("train", train), ("val", val)]:
        for img in subset:
            rel = ensure_unique_name((out / "images" / split), img.name)
            dst_img = out / "images" / split / rel
            safe_copy(img, dst_img, logger, cfg.copy_mode)
            dst_lbl = out / "labels" / split / (Path(rel).with_suffix(".txt").name)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("", encoding="utf-8")

    assert cfg.yaml_dir is not None
    yaml_path = cfg.yaml_dir / f"{out.name.lower()}.yaml"
    write_yaml(yaml_path, out)
    logger.info(f"[neg] Wrote {yaml_path}")


# ------------------------
# Driver for one dataset
# ------------------------

def prepare_one(name: str, src: Path, out: Path, cfg: ConvertConfig, logger: logging.Logger):
    logger.info("=" * 80)
    logger.info(f"DATASET: {name}")
    logger.info(f"  source: {src}")
    logger.info(f"  output: {out}")
    logger.info("=" * 80)

    if not src.exists():
        logger.error(f"[{name}] Source folder not found: {src}")
        return

    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    fmt = detect_format(src, logger)

    try:
        if fmt == "roboflow_yolo":
            convert_roboflow_yolo(src, out, cfg, logger)
        elif fmt == "plain_yolo":
            convert_plain_yolo(src, out, cfg, logger)
        elif fmt == "coco":
            convert_coco(src, out, cfg, logger)
        elif fmt == "voc":
            convert_voc(src, out, cfg, logger)
        else:
            convert_negative(src, out, cfg, logger)
    except Exception as e:
        logger.exception(f"[{name}] Conversion failed: {e}")


def write_mix_yaml(dst_cfg: Path, yolo_dirs: Sequence[Path], logger: logging.Logger):
    # combine absolute folders for a single-person detector
    trains = [d / "images" / "train" for d in yolo_dirs if (d / "images" / "train").exists()]
    vals   = [d / "images" / "val"   for d in yolo_dirs if (d / "images" / "val").exists()]
    content = "train:\n" + "".join([f"  - {p.as_posix()}\n" for p in trains]) + \
              "val:\n"   + "".join([f"  - {p.as_posix()}\n" for p in vals]) + \
              "names:\n  0: person\n"
    dst_cfg.parent.mkdir(parents=True, exist_ok=True)
    dst_cfg.write_text(content, encoding="utf-8")
    logger.info(f"[mix] Wrote combined YAML: {dst_cfg}")


# ------------------------
# CLI
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="Prepare external SAR datasets into YOLOv8 (person-only).")
    ap.add_argument("--ssd-root", type=Path, default=Path(r"D:\data"),
                    help="SSD-Basis, wo ALLES liegen soll (Windows: D:\\data, Linux: /D/data).")
    ap.add_argument("--external-root", type=Path, default=None,
                    help="Rohdatenwurzel. Wenn leer/nicht vorhanden -> ssd-root/external.")
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1],
                    help="Projektwurzel (wo 'cfg' liegt).")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Val-Anteil, falls kein offizieller Split existiert.")
    ap.add_argument("--copy-mode", choices=["copy","hardlink","symlink"], default="copy",
                    help="Wie Bilder kopiert/verknüpft werden.")
    ap.add_argument("--person-name", default="person", help="Zielklassenname (meist 'person').")
    args = ap.parse_args()

    ssd_root: Path = args.ssd_root
    ssd_root.mkdir(parents=True, exist_ok=True)

    # Determine external root (where raw datasets are)
    if args.external_root is None:
        external_root = ssd_root / "external"
    else:
        external_root = args.external_root
        if not external_root.exists() and (ssd_root / "external").exists():
            external_root = ssd_root / "external"

    cfg_root  = args.project_root / "cfg"
    log_root  = args.project_root / "outputs" / "logs"

    logger = make_logger(log_root)
    logger.info(f"ssd-root     : {ssd_root}")
    logger.info(f"external-root: {external_root}")
    logger.info(f"project-root : {args.project_root}")
    logger.info(f"cfg-root     : {cfg_root}")

    # All prepared YOLO outputs will live on SSD
    data_root = ssd_root

    cfg = ConvertConfig(
        person_name=args.person_name,
        val_ratio=args.val_ratio,
        copy_mode=args.copy_mode,
        yaml_dir=cfg_root
    )

    # Quellen (Rohdaten) — Ordnernamen ggf. anpassen!
    sard_src    = external_root / "SARD"              # Kaggle/IEEE SARD
    heridal_src = external_root / "HERIDAL"           # Roboflow Export (YOLOv8 BBoxes)
    ntut_src    = external_root / "NTUT4K"            # meist negatives (nur Fotos)
    zenodo_src  = external_root / "ZENODO7740081"     # auto-detect

    # Ziele (konvertierte YOLO-Sets auf SSD!)
    sard_out    = data_root / "SARDYOLO"
    heridal_out = data_root / "HERIDALYOLO"
    ntut_out    = data_root / "NTUT4KYOLO"
    zenodo_out  = data_root / "ZENODOYOLO"

    # 1) SARD
    prepare_one("SARD", sard_src, sard_out, cfg, logger)

    # 2) HERIDAL
    prepare_one("HERIDAL", heridal_src, heridal_out, cfg, logger)

    # 3) NTUT-4K (negatives)
    prepare_one("NTUT4K", ntut_src, ntut_out, cfg, logger)

    # 4) Zenodo 7740081
    prepare_one("ZENODO7740081", zenodo_src, zenodo_out, cfg, logger)

    # 5) Kombiniertes SAR-YAML (zeigt auf ABSOLUTE SSD-Pfade)
    mix_yaml = cfg_root / "sar_person_mix.yaml"
    write_mix_yaml(mix_yaml, [sard_out, heridal_out, ntut_out, zenodo_out], logger)

    logger.info("All done ✔")


if __name__ == "__main__":
    main()
