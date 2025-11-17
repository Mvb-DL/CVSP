# src/prepare_uavdt_mot.py
from __future__ import annotations
import argparse, os, re, json, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dst: Path, mode: str = "hardlink"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            if not dst.exists():
                os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def read_image_size(p: Path) -> Tuple[int,int]:
    with Image.open(p) as im:
        return im.width, im.height

def seq_from_json_or_name(stem: str, ann_obj: dict) -> str:
    # 1) JSON-Tag "sequence"
    try:
        for t in ann_obj.get("tags", []):
            if t.get("name") == "sequence" and t.get("value"):
                return str(t["value"])
    except Exception:
        pass
    # 2) Prefix wie M0101_img000001 → M0101
    m = re.match(r"^([A-Za-z0-9]+)_img\d+", stem)
    if m:
        return m.group(1)
    # 3) Fallback: kompletter Stem
    return stem

def frame_index_from_name(stem: str) -> int:
    # ..._img000123 → 123
    m = re.search(r"_img(\d+)$", stem)
    if m:
        return int(m.group(1))
    # generischer Fallback: letzte Zahl
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0

def bbox_from_exterior(points: dict) -> Optional[Tuple[float,float,float,float]]:
    ext = points.get("exterior")
    if not ext or len(ext) < 2:
        return None
    (x1,y1), (x2,y2) = ext[0], ext[1]
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    # normalisieren, falls vertauscht
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return x1, y1, w, h

def find_target_id(tags: list) -> Optional[int]:
    for t in tags or []:
        if t.get("name") == "target id":
            try:
                return int(t.get("value"))
            except Exception:
                return None
    return None

def parse_ann_json(ann_path: Path, keep_classes: set[str]) -> Dict:
    """
    Gibt Dict mit:
      seq: str
      frame: int
      size: (W,H)
      gts: List[(id, x, y, w, h)]
    """
    obj = json.loads(ann_path.read_text(encoding="utf-8", errors="ignore"))
    stem = ann_path.stem  # z.B. M0101_img000001
    seq = seq_from_json_or_name(stem, obj)
    frame = frame_index_from_name(stem)
    # Bildgröße
    W = obj.get("size", {}).get("width")
    H = obj.get("size", {}).get("height")
    if not W or not H:
        # notfalls aus Bild lesen (gleichnamige Bilddatei liegt in img/…)
        # Der Aufrufer kann Größe nachträglich befüllen.
        W = H = None

    gts = []
    for o in obj.get("objects", []):
        cls = str(o.get("classTitle", "")).strip().lower()
        if keep_classes and cls not in keep_classes:
            continue
        if o.get("geometryType") != "rectangle":
            continue
        tid = find_target_id(o.get("tags", []))
        if tid is None or tid <= 0:
            # ohne Track-ID nicht MOT-auswertbar → überspringen
            continue
        box = bbox_from_exterior(o.get("points", {}))
        if not box:
            continue
        x, y, w, h = box
        gts.append((tid, x, y, w, h))

    return {"seq": seq, "frame": frame, "size": (W, H), "gts": gts}

def discover_images(img_root: Path) -> List[Path]:
    # akzeptiere verschachtelt und flach
    if any(d.is_dir() for d in img_root.iterdir()):
        imgs = [p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        imgs = [p for p in img_root.glob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)

def read_fps_from_meta(meta_root: Path, seq: str) -> int:
    # Suche nach JSON/TXT in meta, die 'fps' o. ä. enthält
    for p in list(meta_root.glob(f"{seq}.*")) + list(meta_root.glob(f"**/{seq}.*")):
        try:
            if p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                for k in ("fps","frameRate","framerate","frame_rate"):
                    if k in obj:
                        v = int(round(float(obj[k])))
                        return max(1, min(v, 120))
            else:
                txt = p.read_text(encoding="utf-8", errors="ignore").lower()
                m = re.search(r"(fps|framerate|frame_rate)\D+(\d+(\.\d+)?)", txt)
                if m:
                    return max(1, min(int(round(float(m.group(2)))), 120))
        except Exception:
            pass
    return 30

def write_seqinfo(out_dir: Path, seq: str, first_img: Path, W: int, H: int, fps: int, length: int):
    ini = (
        "[Sequence]\n"
        f"name={seq}\n"
        "imDir=img1\n"
        f"frameRate={fps}\n"
        f"seqLength={length}\n"
        f"imWidth={W}\n"
        f"imHeight={H}\n"
        f"imExt={first_img.suffix.lower()}\n"
    )
    (out_dir / "seqinfo.ini").write_text(ini, encoding="utf-8")

def convert_split(raw_split: Path, out_split: Path, copy_mode: str, keep_classes: set[str]):
    img_root = raw_split / "img"
    ann_root = raw_split / "ann"
    meta_root = raw_split / "meta"

    if not img_root.exists() or not ann_root.exists():
        print(f"[WARN] missing 'img' or 'ann' under {raw_split}")
        return

    images = discover_images(img_root)
    if not images:
        print(f"[WARN] no images found in {img_root}")
        return

    # Schritt 1: Frame-Infos sammeln (seq, frame, size, gts)
    seq_frames: Dict[str, List[Dict]] = {}
    img_by_stem = {p.stem: p for p in images}
    ann_files = list(ann_root.rglob("*.json"))

    # Falls 1:1 Mapping Bild:JSON existiert, ist ann_root groß.
    # Wir iterieren über JSONs; falls JSON fehlt, nur Bild kopieren (gt leer).
    for ann in sorted(ann_files):
        stem = ann.stem
        info = parse_ann_json(ann, keep_classes)
        if stem not in img_by_stem:
            # JSON ohne Bild? Überspringen.
            continue
        info["img_path"] = img_by_stem[stem]
        seq_frames.setdefault(info["seq"], []).append(info)

    # Bilder ohne JSON (kein GT):
    stems_with_ann = set([a.stem for a in ann_files])
    for stem, img_path in img_by_stem.items():
        if stem in stems_with_ann:
            continue
        # fülle Minimalinfo
        m = re.match(r"^([A-Za-z0-9]+)_img(\d+)$", stem)
        seq = m.group(1) if m else "SEQ_unk"
        frame = int(m.group(2)) if m else 0
        W, H = read_image_size(img_path)
        seq_frames.setdefault(seq, []).append({
            "seq": seq, "frame": frame, "size": (W,H), "gts": [], "img_path": img_path
        })

    # Schritt 2: MOT-Struktur pro Sequenz schreiben
    for seq, items in seq_frames.items():
        # sortiere nach Original-Frame-Index
        items.sort(key=lambda d: d["frame"])
        out_seq = out_split / seq
        img1 = out_seq / "img1"
        gt_dir = out_seq / "gt"
        ensure_dir(img1); ensure_dir(gt_dir)

        # Größe/FPS bestimmen
        # nimm erste mit valider size, sonst aus Bild lesen
        W = H = None
        for it in items:
            W, H = it["size"]
            if W and H: break
        if not W or not H:
            W, H = read_image_size(items[0]["img_path"])
        fps = read_fps_from_meta(meta_root, seq)

        # Mapping alter frame → neuer frame (1..N)
        frame_map: Dict[int,int] = {}
        for new_idx, it in enumerate(items, start=1):
            old = it["frame"]
            frame_map[old] = new_idx
            dst = img1 / f"{new_idx:06d}{it['img_path'].suffix.lower()}"
            safe_copy(it["img_path"], dst, copy_mode)

        # GT schreiben
        gt_lines: List[str] = []
        kept = 0
        for it in items:
            newf = frame_map[it["frame"]]
            for (tid, x, y, w, h) in it["gts"]:
                gt_lines.append(f"{newf},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
                kept += 1
        (gt_dir / "gt.txt").write_text("".join(gt_lines), encoding="utf-8")

        # seqinfo.ini
        write_seqinfo(out_seq, seq, img1 / f"{1:06d}{items[0]['img_path'].suffix.lower()}", W, H, fps, len(items))

        print(f"[ok] {out_seq} (frames={len(items)}, fps={fps}, gt={'yes' if kept>0 else 'no'}, kept_boxes={kept})")

def main():
    ap = argparse.ArgumentParser(description="UAVDT (DatasetNinja JSON) → MOTChallenge Converter")
    ap.add_argument("--raw", type=Path, required=True, help="UAVDT root with train/test and ann/img/meta")
    ap.add_argument("--out", type=Path, required=True, help="Output root for MOT structure")
    ap.add_argument("--copy-mode", choices=["hardlink","copy"], default="hardlink")
    ap.add_argument("--splits", nargs="+", default=["train","test"])
    ap.add_argument("--keep-classes", nargs="+", default=["person"], help="Classes to keep (case-insensitive)")
    args = ap.parse_args()

    keep = set([c.strip().lower() for c in args.keep_classes])

    for sp in args.splits:
        raw_split = args.raw / sp
        out_split = args.out / sp
        out_split.mkdir(parents=True, exist_ok=True)
        convert_split(raw_split, out_split, args.copy_mode, keep)

    print("\n[done] UAVDT → MOT conversion finished.")

if __name__ == "__main__":
    main()
