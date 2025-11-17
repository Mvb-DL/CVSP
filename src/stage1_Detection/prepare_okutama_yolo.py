from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse, re, shutil, random, os
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -----------------------------
# Utilities
# -----------------------------
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dst: Path, mode: str = "hardlink"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
            return
        except Exception:
            pass
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def im_wh(path: Path) -> Tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return (0, 0)

def yolo_line_from_xyxy(x1, y1, x2, y2, W, H):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2
    yc = y1 + h / 2
    if W <= 0 or H <= 0 or w <= 0 or h <= 0:
        return None
    return f"0 {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}\n"

def write_yaml(path: Path, root: Path):
    content = (
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: person\n"
    )
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")

def extract_frame_number(filename: str) -> Optional[int]:
    """Extract frame number from various naming patterns"""
    patterns = [
        r'frame[_-]?(\d+)',
        r'img[_-]?(\d+)',
        r'^(\d+)$',
        r'(\d{4,})',
    ]
    stem = Path(filename).stem
    for pattern in patterns:
        m = re.search(pattern, stem, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None

# -----------------------------
# Okutama parsing
# -----------------------------
SEQ_ID_RE = re.compile(r"(\d+\.\d+\.\d+)$")
RES_RE = re.compile(r"(\d+)[xX](\d+)")

@dataclass
class SeqInfo:
    seq_id: str
    split: str
    img_dir: Path
    images: List[Path]
    img_size: Tuple[int,int]
    label_file: Optional[Path]
    label_res: Optional[Tuple[int,int]]

def find_sequences(raw: Path) -> List[SeqInfo]:
    seqs: List[SeqInfo] = []

    def gather(base_split: Path, split_name: str):
        for p in base_split.rglob("*"):
            if not p.is_dir():
                continue
            if "Extracted-Frames" not in str(p):
                continue
            m = SEQ_ID_RE.search(str(p))
            if not m:
                continue
            seq_id = m.group(1)
            imgs = sorted([x for x in p.glob("*") if is_image(x)])
            if not imgs:
                continue
            W,H = im_wh(imgs[0])
            
            lbl = None
            lbl_res = None
            labels_root = None
            try:
                ef_idx = [i for i,part in enumerate(p.parts) if part.startswith("Extracted-Frames")]
                anchor_idx = ef_idx[0] if ef_idx else (len(p.parts)-1)
                root_for_labels = Path(*p.parts[:anchor_idx])
                while root_for_labels.name not in ("TrainSetFrames", "TestSetFrames") and root_for_labels.parent != root_for_labels:
                    root_for_labels = root_for_labels.parent
                labels_root = root_for_labels / "Labels"
            except Exception:
                labels_root = None

            cand_dirs = []
            if labels_root and labels_root.exists():
                for kind in ("SingleActionTrackingLabels", "MultiActionTrackingLabels"):
                    k = labels_root / kind
                    if k.exists():
                        cand_dirs.append(k)

            found = False
            for k in cand_dirs:
                for res_dir in k.iterdir():
                    if not res_dir.is_dir(): continue
                    lm = RES_RE.search(res_dir.name)
                    if not lm: continue
                    lw, lh = int(lm.group(1)), int(lm.group(2))
                    lf = res_dir / f"{seq_id}.txt"
                    if lf.exists() and lf.stat().st_size > 0:
                        lbl = lf
                        lbl_res = (lw, lh)
                        found = True
                        break
                if found: break

            seqs.append(SeqInfo(
                seq_id=seq_id,
                split=split_name,
                img_dir=p,
                images=imgs,
                img_size=(W,H),
                label_file=lbl,
                label_res=lbl_res
            ))

    train_root = raw / "TrainSetFrames"
    test_root  = raw / "TestSetFrames"
    if train_root.exists(): gather(train_root, "train")
    if test_root.exists():  gather(test_root,  "test")
    return seqs

def parse_sequence_labels(label_file: Path) -> Dict[int, List[Tuple[float,float,float,float]]]:
    """Returns: dict frame_idx -> list of (x1,y1,x2,y2)"""
    out: Dict[int, List[Tuple[float,float,float,float]]] = {}
    txt = read_text(label_file)
    if not txt: return out
    
    for line in txt.splitlines():
        s = line.strip()
        if not s: continue
        if '"Person"' not in s and "'Person'" not in s and "Person" not in s:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        if len(nums) < 5:
            continue
        try:
            frame = int(float(nums[0]))
            x1 = float(nums[1]); y1 = float(nums[2]); x2 = float(nums[3]); y2 = float(nums[4])
            if x2 <= x1 or y2 <= y1:
                continue
            out.setdefault(frame, []).append((x1,y1,x2,y2))
        except Exception:
            continue
    return out

def build_frame_mapping(images: List[Path], label_frames: List[int], seq_id: str) -> Dict[Path, int]:
    """
    Build mapping from image file -> label frame index.
    Tries multiple strategies to match image filenames to label indices.
    """
    mapping = {}
    
    img_frame_nums = []
    for img in images:
        num = extract_frame_number(img.name)
        img_frame_nums.append((img, num))
    
    label_set = set(label_frames)
    label_min = min(label_frames) if label_frames else 0
    label_max = max(label_frames) if label_frames else 0
    
    # Strategy 1: Direct match
    matches_direct = 0
    if all(num is not None for _, num in img_frame_nums):
        frame_nums = [num for _, num in img_frame_nums]
        matches_direct = sum(1 for fn in frame_nums if fn in label_set)
    
    # Strategy 2: Sequential 0-based
    matches_0based = sum(1 for i in range(len(images)) if i in label_set)
    
    # Strategy 3: Sequential 1-based
    matches_1based = sum(1 for i in range(len(images)) if (i+1) in label_set)
    
    # Strategy 4: Offset mapping
    matches_offset = 0
    offset_value = 0
    if label_min >= len(images):
        for offset in [0, label_min, label_min - len(images)]:
            matches = sum(1 for i in range(len(images)) if (i + offset) in label_set)
            if matches > matches_offset:
                matches_offset = matches
                offset_value = offset
    
    print(f"  Frame mapping: direct={matches_direct}, 0-based={matches_0based}, "
          f"1-based={matches_1based}, offset({offset_value})={matches_offset}")
    print(f"  Label range: {label_min}-{label_max}, Image count: {len(images)}")
    
    best_matches = max(matches_direct, matches_0based, matches_1based, matches_offset)
    
    if best_matches == 0:
        print(f"  WARNING: No frame mapping found!")
        if label_frames[:5]:
            print(f"  Label frames (first 5): {label_frames[:5]}")
        if images[:5]:
            print(f"  Image names (first 5): {[img.name for img in images[:5]]}")
        return mapping
    
    if matches_direct >= best_matches and all(num is not None for _, num in img_frame_nums):
        for img, num in img_frame_nums:
            if num in label_set:
                mapping[img] = num
    elif matches_offset >= best_matches:
        for i, img in enumerate(images):
            frame_idx = i + offset_value
            if frame_idx in label_set:
                mapping[img] = frame_idx
    elif matches_1based > matches_0based:
        for i, img in enumerate(images):
            if (i+1) in label_set:
                mapping[img] = i + 1
    else:
        for i, img in enumerate(images):
            if i in label_set:
                mapping[img] = i
    
    return mapping

# -----------------------------
# Conversion
# -----------------------------
def convert_okutama(raw: Path, out: Path, val_ratio: float = 0.2, copy_mode: str = "hardlink", min_pos_frames: int = 5):
    seqs = find_sequences(raw)
    print(f"[okutama] found sequences: {len(seqs)}")

    labeled: List[SeqInfo] = []
    for s in seqs:
        if not s.label_file:
            continue
        frames_boxes = parse_sequence_labels(s.label_file)
        n_boxes = sum(len(v) for v in frames_boxes.values())
        if n_boxes >= 1:
            labeled.append(s)

    print(f"[okutama] labeled sequences with boxes: {len(labeled)}")

    labeled_filtered: List[SeqInfo] = []
    for s in labeled:
        frames_boxes = parse_sequence_labels(s.label_file)
        pos_frames = [f for f, bb in frames_boxes.items() if len(bb) > 0]
        if len(pos_frames) >= min_pos_frames:
            labeled_filtered.append(s)
    if not labeled_filtered:
        labeled_filtered = labeled

    print(f"[okutama] labeled sequences (>= {min_pos_frames} pos frames): {len(labeled_filtered)}")

    random.Random(42).shuffle(labeled_filtered)
    n_val = max(1, int(len(labeled_filtered) * val_ratio))
    val_seqs = set(s.seq_id for s in labeled_filtered[:n_val])
    train_seqs = set(s.seq_id for s in labeled_filtered[n_val:])

    img_tr = out / "images" / "train"
    img_va = out / "images" / "val"
    lb_tr  = out / "labels" / "train"
    lb_va  = out / "labels" / "val"
    for d in (img_tr, img_va, lb_tr, lb_va):
        ensure_dir(d)

    total_imgs = 0
    total_lbls = 0
    val_pos    = 0
    train_pos  = 0

    for s in seqs:
        frames_boxes = {}
        if s.label_file:
            frames_boxes = parse_sequence_labels(s.label_file)

        if not frames_boxes:
            continue

        print(f"\n[seq {s.seq_id}] {len(s.images)} images, {len(frames_boxes)} labeled frames, {sum(len(b) for b in frames_boxes.values())} boxes")
        
        label_frame_list = sorted(frames_boxes.keys())
        frame_map = build_frame_mapping(s.images, label_frame_list, s.seq_id)

        if s.label_res:
            lw, lh = s.label_res
        else:
            lw, lh = s.img_size

        matched_frames = 0
        for img in s.images:
            label_frame_idx = frame_map.get(img)
            if label_frame_idx is None:
                continue

            boxes = frames_boxes.get(label_frame_idx, [])
            
            if not boxes:
                continue
            
            W, H = s.img_size

            target_img_dir = img_tr if s.seq_id in train_seqs else (img_va if s.seq_id in val_seqs else img_tr)
            target_lbl_dir = lb_tr  if s.seq_id in train_seqs else (lb_va  if s.seq_id in val_seqs else lb_tr)

            dst_img = target_img_dir / img.name
            safe_copy(img, dst_img, mode=copy_mode)
            total_imgs += 1

            dst_lbl = target_lbl_dir / (img.stem + ".txt")
            lines: List[str] = []
            for (x1, y1, x2, y2) in boxes:
                if lw > 0 and lh > 0 and (lw != W or lh != H):
                    sx = W / float(lw)
                    sy = H / float(lh)
                    xx1 = x1 * sx; yy1 = y1 * sy; xx2 = x2 * sx; yy2 = y2 * sy
                else:
                    xx1, yy1, xx2, yy2 = x1, y1, x2, y2
                ln = yolo_line_from_xyxy(xx1, yy1, xx2, yy2, W, H)
                if ln: lines.append(ln)
            
            dst_lbl.write_text("".join(lines), encoding="utf-8")
            matched_frames += 1
            if target_lbl_dir is lb_va:
                val_pos += 1
            else:
                train_pos += 1
            total_lbls += 1
        
        print(f"  -> Matched {matched_frames} frames with labels")

    print(f"\n[okutama] images copied: {total_imgs}, labels written: {total_lbls}")
    print(f"[okutama] train frames with persons: {train_pos}")
    print(f"[okutama] val frames with persons: {val_pos}")

    yaml_path = Path(__file__).resolve().parents[1] / "cfg" / "okutamayolo.yaml"
    write_yaml(yaml_path, out)
    print(f"[okutama] wrote YAML: {yaml_path}")

def main():
    ap = argparse.ArgumentParser("Prepare Okutama -> YOLOv8 (person-only) with sequence labels")
    ap.add_argument("--raw", type=Path, required=True, help=r"D:\data\Okutama")
    ap.add_argument("--out", type=Path, required=True, help=r"D:\data\OkutamaYOLO")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--copy-mode", choices=["hardlink","copy","symlink"], default="hardlink")
    ap.add_argument("--min-pos-frames", type=int, default=5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    convert_okutama(args.raw, args.out, val_ratio=args.val_ratio, copy_mode=args.copy_mode, min_pos_frames=args.min_pos_frames)

if __name__ == "__main__":
    main()