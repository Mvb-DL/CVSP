from __future__ import annotations
from pathlib import Path
import argparse, random, shutil

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists(): return p
    return None

def main():
    ap = argparse.ArgumentParser("Make Okutama val from train (move subset with GT)")
    ap.add_argument("--root", type=Path, required=True, help="D:/data/OkutamaYOLO")
    ap.add_argument("--ratio", type=float, default=0.15, help="ratio of positive (non-empty) labels to move to val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lbl_tr = args.root / "labels" / "train"
    img_tr = args.root / "images" / "train"
    lbl_va = args.root / "labels" / "val"
    img_va = args.root / "images" / "val"
    for d in (lbl_va, img_va): d.mkdir(parents=True, exist_ok=True)

    # sammle positive und negative train-labels
    pos, neg = [], []
    for p in lbl_tr.glob("*.txt"):
        if p.read_text(encoding="utf-8", errors="ignore").strip():
            pos.append(p)
        else:
            neg.append(p)

    if not pos:
        print("[okutama][resplit] No positive labels in train — abort.")
        return

    random.Random(args.seed).shuffle(pos)
    n_val = max(100, int(len(pos) * args.ratio))  # Ziel: genug GTs
    sel_pos = pos[:n_val]

    # optional: etwas negatives Material mitziehen (ähnliche Menge)
    random.Random(args.seed).shuffle(neg)
    sel_neg = neg[:n_val]

    moved = 0
    for lbl in sel_pos + sel_neg:
        stem = lbl.stem
        img = find_image(img_tr, stem)
        if img is None:
            continue
        # move (verschieben, nicht kopieren → keine Data-Leaks)
        shutil.move(str(lbl), str(lbl_va / lbl.name))
        shutil.move(str(img), str(img_va / img.name))
        moved += 1

    print(f"[okutama][resplit] moved files: {moved} (labels+images)")
    # Hinweis: übrig gebliebene leere .jpg ohne .txt im val gibt es nicht, wir haben Paare verschoben.

if __name__ == "__main__":
    main()
