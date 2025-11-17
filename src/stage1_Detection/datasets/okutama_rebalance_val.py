from __future__ import annotations
from pathlib import Path
import argparse, random, shutil

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def nonempty(p: Path) -> bool:
    try:
        return bool(p.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser("Rebalance Okutama val: move positives (and optional negatives) from train -> val")
    ap.add_argument("--root", type=Path, required=True, help="z.B. D:/data/OkutamaYOLO")
    ap.add_argument("--target-pos", type=int, default=400, help="Zielanzahl nicht-leerer Labels in val")
    ap.add_argument("--neg-mult", type=float, default=1.0, help="wie viele negative pro verschobenem positiven (0=keine)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lbl_tr = args.root / "labels" / "train"
    img_tr = args.root / "images" / "train"
    lbl_va = args.root / "labels" / "val"
    img_va = args.root / "images" / "val"
    lbl_va.mkdir(parents=True, exist_ok=True)
    img_va.mkdir(parents=True, exist_ok=True)

    # status
    val_txts = list(lbl_va.glob("*.txt"))
    val_pos = sum(1 for p in val_txts if nonempty(p))
    print(f"[rb] current val positives: {val_pos}, target: {args.target_pos}")

    if val_pos >= args.target_pos:
        print("[rb] nothing to do.")
        return

    # sammle train-Kandidaten
    tr_pos = [p for p in lbl_tr.glob("*.txt") if nonempty(p)]
    tr_neg = [p for p in lbl_tr.glob("*.txt") if not nonempty(p)]
    random.Random(args.seed).shuffle(tr_pos)
    random.Random(args.seed).shuffle(tr_neg)

    need_pos = max(0, args.target_pos - val_pos)
    sel_pos = tr_pos[:need_pos]
    sel_neg = tr_neg[: int(len(sel_pos) * args.neg_mult)]

    moved = 0
    for lbl in sel_pos + sel_neg:
        stem = lbl.stem
        img = find_image(img_tr, stem)
        if img is None:
            continue
        # verschieben (keine Duplikate)
        shutil.move(str(lbl), str(lbl_va / lbl.name))
        shutil.move(str(img), str(img_va / img.name))
        moved += 1

    # report neu zählen
    val_txts = list(lbl_va.glob("*.txt"))
    val_pos2 = sum(1 for p in val_txts if nonempty(p))
    print(f"[rb] moved pairs: {moved}")
    print(f"[rb] new val positives: {val_pos2} (files in val: {len(val_txts)})")

if __name__ == "__main__":
    main()
