import random
import shutil
from pathlib import Path

# ---------------------------------------------------------
# Pfade anpassen, falls nötig
# ---------------------------------------------------------

# Referenzdatensätze
DATA_ROOT = Path(r"D:\data")
VISDRONE_ROOT = DATA_ROOT / "VisdroneYOLO"
HERIDAL_ROOT = DATA_ROOT / "HERIDALYOLO"
SARD_ROOT = DATA_ROOT / "SARDYOLO"

# NTUT (voller Datensatz, YOLO-Struktur)
NTUT_SRC_ROOT = DATA_ROOT / "NTUT4KYOLO"

# Ziel: verkleinerter NTUT-Datensatz mit gleicher YOLO-Struktur
NTUT_DST_ROOT = DATA_ROOT / "NTUT4KYOLO_reduced"

SPLITS = ["train", "val"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(42)  # reproduzierbares Sampling


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def list_images(img_dir: Path):
    if not img_dir.exists():
        return []
    return [
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def list_labels(lbl_dir: Path):
    if not lbl_dir.exists():
        return []
    return [p for p in lbl_dir.rglob("*.txt") if p.is_file()]


def count_split(root: Path, split: str):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    imgs = list_images(img_dir)
    lbls = list_labels(lbl_dir)
    return len(imgs), len(lbls)


def count_dataset(root: Path, name: str):
    print(f"=== {name} ===")
    total_imgs = 0
    total_lbls = 0
    per_split = {}
    for split in SPLITS:
        n_img, n_lbl = count_split(root, split)
        per_split[split] = (n_img, n_lbl)
        total_imgs += n_img
        total_lbls += n_lbl
        print(f"  {split:5s} -> images: {n_img:6d} | labels: {n_lbl:6d}")
    print(f"  TOTAL -> images: {total_imgs:6d} | labels: {total_lbls:6d}\n")
    return {
        "name": name,
        "root": root,
        "per_split": per_split,
        "total_images": total_imgs,
        "total_labels": total_lbls,
    }


# ---------------------------------------------------------
# 1) Referenz-Datensätze zählen: Visdrone, HERIDAL, SARD
# ---------------------------------------------------------

vis_info = count_dataset(VISDRONE_ROOT, "VisdroneYOLO")
her_info = count_dataset(HERIDAL_ROOT, "HERIDALYOLO")
sard_info = count_dataset(SARD_ROOT, "SARDYOLO")

ref_datasets = [vis_info, her_info, sard_info]

# Kleinsten Referenzdatensatz (nach Gesamtbildern) finden
min_ref = min(ref_datasets, key=lambda d: d["total_images"])
target_total_images = min_ref["total_images"]

print("=== Referenz-Zusammenfassung ===")
for d in ref_datasets:
    print(f"{d['name']:12s} -> total images: {d['total_images']:6d}")
print(f"\nKleinster Referenz-Datensatz: {min_ref['name']} "
      f"mit {target_total_images} Bildern (train+val)\n")


# ---------------------------------------------------------
# 2) NTUT-Quelle zählen
# ---------------------------------------------------------

print("=== NTUT4KYOLO (voll) ===")
ntut_info = count_dataset(NTUT_SRC_ROOT, "NTUT4KYOLO")

ntut_train_imgs = ntut_info["per_split"]["train"][0]
ntut_val_imgs = ntut_info["per_split"]["val"][0]
ntut_total_imgs = ntut_info["total_images"]

if ntut_total_imgs == 0:
    raise RuntimeError("NTUT4KYOLO scheint keine Bilder zu enthalten!")

if target_total_images > ntut_total_imgs:
    raise RuntimeError(
        f"Referenz-Ziel ({target_total_images} Bilder) ist größer als NTUT-Gesamt "
        f"({ntut_total_imgs}). Sampling nicht möglich."
    )

# Verhältnis train/val aus NTUT behalten
train_ratio = ntut_train_imgs / ntut_total_imgs if ntut_total_imgs > 0 else 0.8
desired_train = int(round(target_total_images * train_ratio))
desired_val = target_total_images - desired_train

# Sicherstellen, dass wir nicht mehr ziehen als vorhanden
if desired_train > ntut_train_imgs:
    desired_train = ntut_train_imgs
    desired_val = target_total_images - desired_train
if desired_val > ntut_val_imgs:
    desired_val = ntut_val_imgs
    desired_train = target_total_images - desired_val

print("=== Sampling-Plan für NTUT4KYOLO_reduced ===")
print(f"NTUT gesamt:  train={ntut_train_imgs}, val={ntut_val_imgs}, total={ntut_total_imgs}")
print(f"Ziel gesamt:  {target_total_images} Bilder")
print(f"Geplantes Sampling: train={desired_train}, val={desired_val}")
print()

# ---------------------------------------------------------
# 3) NTUT4KYOLO_reduced erzeugen
# ---------------------------------------------------------

# Zielstruktur anlegen
for sub in ["images", "labels"]:
    for split in SPLITS:
        (NTUT_DST_ROOT / sub / split).mkdir(parents=True, exist_ok=True)

def sample_and_copy_split(split: str, n_target: int):
    src_img_dir = NTUT_SRC_ROOT / "images" / split
    src_lbl_dir = NTUT_SRC_ROOT / "labels" / split

    dst_img_dir = NTUT_DST_ROOT / "images" / split
    dst_lbl_dir = NTUT_DST_ROOT / "labels" / split

    all_images = list_images(src_img_dir)
    n_available = len(all_images)
    print(f"[{split}] verfügbare Bilder: {n_available}, zu kopieren: {n_target}")

    if n_target <= 0 or n_available == 0:
        print(f"[{split}] Nichts zu kopieren.")
        return 0, 0

    if n_target > n_available:
        print(f"[WARN] {split}: Zielanzahl {n_target} > verfügbar {n_available}, "
              f"reduziere auf {n_available}")
        n_target = n_available

    sampled = random.sample(all_images, n_target)

    copied_imgs = 0
    copied_lbls = 0

    for img_path in sampled:
        rel = img_path.relative_to(src_img_dir)

        dst_img_path = dst_img_dir / rel
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_img_path)
        copied_imgs += 1

        # Label-Pfad mit gleicher relativer Struktur
        label_rel = rel.with_suffix(".txt")
        src_label_path = src_lbl_dir / label_rel
        if src_label_path.exists():
            dst_label_path = dst_lbl_dir / label_rel
            dst_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_label_path, dst_label_path)
            copied_lbls += 1
        else:
            # Falls NTUT reine Negative hat, ist es okay, wenn kein Label existiert.
            # Optionaler Hinweis:
            # print(f"[INFO] Kein Label für {img_path.name} gefunden.")
            pass

    print(f"[{split}] kopiert -> images: {copied_imgs}, labels: {copied_lbls}\n")
    return copied_imgs, copied_lbls


print("=== Kopiere Subset von NTUT4KYOLO nach NTUT4KYOLO_reduced ===")
cop_train_imgs, cop_train_lbls = sample_and_copy_split("train", desired_train)
cop_val_imgs, cop_val_lbls = sample_and_copy_split("val", desired_val)

# ---------------------------------------------------------
# 4) Abschließende Zählung des neuen NTUT-Datensatzes
# ---------------------------------------------------------

print("=== NTUT4KYOLO_reduced (final) ===")
reduced_info = count_dataset(NTUT_DST_ROOT, "NTUT4KYOLO_reduced")

print("\nFERTIG.")
print(f"Referenz-Zielbilder (kleinster Datensatz): {target_total_images}")
print(f"NTUT4KYOLO_reduced total images:          {reduced_info['total_images']}")
print(f"Neuer Datensatz-Root: {NTUT_DST_ROOT}")
