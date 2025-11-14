# src/stage3_train_sar.py
from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import platform
from datetime import datetime

from ultralytics import YOLO
from src.config import PROJECT_ROOT, DEVICE


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _device_arg_from_torch_device(dev) -> int | str:
    if getattr(dev, "type", "") == "cuda":
        return dev.index if dev.index is not None else 0
    return "cpu"


def _resolve_val_dirs_from_yaml(yaml_path: Path) -> list[Path]:
    """
    Minimaler YAML-Reader für 'path:' und 'val:' (String oder Liste).
    Liefert eine Liste von Bild-Ordnern (kann auch mehrere sein).
    """
    txt = Path(yaml_path).read_text(encoding="utf-8")
    lines = txt.splitlines()

    root = None
    vals: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("path:"):
            root = Path(s.split(":", 1)[1].strip().strip("'\""))
        elif s.startswith("val:"):
            rest = s.split(":", 1)[1].strip()
            if rest and not rest.startswith("-"):
                vals.append(rest.strip("'\""))
            else:
                j = i + 1
                while j < len(lines) and lines[j].lstrip().startswith("-"):
                    v = lines[j].split("-", 1)[1].strip().strip("'\"")
                    vals.append(v)
                    j += 1
                i = j - 1
        i += 1

    out_dirs: list[Path] = []
    for v in vals:
        p = Path(v)
        if not p.is_absolute() and root is not None:
            p = root / v
        out_dirs.append(p)

    if not out_dirs and root is not None:
        out_dirs = [root / "images" / "val"]

    return out_dirs


def _fp_only_eval(m: YOLO, sources: list[Path], imgsz: int, conf: float, iou: float, device_arg) -> dict:
    """
    Fallback-Evaluation für Datensätze ohne Labels:
      - läuft alle Val-Bilder durch
      - zählt Vorhersagen (Boxen)
      - meldet avg_fp_per_image, total_preds, images, max_preds_single_image
    """
    total_imgs = 0
    total_preds = 0
    max_per_img = 0

    for src in sources:
        if not src.exists():
            continue
        preds_gen = m.predict(
            source=str(src),
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            stream=True,
            device=device_arg,
            verbose=False,
        )
        for r in preds_gen:
            n = 0 if getattr(r, "boxes", None) is None else int(len(r.boxes))
            total_imgs += 1
            total_preds += n
            if n > max_per_img:
                max_per_img = n

    avg_fp = total_preds / max(total_imgs, 1)
    return {
        "avg_fp_per_image": float(avg_fp),
        "images": int(total_imgs),
        "total_preds": int(total_preds),
        "max_preds_single_image": int(max_per_img),
    }


def _save_metrics(exp_dir: Path, name: str, metrics: dict):
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    outp = exp_dir / "logs" / f"{name}_{exp_dir.name}.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return outp


def _run_validation(weights_path: Path, args, device_arg, exp_dir: Path):
    print("\n[validation] Running validation …")
    m = YOLO(str(weights_path))
    try:
        # Standard-Validierung (funktioniert nur mit gelabelten Datasets)
        val = m.val(
            data=str(args.data),
            split="val",
            plots=bool(args.val_plots),
            batch=int(args.val_batch),
            workers=int(args.val_workers),
            imgsz=int(args.val_imgsz),
            conf=float(args.val_conf),
            iou=float(args.val_iou),
            device=device_arg,
        )

        # Robust prüfen, ob es GT-Labels gab
        p = getattr(val.box, "p", None)
        has_labels = False
        try:
            has_labels = p is not None and (len(p) > 0)  # numpy/torch arrays sind len()-fähig
        except Exception:
            pass
        if not has_labels:
            raise ValueError("No GT labels found in val set")

        metrics = {
            "precision": float(val.box.p[0]),
            "recall": float(val.box.r[0]),
            "mAP50": float(val.box.map50),
            "mAP50-95": float(val.box.map),
            "fitness": float(val.fitness),
        }
        outp = _save_metrics(exp_dir, "validation_metrics", metrics)
        print("[validation]", metrics)
        print(f"[validation] Saved to: {outp}")

    except Exception as e:
        # Fallback für reine Negativ-Datasets (z. B. NTUT4K)
        print(f"[validation] No-label dataset detected or validator failed ({e}). Falling back to FP-only eval.")
        val_dirs = _resolve_val_dirs_from_yaml(Path(args.data))
        metrics = _fp_only_eval(
            m,
            val_dirs,
            imgsz=int(args.val_imgsz),
            conf=float(args.val_conf),
            iou=float(args.val_iou),
            device_arg=device_arg,
        )
        outp = _save_metrics(exp_dir, "validation_metrics_FPonly", metrics)
        print("[validation][FP-only]", metrics)
        print(f"[validation] Saved to: {outp}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage-3 SAR Finetuning & Validation (YOLOv8m, RTX A2000-safe)")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "cfg" / "sar_person_mix.yaml")
    ap.add_argument("--weights", type=Path, required=True, help="Pfad zu best.pt oder last.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=6)     # RTX A2000-safe
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--resume", action="store_true", help="Resume from last.pt (behält Optimizer/EMA)")

    # Validation / Val-only
    ap.add_argument("--val-only", action="store_true", help="Nur validieren, nicht trainieren")
    ap.add_argument("--val-batch", type=int, default=2)
    ap.add_argument("--val-workers", type=int, default=0)
    ap.add_argument("--val-imgsz", type=int, default=800)
    ap.add_argument("--val-plots", action="store_true")
    ap.add_argument("--val-conf", type=float, default=0.25)
    ap.add_argument("--val-iou", type=float, default=0.50)

    args = ap.parse_args()

    # Experiment-Ordner (für unsere Reports/Kopien)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "VAL" if args.val_only else ""
    exp_dir = PROJECT_ROOT / "experiments" / f"yolov8m_sar_stage3_ft_{tag}_{ts}".strip("_")
    (exp_dir / "runs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "weights").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    # Device-Mapping
    device_arg = _device_arg_from_torch_device(DEVICE)

    # -------- Val-only Modus --------
    if args.val_only:
        print("\n" + "=" * 80)
        print("  🔎 LIGHT VALIDATION (no training)")
        print(f"  data : {args.data}")
        print(f"  model: {args.weights}")
        print(f"  imgsz: {args.val_imgsz} | batch: {args.val_batch} | workers: {args.val_workers} | plots: {bool(args.val_plots)}")
        print("=" * 80 + "\n")
        _run_validation(args.weights, args, device_arg, exp_dir)
        print("\n" + "=" * 80)
        print(f"✓ Validation results saved to: {exp_dir}")
        print("=" * 80 + "\n")
        return

    # --------------------------
    # TRAINING
    # --------------------------
    is_windows = platform.system().lower().startswith("win")

    # HERIDAL-sichere, konservative Augs/LR – aktiviert wenn im --data Pfad "heridal" vorkommt
    is_heridal = "heridal" in str(args.data).lower()

    lr0        = 0.002 if is_heridal else 0.003
    mosaic     = 0.0   if is_heridal else 0.15
    mixup      = 0.0
    copy_paste = 0.05  if is_heridal else 0.1
    degrees    = 6.0   if is_heridal else 8.0
    scale      = 0.90  if is_heridal else 0.85
    translate  = 0.15  if is_heridal else 0.20
    workers    = 2 if is_windows else 4
    batch      = args.batch

    # Trainingskonfig (für Nachvollziehbarkeit abspeichern)
    cfg = {
        "stage": "SAR-stage3-finetune",
        "data": str(args.data),
        "start_weights": str(args.weights),
        "epochs": args.epochs,
        "batch": batch,
        "imgsz": args.imgsz,
        "device": str(DEVICE),
        "workers": workers,
        "resume": bool(args.resume),
        "aug": {
            "hsv_h": 0.03, "hsv_s": 0.8, "hsv_v": 0.6,
            "degrees": degrees, "scale": scale, "translate": translate,
            "mosaic": mosaic, "mixup": mixup, "copy_paste": copy_paste, "close_mosaic": 3
        },
        "optim": {"opt": "AdamW", "lr0": lr0, "lrf": 0.01, "cos_lr": True,
                  "warmup_epochs": 1, "weight_decay": 5e-4, "patience": 8, "nbs": 64},
        "inference": {"iou": 0.5, "max_det": 150}
    }
    with open(exp_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\n" + "=" * 80)
    print("  🚀 STAGE-3 SAR FINETUNE (RTX A2000 safe)")
    print(f"  data:   {args.data}")
    print(f"  start:  {args.weights}")
    print(f"  epochs: {args.epochs} | imgsz: {args.imgsz} | batch: {batch} | workers: {workers} | resume: {bool(args.resume)}")
    print("=" * 80 + "\n")

    # Modell laden
    model = YOLO(str(args.weights))

    # Trainingsargumente aufbauen (Resume überschreibt project/name)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device_arg,
        workers=workers,
        amp=True,
        optimizer="AdamW",
        lr0=lr0, lrf=0.01, cos_lr=True, nbs=64,
        patience=8,
        warmup_epochs=1,
        warmup_momentum=0.5,
        weight_decay=0.0005,
        cache=False,  # RAM/VRAM-schonend
        hsv_h=0.03, hsv_s=0.8, hsv_v=0.6,
        degrees=degrees, flipud=0.0, fliplr=0.5,
        mosaic=mosaic, mixup=mixup, copy_paste=copy_paste,
        scale=scale, translate=translate, perspective=0.002,
        close_mosaic=3,
        classes=[0],
        iou=0.5, max_det=150,
        verbose=True,
        plots=True,
    )

    if args.resume:
        # Beim Resume NICHT project/name setzen – YOLO hängt an den Original-Run an
        train_kwargs["resume"] = True
    else:
        train_kwargs["project"] = str(exp_dir / "runs")
        train_kwargs["name"] = "train"
        train_kwargs["exist_ok"] = True

    # Training ausführen
    results = model.train(**train_kwargs)

    # Run-Verzeichnis herausfinden (robust für neue Runs und Resume)
    run_dir: Path
    try:
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        run_dir = Path(save_dir) if save_dir else (exp_dir / "runs" / "train")
    except Exception:
        run_dir = exp_dir / "runs" / "train"

    # Artefakte kopieren
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    # Einheitliche Namen für diese Session
    if best.exists():
        shutil.copy(best, exp_dir / "weights" / f"best_{exp_dir.name}.pt")
    if last.exists():
        shutil.copy(last, exp_dir / "weights" / f"last_{exp_dir.name}.pt")
    if (run_dir / "results.csv").exists():
        shutil.copy(run_dir / "results.csv", exp_dir / "logs" / f"results_{exp_dir.name}.csv")

    # --------------------------
    # VALIDATION nach dem Training
    # --------------------------
    best_path = exp_dir / "weights" / f"best_{exp_dir.name}.pt"
    if best_path.exists():
        # Val-Args für konsistentes Verhalten auch im Trainingslauf setzen
        class Obj:
            pass
        val_args = Obj()
        val_args.data = args.data
        val_args.val_plots = False
        val_args.val_batch = 2
        val_args.val_workers = 0
        val_args.val_imgsz = args.imgsz
        val_args.val_conf = 0.25
        val_args.val_iou = 0.50
        _run_validation(best_path, val_args, device_arg, exp_dir)
    else:
        print("[validation][WARN] best weights not found.")

    print("\n" + "=" * 80)
    print(f"✓ All results saved to: {exp_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
