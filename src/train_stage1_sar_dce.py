#!/usr/bin/env python3
"""
Stage-1 Training: DCE-YOLOv8m auf SARD + HERIDAL + VisDrone

Reproduziert das Stage-1-Modell:
  experiments/stage1_sar/dce_yolov8m_composite_run/weights/best.pt
das du später als Basis für Stage-2 genutzt hast.
"""


"""
python src/train_stage1_sar_dce.py \
  --model-yaml cfg/yolov8m_dce.yaml \
  --data cfg/sar_composite.yaml \
  --epochs 100 \
  --batch 24 \
  --imgsz 1280 \
  --device 0
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-yaml",
        default="cfg/yolov8m_dce.yaml",
        help="DCE-YOLOv8m Architektur-Config",
    )
    parser.add_argument(
        "--data",
        default="cfg/sar_composite.yaml",
        help="Composite YAML (SARD + HERIDAL + VisDrone)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--project", default="experiments/stage1_sar")
    parser.add_argument("--name", default="dce_yolov8m_composite_run")
    parser.add_argument("--device", default="0", help="z.B. 0, 1 oder 'cpu'")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional: Stage-1 Checkpoint (best/last) zum Fortsetzen",
    )
    
    args = parser.parse_args()

    # -------------------------------
    # Modell laden
    # -------------------------------
    if args.resume_from:
        ckpt = Path(args.resume_from)
        if not ckpt.exists():
            raise FileNotFoundError(f"Stage-1 Checkpoint nicht gefunden: {ckpt}")
        print(f"🔄 Resuming Stage-1 von Checkpoint: {ckpt}")
        model = YOLO(ckpt)
        resume_flag = True
        pretrained_arg = None
    else:
        print(
            f"📦 Starte Stage-1 von DCE-Config {args.model_yaml} "
            f"mit pretrained=yolov8m.pt"
        )
        model = YOLO(args.model_yaml)
        resume_flag = False
        pretrained_arg = "yolov8m.pt"

    # -------------------------------
    # Training starten (Stage-1)
    # -------------------------------
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        resume=resume_flag,

        # Pretrained nur beim Erststart
        pretrained=pretrained_arg,

        # Optimizer / LR wie in deinen Logs
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5.0,
        patience=30,
        weight_decay=0.0005,
        momentum=0.937,

        # Augmentations (aus dem Stage-1 Log):
        auto_augment="randaugment",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.6,
        fliplr=0.5,
        flipud=0.1,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        erasing=0.4,

        # Sonstiges
        seed=0,
        deterministic=True,
        save=True,
        plots=True,
        exist_ok=True,
        workers=8,
    )

    out_dir = Path(args.project) / args.name / "weights"
    print("✅ Stage-1 Training abgeschlossen.")
    print("   Checkpoints unter:", out_dir)
    print("   -> best.pt = Stage-1 Basis für Stage-2")


if __name__ == "__main__":
    main()
