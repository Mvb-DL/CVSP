#!/usr/bin/env python3
"""
Stage-2 Training: Hard-Negative Finetune (DCE-YOLOv8m)
- Startet von Stage-1 best.pt
- Trainiert auf Hard-Negatives + etwas echte Positives
- KEIN automatisches Resume aus Stage-1
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-yaml",
        default="cfg/yolov8m_dce.yaml",
        help="Model-Architektur (DCE-YOLOv8m)",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="Stage-1 Checkpoint (z.B. experiments/stage1_sar/.../best.pt)",
    )
    parser.add_argument(
        "--data",
        default="cfg/sar_hardneg_stage2.yaml",
        help="Hard-Negative Dataset YAML",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--project", default="experiments/stage2_hardneg")
    parser.add_argument("--name", default="dce_yolov8m_hardneg_run")
    parser.add_argument(
        "--device",
        default="0",
        help="Device für Training (z.B. 0, 1 oder 'cpu')",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Anzahl der ersten Layer, die eingefroren werden (0 = kein Freeze)",
    )

    # optional: Stage-2 selbst wieder aufnehmen (falls Stage-2 zwischendurch abbricht)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional: Stage-2 Checkpoint (best/last) für erneutes Fine-Tuning",
    )

    args = parser.parse_args()

    # -------------------------------
    # 1) Modell laden
    # -------------------------------
    if args.resume_from:
        ckpt = Path(args.resume_from)
        if not ckpt.exists():
            raise FileNotFoundError(f"Stage-2 Checkpoint nicht gefunden: {ckpt}")
        print(
            f"🔄 Stage-2 Checkpoint gefunden: {ckpt}. "
            f"Training wird von diesem Punkt fortgesetzt (neue {args.epochs} Epochen auf Stage-2-Set)."
        )
        # Stage-2-Checkpoint direkt laden
        model = YOLO(ckpt)
    else:
        ckpt = Path(args.pretrained)
        if not ckpt.exists():
            raise FileNotFoundError(f"Pretrained Stage-1 Checkpoint nicht gefunden: {ckpt}")
        print(f"📦 Starte Hard-Negative-Finetune von Stage-1 Modell: {ckpt}")
        # DCE-Architektur + Stage-1-Gewichte laden
        model = YOLO(args.model_yaml)
        model.load(ckpt)

    # -------------------------------
    # 2) Training starten (Stage-2)
    #    WICHTIG: KEIN resume=True!
    # -------------------------------
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        resume=False,        # ganz wichtig: neues Stage-2-Experiment, kein Stage-1-Resume
        device=args.device,

        # Feines Finetuning -> kleinere LR
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=3.0,
        patience=20,         # Early-Stopping für Stage-2
        weight_decay=0.0005,
        momentum=0.937,

        # <<< Paper-Style Augmentations für SAR >>>
        auto_augment=None,  # RandAugment explizit ausschalten
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.9,
        fliplr=0.5,
        flipud=0.0,
        degrees=15.0,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        erasing=0.0,

        # Backbone (weitgehend) einfrieren, damit Stage-1 nicht zerstört wird
        freeze=args.freeze,

        # Sonstiges
        save=True,
        plots=True,
        exist_ok=True,
        workers=8,
    )

    print("✅ Stage-2 Hard-Negative-Finetune abgeschlossen.")
    print("   Ergebnisse unter:", Path(args.project) / args.name)


if __name__ == "__main__":
    main()
