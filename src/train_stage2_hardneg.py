#!/usr/bin/env python3
"""
Stage-2 Training: Hard-Negative Finetune (DCE-YOLOv8m)
- Startet von Stage-1 best.pt
- Trainiert kurz auf Hard-Negatives + ein bisschen echte Positives
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--project", default="experiments/stage2_hardneg")
    parser.add_argument("--name", default="dce_yolov8m_hardneg_run")

    # optional: Stage-2 selbst wieder aufnehmen (falls die 5 Epochen mal abbrechen)
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
            f"Training wird von diesem Punkt fortgesetzt "
            f"(neue {args.epochs} Epochen auf Hard-Negativ-Set)."
        )
        # Stage-2-Checkpoint direkt laden
        model = YOLO(ckpt)
    else:
        ckpt = Path(args.pretrained)
        if not ckpt.exists():
            raise FileNotFoundError(f"Pretrained Checkpoint nicht gefunden: {ckpt}")
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
        resume=False,   # GANZ wichtig: neues Stage-2-Experiment, kein Stage-1-Resume

        # Feines Finetuning -> kleinere LR, sanfte Augmentierung
        optimizer="AdamW",
        lr0=0.0005,     # kleiner als Stage-1
        warmup_epochs=1.0,
        patience=10,    # kurz, wir trainieren nur 5 Epochen

        # leichte Augmentierung
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        device=0,
        save=True,
        plots=True,
        exist_ok=True,
    )

    print("✅ Stage-2 Hard-Negative-Finetune abgeschlossen.")
    print("   Ergebnisse unter:", Path(args.project) / args.name)


if __name__ == "__main__":
    main()
