#!/usr/bin/env python3
"""
Stage-1 Training: DCE-YOLOv8m on VisDrone (Personenklassse)

- Architektur: cfg/yolov8m_dce.yaml (DCE, ERB, SCDown etc.)
- Optionales Warmstarten aus COCO-Weights (z. B. yolov8m.pt)
- Single-Class (person) Training auf VisDrone
"""

from pathlib import Path
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from config import PROJECT_ROOT, DEVICE


class DCEVisDroneTrainer:
    """
    Stage-1 Trainer für VisDrone mit DCE-YOLOv8m Architektur.

    - Nutzt ein Custom-YAML (z. B. cfg/yolov8m_dce.yaml)
    - Optionales Warmstarten aus COCO-pretrained yolov8m.pt
    - Single-Class Personendetektion auf VisDrone
    """

    def __init__(
        self,
        model_yaml: str = "cfg/yolov8m_dce.yaml",
        pretrained: str | None = None,
        experiment_name: str = "stage1_dce_visdrone",
    ) -> None:
        self.model_yaml = model_yaml
        self.pretrained = Path(pretrained).resolve() if pretrained else None
        self.device = DEVICE

        self.exp_dir = self._setup_experiment_dir(experiment_name)
        self.models_dir = self.exp_dir / "models"
        self.plots_dir = self.exp_dir / "plots"
        self.logs_dir = self.exp_dir / "logs"
        self.configs_dir = self.exp_dir / "configs"

        print("=" * 70)
        print("DCE-YOLOv8m Stage-1 VisDrone Trainer")
        print("=" * 70)
        print(f"Model YAML: {self.model_yaml}")
        if self.pretrained:
            print(f"Pretrained weights: {self.pretrained}")
        else:
            print("Pretrained weights: None (training from scratch)")
        print(f"Device: {self.device}")
        print(f"Experiment dir: {self.exp_dir}")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # Infrastruktur
    # -------------------------------------------------------------------------

    def _setup_experiment_dir(self, experiment_name: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = PROJECT_ROOT / "experiments" / f"{experiment_name}_{ts}"
        for sub in ["models", "plots", "logs", "configs"]:
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _save_config(self, stage: str, cfg: dict):
        """Save training configuration for reproducibility."""
        config_file = self.exp_dir / "configs" / f"{stage}_config.json"
        # Nicht-JSON-Objekte (torch.device, Path, ...) via str() speichern
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, default=str)
        print(f"[config] Saved config: {config_file}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    def _plot_training_curves(self, results_csv: Path, target_map50: float = 0.65) -> None:
        if not results_csv.exists():
            print(f"[WARN] results.csv not found: {results_csv}")
            return

        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Stage1 DCE-VisDrone Training Progress", fontsize=16, fontweight="bold")

        # Box-Loss
        if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
            axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="train", linewidth=2)
            axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="val", linewidth=2)
            axes[0, 0].set_title("Box loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

        # mAP@0.5
        if "metrics/mAP50(B)" in df.columns:
            axes[0, 1].plot(
                df["epoch"],
                df["metrics/mAP50(B)"],
                linewidth=2,
                marker="o",
                markersize=3,
            )
            axes[0, 1].axhline(
                y=target_map50,
                color="green",
                linestyle="--",
                label=f"target {target_map50:.2f}",
            )
            axes[0, 1].set_title("mAP@0.5 (val)")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylim(0, 1.0)
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)

        # Precision / Recall
        if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
            axes[1, 0].plot(
                df["epoch"],
                df["metrics/precision(B)"],
                label="precision",
                linewidth=2,
            )
            axes[1, 0].plot(
                df["epoch"],
                df["metrics/recall(B)"],
                label="recall",
                linewidth=2,
            )
            axes[1, 0].set_title("Precision / Recall (val)")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylim(0, 1.0)
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        # Lernrate
        if "lr/pg0" in df.columns:
            axes[1, 1].plot(df["epoch"], df["lr/pg0"], linewidth=2)
            axes[1, 1].set_title("Learning rate (pg0)")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_yscale("log")
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        out_path = self.plots_dir / "stage1_dce_visdrone_curves.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] training curves saved to {out_path}")

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(
        self,
        data_yaml: str = "cfg/visdrone_person.yaml",
        epochs: int = 50,      # <--- neu: 50
        batch: int = 4,        # <--- neu: 4 als Standard für RTX 5000
        imgsz: int = 800,
        patience: int = 20,    # <--- neu: 20
    ) -> Path:
        """
        Stage-1 Training auf VisDrone mit DCE-YOLOv8m.

        Hyperparameter basieren auf der Konfiguration, mit der dein
        nicht-DCE-VisDrone-Baseline-Modell entstanden ist, übertragen
        auf die DCE-Architektur, mit leicht reduzierter LR und Augmentation
        zur Stabilisierung.
        """
        print("\n" + "=" * 70)
        print("Stage-1 DCE-VisDrone training")
        print("=" * 70)
        print(f"epochs={epochs}, batch={batch}, imgsz={imgsz}, patience={patience}")
        print(f"data={data_yaml}")
        print()

        train_args = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": self.device,

            # Optimizer / LR-Schedule (etwas konservativer)
            "optimizer": "AdamW",
            "lr0": 0.005,          # <--- reduziert von 0.01
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 2.0,  # <--- 3 -> 2
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": True,
            "patience": patience,

            # Logging / Saving
            "val": True,
            "plots": True,
            "save": True,
            "save_period": 10,

            # Augmentierung (leicht entschärft)
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.35,        # <--- 0.4 -> 0.35
            "degrees": 10.0,
            "translate": 0.2,
            "scale": 0.9,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.9,        # <--- 1.0 -> 0.9
            "mixup": 0.10,        # <--- 0.15 -> 0.10
            "copy_paste": 0.20,   # <--- 0.3 -> 0.2
            "auto_augment": None,
            "erasing": 0.0,

            # Mosaic etwas früher schließen
            "close_mosaic": 15,   # <--- 20 -> 15

            # Loss-Gewichte
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,

            # NMS / Eval
            "iou": 0.5,
            "conf": 0.001,
            "max_det": 300,
            "single_cls": True,

            # Performance
            "workers": 4,
            "cache": False,
            "rect": False,

            # Projektstruktur
            "project": str(self.models_dir),
            "name": "stage1_dce_visdrone",
            "exist_ok": True,
        }

        # Konfiguration speichern (inkl. YAML & Pretrained-Pfad)
        self._save_config(
            "stage1_dce_visdrone",
            {
                "model_yaml": self.model_yaml,
                "pretrained": str(self.pretrained) if self.pretrained else None,
                "train_args": train_args,
            },
        )

        # Modell aufbauen
        model = YOLO(self.model_yaml)
        if self.pretrained is not None:
            # Warmstart: versucht, kompatible Gewichte aus yolov8m.pt zu übernehmen
            print(f"[INIT] loading pretrained weights from {self.pretrained}")
            model.load(str(self.pretrained))

        # Training
        results = model.train(**train_args)

        # Bestes Checkpoint bestimmen
        best_model = Path(train_args["project"]) / train_args["name"] / "weights" / "best.pt"

        # Finales VisDrone-Val (Standard-Konfiguration)
        print("\n" + "-" * 70)
        print("Final validation on VisDrone (conf=0.25, iou=0.5)")
        print("-" * 70)
        val_results = model.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch,
            conf=0.25,
            iou=0.5,
            device=self.device,
            plots=True,
            save_json=True,
        )

        metrics = {
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "fitness": float(val_results.fitness),
        }

        metrics_path = self.logs_dir / "stage1_dce_visdrone_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"stage1_dce_visdrone": {"visdrone": metrics}}, f, indent=2)

        print("\nSummary metrics (VisDrone):")
        for k, v in metrics.items():
            print(f"  {k:>10s}: {v:.4f}")

        # Trainingskurven plotten
        results_csv = Path(train_args["project"]) / train_args["name"] / "results.csv"
        self._plot_training_curves(results_csv, target_map50=0.65)

        print("\nStage-1 DCE-VisDrone training complete.")
        print(f"Best checkpoint: {best_model}")
        print(f"Metrics JSON:  {metrics_path}")
        print(f"Plots:         {self.plots_dir}")
        print()
        return best_model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train DCE-YOLOv8m Stage-1 on VisDrone")
    parser.add_argument(
        "--model-yaml",
        type=str,
        default="cfg/yolov8m_dce.yaml",
        help="Pfad zum DCE-YOLOv8 Modell-YAML",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional: COCO-Weights (z. B. yolov8m.pt) für Warmstart",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="cfg/visdrone_person.yaml",
        help="VisDrone-Dataset YAML",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="stage1_dce_visdrone",
        help="Experiment-Name (Präfix für Ordner)",
    )
    parser.add_argument("--epochs", type=int, default=50)   # <--- 50
    parser.add_argument("--batch", type=int, default=4)     # <--- 4
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--patience", type=int, default=20) # <--- 20

    args = parser.parse_args()

    trainer = DCEVisDroneTrainer(
        model_yaml=args.model_yaml,
        pretrained=args.pretrained,
        experiment_name=args.experiment,
    )

    trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
