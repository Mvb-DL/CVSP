from pathlib import Path
import argparse
import json
import shutil
from datetime import datetime
from typing import Dict, Optional
import platform

from ultralytics import YOLO
import torch

# Handle imports
try:
    from src.config import PROJECT_ROOT, DEVICE, DATA_ROOT
except ModuleNotFoundError:
    from config import PROJECT_ROOT, DEVICE, DATA_ROOT


class Stage2Trainer:
    """
    Stage 2 SAR Fine-Tuning with DCE comparison

    Trains:
      - Stage 2A: Standard YOLOv8m (baseline)
      - Stage 2B: YOLOv8m + DCE modules (experimental)

    Typical usage:

      # Train both variants (standard + DCE) starting from Stage 1 VisDrone weights
      python src/stage2_with_dce_comparison.py ^
          --stage1-weights checkpoints/best.pt ^
          --train-both

      # Only standard
      python src/stage2_with_dce_comparison.py ^
          --stage1-weights checkpoints/best.pt ^
          --variant standard

      # Only DCE
      python src/stage2_with_dce_comparison.py ^
          --stage1-weights checkpoints/best.pt ^
          --variant dce
    """

    def __init__(self, stage1_weights: Path, experiment_name: str = "stage2_comparison"):
        self.stage1_weights = stage1_weights
        self.experiment_name = experiment_name
        self.device = DEVICE

        # Platform-specific settings
        self.is_windows = platform.system().lower().startswith("win")
        self.workers = 2 if self.is_windows else 4

        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = PROJECT_ROOT / "experiments" / f"{experiment_name}_{timestamp}"
        self.results_dir = self.exp_dir / "results"
        self.models_dir = self.exp_dir / "models"
        self.logs_dir = self.exp_dir / "logs"
        self.checkpoints_dir = PROJECT_ROOT / "checkpoints"

        for d in [self.results_dir, self.models_dir, self.logs_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🔬 Stage 2 DCE Comparison Initialized")
        print(f"{'='*80}")
        print(f"📁 Experiment: {self.exp_dir.name}")
        print(f"🖥️  Device: {self.device}")
        print(f"🏋️ Stage 1 Weights: {stage1_weights}")
        print(f"{'='*80}\n")

    def _device_arg(self):
        """Convert torch device from config to Ultralytics format"""
        if getattr(self.device, "type", "") == "cuda":
            return self.device.index if self.device.index is not None else 0
        return "cpu"

    # ======================================================================
    # STAGE 2A: STANDARD YOLOv8m (Baseline)
    # ======================================================================

    def train_standard(
        self,
        data_yaml: str = "cfg/sar_pos_only.yaml",
        epochs: int = 50,
        batch: int = 6,
        imgsz: int = 800,
        patience: int = 15
    ) -> Path:
        """
        Train Stage 2A: Standard YOLOv8m (baseline)

        This is your proven approach - no architectural changes.
        """
        print(f"\n{'='*80}")
        print("🎯 STAGE 2A: Standard YOLOv8m Fine-Tuning")
        print(f"{'='*80}")
        print(f"Strategy: Your proven SAR fine-tuning approach")
        print(f"Architecture: Standard YOLOv8m (no DCE modules)")
        print(f"Target: mAP50 > 0.70 (HERIDAL), > 0.80 (SARD)")
        print(f"{'='*80}\n")

        config = {
            "variant": "standard",
            "stage1_weights": str(self.stage1_weights),
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": str(self.device),
            "workers": self.workers,
            "optimizer": "AdamW",
            "lr0": 0.003,
            "lrf": 0.01,
            "patience": patience,
            "augmentation": {
                "hsv_h": 0.03,
                "hsv_s": 0.8,
                "hsv_v": 0.6,
                "degrees": 8.0,
                "translate": 0.2,
                "scale": 0.85,
                "mosaic": 0.15,
                "mixup": 0.0,
                "copy_paste": 0.1,
            }
        }

        # Save config
        with open(self.logs_dir / "stage2a_standard_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Load model (full Stage 1 checkpoint)
        model = YOLO(str(self.stage1_weights))

        print(f"🏋️ Starting Stage 2A training...\n")

        # Train
        _ = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self._device_arg(),
            workers=self.workers,

            # Optimizer
            optimizer="AdamW",
            lr0=0.003,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1.0,
            warmup_momentum=0.5,
            cos_lr=True,
            nbs=64,
            patience=patience,

            # Reduced augmentation
            hsv_h=0.03,
            hsv_s=0.8,
            hsv_v=0.6,
            degrees=8.0,
            translate=0.2,
            scale=0.85,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.15,
            mixup=0.0,
            copy_paste=0.1,
            perspective=0.002,
            close_mosaic=3,

            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,

            # Inference
            iou=0.5,
            max_det=150,
            classes=[0],

            # Performance
            cache=False,
            rect=False,
            single_cls=True,
            amp=True,

            # Project
            project=str(self.models_dir),
            name="stage2a_standard",
            exist_ok=True,
            plots=True,
            save=True,
            save_period=5,
            val=True,
            verbose=True,
        )

        best_model = self.models_dir / "stage2a_standard" / "weights" / "best.pt"

        # Validate on all datasets
        print(f"\n{'='*80}")
        print("📊 Stage 2A Validation on all datasets")
        print(f"{'='*80}\n")

        metrics = self._validate_all_datasets(best_model, "stage2a_standard")

        # Save metrics
        with open(self.logs_dir / "stage2a_standard_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Copy to checkpoints
        checkpoint_path = self.checkpoints_dir / "stage2a_standard_best.pt"
        shutil.copy(best_model, checkpoint_path)
        print(f"\n✅ Stage 2A Complete! Checkpoint: {checkpoint_path.name}")

        return best_model

    # ======================================================================
    # STAGE 2B: DCE Architecture
    # ======================================================================

    def train_dce(
        self,
        data_yaml: str = "cfg/sar_pos_only.yaml",
        epochs: int = 50,
        batch: int = 6,
        imgsz: int = 800,
        patience: int = 15,
        dce_yaml: str = "cfg/yolov8m_dce.yaml"
    ) -> Path:
        """
        Train Stage 2B: YOLOv8m + DCE modules (experimental)

        Uses DCE modules in early backbone layers for efficient feature extraction.
        """
        print(f"\n{'='*80}")
        print("🎯 STAGE 2B: YOLOv8m + DCE Fine-Tuning")
        print(f"{'='*80}")
        print(f"Strategy: DCE-enhanced architecture")
        print(f"Architecture: YOLOv8m + DCE modules (layers 1-4)")
        print(f"Expected: Better small object detection, fewer parameters")
        print(f"{'='*80}\n")

        config = {
            "variant": "dce",
            "stage1_weights": str(self.stage1_weights),
            "dce_yaml": dce_yaml,
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": str(self.device),
            "workers": self.workers,
            "optimizer": "AdamW",
            "lr0": 0.003,
            "lrf": 0.01,
            "patience": patience,
            "augmentation": {
                "hsv_h": 0.03,
                "hsv_s": 0.8,
                "hsv_v": 0.6,
                "degrees": 8.0,
                "translate": 0.2,
                "scale": 0.85,
                "mosaic": 0.15,
                "mixup": 0.0,
                "copy_paste": 0.1,
            }
        }

        with open(self.logs_dir / "stage2b_dce_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Check if DCE YAML exists; if not, create it
        dce_yaml_path = PROJECT_ROOT / dce_yaml
        if not dce_yaml_path.exists():
            print(f"⚠️  DCE YAML not found: {dce_yaml_path}")
            print(f"   Creating default DCE YAML with DCE + ERB + SCDown...")
            self._create_dce_yaml(dce_yaml_path)

        # Load model with DCE architecture
        print(f"📦 Loading DCE architecture from: {dce_yaml_path}")
        model = YOLO(str(dce_yaml_path))

        # Load weights from Stage 1 (transfer what's compatible)
        print(f"🔄 Transferring Stage 1 weights into DCE model...")
        try:
            stage1_yolo = YOLO(str(self.stage1_weights))
            stage1_state = stage1_yolo.model.state_dict()
            model_state = model.model.state_dict()

            transferred = 0
            skipped = 0
            for name, param in stage1_state.items():
                if name in model_state and model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred += 1
                else:
                    skipped += 1

            model.model.load_state_dict(model_state, strict=False)
            print(f"✅ Transferred {transferred} layers, skipped {skipped} incompatible layers")
        except Exception as e:
            print(f"⚠️  Could not transfer weights: {e}")
            print(f"   → Training DCE architecture from scratch...")

        print(f"\n🏋️ Starting Stage 2B training...\n")

        # Train DCE model (same hyperparameters as standard for fair comparison)
        _ = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self._device_arg(),
            workers=self.workers,

            optimizer="AdamW",
            lr0=0.003,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1.0,
            warmup_momentum=0.5,
            cos_lr=True,
            nbs=64,
            patience=patience,

            # same augmentation as standard
            hsv_h=0.03,
            hsv_s=0.8,
            hsv_v=0.6,
            degrees=8.0,
            translate=0.2,
            scale=0.85,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.15,
            mixup=0.0,
            copy_paste=0.1,
            perspective=0.002,
            close_mosaic=3,

            box=7.5,
            cls=0.5,
            dfl=1.5,

            iou=0.5,
            max_det=150,
            classes=[0],

            cache=False,
            rect=False,
            single_cls=True,
            amp=True,

            project=str(self.models_dir),
            name="stage2b_dce",
            exist_ok=True,
            plots=True,
            save=True,
            save_period=5,
            val=True,
            verbose=True,
        )

        best_model = self.models_dir / "stage2b_dce" / "weights" / "best.pt"

        print(f"\n{'='*80}")
        print("📊 Stage 2B Validation on all datasets")
        print(f"{'='*80}\n")

        metrics = self._validate_all_datasets(best_model, "stage2b_dce")

        with open(self.logs_dir / "stage2b_dce_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        checkpoint_path = self.checkpoints_dir / "stage2b_dce_best.pt"
        shutil.copy(best_model, checkpoint_path)
        print(f"\n✅ Stage 2B Complete! Checkpoint: {checkpoint_path.name}")

        return best_model

    # ======================================================================
    # VALIDATION ON ALL DATASETS
    # ======================================================================

    def _validate_all_datasets(self, weights_path: Path, variant_name: str) -> Dict:
        """Validate model on SARD, HERIDAL, VISDRONE"""
        model = YOLO(str(weights_path))

        datasets = [
            ("SARD", "cfg/sardyolo.yaml"),
            ("HERIDAL", "cfg/heridalyolo.yaml"),
            ("VISDRONE", "cfg/visdrone_person.yaml"),
        ]

        all_metrics = {}

        for name, yaml_path in datasets:
            yaml_file = PROJECT_ROOT / yaml_path
            if not yaml_file.exists():
                print(f"⚠️  Skipping {name}: {yaml_path} not found")
                continue

            print(f"Validating on {name}...")
            try:
                val_results = model.val(
                    data=str(yaml_file),
                    imgsz=800,
                    batch=2,
                    conf=0.25,
                    iou=0.5,
                    device=self._device_arg(),
                    plots=False
                )

                metrics = {
                    "dataset": name,
                    "variant": variant_name,
                    "precision": float(val_results.box.p[0]),
                    "recall": float(val_results.box.r[0]),
                    "mAP50": float(val_results.box.map50),
                    "mAP50-95": float(val_results.box.map),
                    "fitness": float(val_results.fitness),
                }

                print(f"  {name}: mAP50={metrics['mAP50']:.3f}, Recall={metrics['recall']:.3f}")

            except Exception as e:
                print(f"  ❌ Validation failed on {name}: {e}")
                metrics = {"dataset": name, "variant": variant_name, "error": str(e)}

            all_metrics[name] = metrics

        return all_metrics

    def _create_dce_yaml(self, output_path: Path):
        """Create default DCE YAML config (DCE + ERB + SCDown)"""
        import yaml

        config = {
            'nc': 1,
            'depth_multiple': 0.67,
            'width_multiple': 0.75,

            'backbone': [
                [-1, 1, 'Conv',   [64, 3, 2]],    # 0-P1/2
                [-1, 1, 'DCE',    [64, 2]],       # 1
                [-1, 1, 'Conv',   [128, 3, 2]],   # 2-P2/4
                [-1, 1, 'DCE',    [128, 2]],      # 3
                [-1, 1, 'ERB',    [128, 1]],      # 4
                [-1, 1, 'SCDown', [256, 3, 2]],   # 5-P3/8
                [-1, 2, 'ERB',    [256, 2]],      # 6
                [-1, 1, 'SCDown', [512, 3, 2]],   # 7-P4/16
                [-1, 2, 'ERB',    [512, 2]],      # 8
                [-1, 1, 'SPPF',   [512, 5]],      # 9
            ],

            'head': [
                [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],
                [-1, 2, 'ERB', [512, 2]],

                [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],
                [-1, 2, 'ERB', [256, 2]],  # P3/8-small

                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 12], 1, 'Concat', [1]],
                [-1, 2, 'ERB', [512, 2]],  # P4/16-medium

                [[15, 18], 1, 'Detect', ['nc']],
            ]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Created DCE YAML: {output_path}")

    def compare_results(self, standard_metrics: Dict, dce_metrics: Dict):
        """Generate comparison report"""
        print(f"\n{'='*80}")
        print("📊 STAGE 2 COMPARISON REPORT")
        print(f"{'='*80}\n")

        print(f"{'Dataset':<12} {'Variant':<10} {'mAP50':<8} {'Recall':<8} {'Precision':<10}")
        print(f"{'-'*80}")

        for dataset in ["SARD", "HERIDAL", "VISDRONE"]:
            if dataset in standard_metrics:
                m = standard_metrics[dataset]
                print(f"{dataset:<12} {'Standard':<10} {m.get('mAP50', 0):>7.3f} "
                      f"{m.get('recall', 0):>7.3f} {m.get('precision', 0):>9.3f}")

            if dataset in dce_metrics:
                m = dce_metrics[dataset]
                print(f"{dataset:<12} {'DCE':<10} {m.get('mAP50', 0):>7.3f} "
                      f"{m.get('recall', 0):>7.3f} {m.get('precision', 0):>9.3f}")

            if dataset in standard_metrics and dataset in dce_metrics:
                std_map = standard_metrics[dataset].get('mAP50', 0)
                dce_map = dce_metrics[dataset].get('mAP50', 0)
                improvement = ((dce_map - std_map) / std_map * 100) if std_map > 0 else 0
                symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f"{'':<12} {symbol} Δ: {improvement:+.1f}%")

            print()

        print(f"{'='*80}\n")

        comparison = {
            "standard": standard_metrics,
            "dce": dce_metrics,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.results_dir / "stage2_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"💾 Comparison saved to: {self.results_dir / 'stage2_comparison.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 SAR Fine-Tuning with DCE Comparison"
    )

    parser.add_argument("--stage1-weights", type=str, required=True,
                        help="Path to Stage 1 best.pt (e.g. checkpoints/best.pt)")
    parser.add_argument("--variant", type=str, choices=["standard", "dce", "both"],
                        default="both", help="Which variant to train")
    parser.add_argument("--train-both", action="store_true",
                        help="Train both variants (same as --variant both)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--batch", type=int, default=6,
                        help="Batch size")
    parser.add_argument("--experiment", type=str, default="stage2_comparison",
                        help="Experiment name")

    args = parser.parse_args()

    stage1_path = Path(args.stage1_weights)
    if not stage1_path.exists():
        print(f"❌ Error: Stage 1 weights not found: {stage1_path}")
        exit(1)

    trainer = Stage2Trainer(
        stage1_weights=stage1_path,
        experiment_name=args.experiment
    )

    variant = "both" if args.train_both else args.variant

    standard_metrics = None
    dce_metrics = None

    if variant in ["standard", "both"]:
        _ = trainer.train_standard(epochs=args.epochs, batch=args.batch)
        standard_metrics = json.load(open(trainer.logs_dir / "stage2a_standard_metrics.json"))

    if variant in ["dce", "both"]:
        _ = trainer.train_dce(epochs=args.epochs, batch=args.batch)
        dce_metrics = json.load(open(trainer.logs_dir / "stage2b_dce_metrics.json"))

    if variant == "both" and standard_metrics and dce_metrics:
        trainer.compare_results(standard_metrics, dce_metrics)

    print(f"\n{'='*80}")
    print("✅ STAGE 2 TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Results: {trainer.exp_dir}")
    print(f"📦 Checkpoints: {trainer.checkpoints_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
