"""
Complete SAR Training Pipeline - YOLOv8m
Combines:
- Paper-inspired aggressive Stage 1 (VisDrone foundation)
- Your proven Stage 3 RTX A2000-safe finetuning approach
- Flexible validation with FP-only fallback

Training Strategy:
1. Stage 1: Strong VisDrone foundation (150 epochs, aggressive aug)
2. Stage 2: SAR fine-tuning (SARD + HERIDAL, reduced aug, your proven setup)
3. Optional Stage 3: Hard negative mining (NTUT4K)

Author: Based on your stage3_train_sar.py + paper methodology
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import shutil
import platform
from datetime import datetime
from typing import Optional, Dict, List

from ultralytics import YOLO
from config import PROJECT_ROOT, DEVICE, DATA_ROOT


# =============================================================================
# HELPER FUNCTIONS (from your stage3_train_sar.py)
# =============================================================================

def _device_arg_from_torch_device(dev) -> int | str:
    """Convert torch device to Ultralytics device arg."""
    if getattr(dev, "type", "") == "cuda":
        return dev.index if dev.index is not None else 0
    return "cpu"


def _resolve_val_dirs_from_yaml(yaml_path: Path) -> list[Path]:
    """
    Minimal YAML reader for 'path:' and 'val:' (string or list).
    Returns list of image directories.
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

    # Fallback for classic YOLO structure
    if not out_dirs and root is not None:
        out_dirs = [root / "images" / "val"]

    return out_dirs


def _fp_only_eval(m: YOLO, sources: list[Path], imgsz: int, conf: float, iou: float, device_arg) -> dict:
    """
    Fallback evaluation for datasets without labels (e.g., NTUT4K).
    Counts predictions and returns FP-only metrics.
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
            n = 0 if r.boxes is None else int(len(r.boxes))
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
        "mode": "FP-only"
    }


def _save_metrics(exp_dir: Path, name: str, metrics: dict) -> Path:
    """Save metrics to JSON file."""
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    outp = exp_dir / "logs" / f"{name}.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return outp


def _unique_path(dst: Path) -> Path:
    """Generate unique file path if file exists (file_1.pt, file_2.pt, ...)."""
    if not dst.exists():
        return dst
    stem, suf = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


def _run_validation(
    weights_path: Path,
    data_yaml: Path,
    exp_dir: Path,
    val_batch: int = 2,
    val_workers: int = 0,
    val_imgsz: int = 800,
    val_conf: float = 0.25,
    val_iou: float = 0.50,
    val_plots: bool = False,
    device_arg = 0,
    dataset_name: str = "unknown"
) -> Dict:
    """
    Run validation with automatic FP-only fallback for unlabeled datasets.
    Returns metrics dict.
    """
    print(f"\n[validation] Running validation on {dataset_name}...")
    m = YOLO(str(weights_path))
    
    try:
        # Standard validation (works ONLY with labeled datasets)
        val = m.val(
            data=str(data_yaml),
            split="val",
            plots=val_plots,
            batch=val_batch,
            workers=val_workers,
            imgsz=val_imgsz,
            conf=val_conf,
            iou=val_iou,
            device=device_arg,
        )

        # Check if there were GT labels
        p = getattr(val.box, "p", None)
        if p is None or (hasattr(p, "size") and p.size == 0):
            raise ValueError("No GT labels found in val set")

        metrics = {
            "model": weights_path.stem,
            "dataset": dataset_name,
            "precision": float(val.box.p[0]),
            "recall": float(val.box.r[0]),
            "mAP50": float(val.box.map50),
            "mAP50-95": float(val.box.map),
            "fitness": float(val.fitness),
            "mode": "GT"
        }
        
        print(f"[validation][{dataset_name}] P={metrics['precision']:.3f} | "
              f"R={metrics['recall']:.3f} | mAP50={metrics['mAP50']:.3f}")

    except Exception as e:
        # Fallback for negative-only datasets (e.g., NTUT4K)
        print(f"[validation][{dataset_name}] No labels detected ({e}). Using FP-only eval.")
        val_dirs = _resolve_val_dirs_from_yaml(data_yaml)
        metrics = _fp_only_eval(
            m, val_dirs,
            imgsz=val_imgsz,
            conf=val_conf,
            iou=val_iou,
            device_arg=device_arg
        )
        metrics["model"] = weights_path.stem
        metrics["dataset"] = dataset_name
        
        print(f"[validation][{dataset_name}][FP-only] "
              f"avg_fp_per_image={metrics['avg_fp_per_image']:.3f}")

    outp = _save_metrics(exp_dir, f"val_{dataset_name}_{weights_path.stem}", metrics)
    print(f"[validation] Saved to: {outp.name}")
    
    return metrics


# =============================================================================
# MAIN TRAINER CLASS
# =============================================================================

class CompleteSARTrainer:
    """
    Complete training pipeline for aerial person detection.
    Combines paper methodology with your proven RTX A2000-safe approach.
    """
    
    def __init__(self, base_model: str = "yolov8m.pt", experiment_name: str = "sar_complete"):
        self.base_model = base_model
        self.experiment_name = experiment_name
        self.device = DEVICE
        self.device_arg = _device_arg_from_torch_device(DEVICE)
        
        # Platform-specific settings (from your code)
        self.is_windows = platform.system().lower().startswith("win")
        self.workers_stage1 = 4  # Stage 1: can use more workers
        self.workers_stage2 = 2 if self.is_windows else 4  # Stage 2: conservative
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = PROJECT_ROOT / "experiments" / f"{experiment_name}_{timestamp}"
        self.models_dir = self.exp_dir / "models"
        self.logs_dir = self.exp_dir / "logs"
        self.checkpoints_dir = PROJECT_ROOT / "checkpoints"
        
        for d in [self.models_dir, self.logs_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🚀 SAR Training Pipeline Initialized")
        print(f"{'='*80}")
        print(f"📁 Experiment: {self.exp_dir.name}")
        print(f"🖥️  Device: {self.device} ({self.device_arg})")
        print(f"💻 Platform: {'Windows (2 workers)' if self.is_windows else 'Unix (4 workers)'}")
        print(f"{'='*80}\n")
    
    def _save_config(self, stage: str, config: dict):
        """Save training configuration for reproducibility."""
        config_file = self.logs_dir / f"{stage}_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"💾 Config saved: {config_file.name}")
    
    def _copy_best_to_checkpoint(self, best_path: Path, stage: str, extra_tag: str = ""):
        """Copy best.pt to checkpoints directory with descriptive name."""
        if not best_path.exists():
            print(f"[checkpoint] WARN: {best_path} not found")
            return None
        
        ckpt_name = f"{stage}_{extra_tag}_best.pt".replace("__", "_")
        dst = _unique_path(self.checkpoints_dir / ckpt_name)
        shutil.copy(best_path, dst)
        print(f"[checkpoint] ✓ Copied to: {dst.name}")
        return dst
    
    # =========================================================================
    # STAGE 1: VisDrone Foundation (Paper-inspired)
    # =========================================================================
    
    def stage1_visdrone_foundation(
        self,
        data_yaml: str = "cfg/visdrone_person.yaml",
        epochs: int = 150,
        batch: int = 16,
        imgsz: int = 800,
        patience: int = 30,
        resume: bool = False
    ) -> Path:
        """
        Stage 1: Build strong VisDrone foundation
        
        Goal: mAP50 > 0.65 (vs. your current 0.545)
        Strategy: Aggressive augmentation, long training, early stopping
        Based on Ciccone & Ceruti (2025) methodology
        """
        print(f"\n{'='*80}")
        print("🎯 STAGE 1: VisDrone Foundation Training")
        print(f"{'='*80}")
        print(f"Target: mAP50 > 0.65 (your current best: 0.545)")
        print(f"Epochs: {epochs} (patience: {patience})")
        print(f"Strategy: AGGRESSIVE augmentation for robust features")
        print(f"Batch: {batch} | Workers: {self.workers_stage1}")
        print(f"{'='*80}\n")
        
        config = {
            "stage": "stage1_visdrone_foundation",
            "model": self.base_model,
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": str(self.device),
            "workers": self.workers_stage1,
            
            # Optimizer - AdamW from your successful runs
            "optimizer": "AdamW",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            
            # Learning rate schedule
            "cos_lr": True,
            "nbs": 64,
            "patience": patience,
            
            # AGGRESSIVE AUGMENTATION (Paper values)
            "augmentation": {
                "hsv_h": 0.015,      # Paper: 0.015
                "hsv_s": 0.7,        # Paper: 0.7
                "hsv_v": 0.4,        # Paper: 0.4
                "degrees": 10.0,     # Rotation
                "translate": 0.2,    # Translation
                "scale": 0.9,        # Scale variation
                "fliplr": 0.5,       # Horizontal flip
                "flipud": 0.0,       # No vertical flip for aerial
                "mosaic": 1.0,       # Strong mosaic (paper: 1.0)
                "mixup": 0.15,       # Moderate mixup
                "copy_paste": 0.3,   # Strong copy-paste for small objects
                "perspective": 0.0,  # Not useful for aerial
                "shear": 0.0,
                "erasing": 0.0,
                "auto_augment": None,
                "close_mosaic": 20,  # Disable in last 20 epochs
            },
            
            # Loss weights (your proven values)
            "loss_weights": {
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
            },
            
            # NMS and evaluation
            "inference": {
                "iou": 0.5,
                "conf": 0.001,
                "max_det": 300,  # VisDrone has many objects
            },
            
            "cache": False,
            "rect": False,
            "single_cls": True,
            "amp": True,
        }
        
        self._save_config("stage1_visdrone", config)
        
        # Train
        model = YOLO(self.base_model)
        
        print(f"🏋️ Starting Stage 1 training...")
        print(f"⏰ Expected time: ~8-12 hours on RTX 3060/A2000")
        print(f"💡 Tip: Monitor results.csv for convergence\n")
        
        run_name = "stage1_visdrone"
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device_arg,
            workers=self.workers_stage1,
            
            # Optimizer
            optimizer="AdamW",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=True,
            nbs=64,
            patience=patience,
            
            # Augmentation (AGGRESSIVE)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.2,
            scale=0.9,
            fliplr=0.5,
            flipud=0.0,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.3,
            perspective=0.0,
            shear=0.0,
            close_mosaic=20,
            
            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Inference
            iou=0.5,
            max_det=300,
            classes=[0],  # Person only
            
            # Performance
            cache=False,
            rect=False,
            single_cls=True,
            amp=True,
            
            # Project structure
            project=str(self.models_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            save_period=10,
            val=True,
            verbose=True,
        )
        
        # Get best model
        best_model = self.models_dir / run_name / "weights" / "best.pt"
        
        # Validate
        print(f"\n{'-'*80}")
        print("📊 Stage 1 Final Validation on VisDrone")
        print(f"{'-'*80}")
        
        metrics = _run_validation(
            weights_path=best_model,
            data_yaml=Path(data_yaml),
            exp_dir=self.exp_dir,
            val_batch=batch,
            val_workers=self.workers_stage1,
            val_imgsz=imgsz,
            val_conf=0.25,
            val_iou=0.5,
            val_plots=True,
            device_arg=self.device_arg,
            dataset_name="VISDRONE"
        )
        
        print(f"\n✅ Stage 1 Complete!")
        print(f"   mAP50:     {metrics['mAP50']:.4f} (target: > 0.65)")
        print(f"   mAP50-95:  {metrics['mAP50-95']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}\n")
        
        if metrics['mAP50'] < 0.60:
            print(f"⚠️  WARNING: mAP50 below 0.60!")
            print(f"   Consider:")
            print(f"   - Increasing epochs (current: {epochs})")
            print(f"   - Checking data quality")
            print(f"   - Reviewing augmentation settings\n")
        
        # Copy to checkpoints
        self._copy_best_to_checkpoint(best_model, "stage1", "visdrone_150ep")
        
        return best_model
    
    # =========================================================================
    # STAGE 2: SAR Fine-Tuning (Your proven approach + Paper strategy)
    # =========================================================================
    
    def stage2_sar_finetuning(
        self,
        stage1_weights: Path,
        data_yaml: str = "cfg/sar_pos_only.yaml",  # SARD + HERIDAL
        epochs: int = 50,
        batch: int = 6,  # RTX A2000-safe
        imgsz: int = 800,
        patience: int = 15,
        freeze: int = 0,
        skip_epoch_val: bool = False
    ) -> Path:
        """
        Stage 2: SAR Fine-Tuning with REDUCED augmentation
        
        Based on:
        - Your proven stage3_train_sar.py approach
        - Paper strategy: softer augmentation for fine-tuning
        - RTX A2000-safe settings (batch=6, workers=2 on Windows)
        """
        print(f"\n{'='*80}")
        print("🎯 STAGE 2: SAR Fine-Tuning (SARD + HERIDAL)")
        print(f"{'='*80}")
        print(f"Starting from: {stage1_weights.name}")
        print(f"Target: mAP50 > 0.70 (HERIDAL), > 0.80 (SARD)")
        print(f"Strategy: REDUCED augmentation (paper strategy)")
        print(f"Batch: {batch} | Workers: {self.workers_stage2} | Freeze: {freeze}")
        print(f"{'='*80}\n")
        
        config = {
            "stage": "stage2_sar_finetuning",
            "start_weights": str(stage1_weights),
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": str(self.device),
            "workers": self.workers_stage2,
            "freeze": freeze,
            
            # Lower learning rate for fine-tuning
            "optimizer": "AdamW",
            "lr0": 0.003,  # Lower than stage 1 (0.01)
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 1.0,  # Shorter warmup
            "cos_lr": True,
            "nbs": 64,
            "patience": patience,
            
            # REDUCED AUGMENTATION (Paper + your values)
            "augmentation": {
                "hsv_h": 0.03,       # Your value (was 0.015 in stage1)
                "hsv_s": 0.8,        # Your value (was 0.7)
                "hsv_v": 0.6,        # Your value (was 0.4)
                "degrees": 8.0,      # Your value (was 10.0)
                "translate": 0.2,    # Your value
                "scale": 0.85,       # Your value
                "fliplr": 0.5,
                "flipud": 0.0,
                "mosaic": 0.15,      # Your value (was 1.0 in stage1)
                "mixup": 0.0,        # DISABLED (paper strategy)
                "copy_paste": 0.1,   # Your value (was 0.3)
                "perspective": 0.002,
                "close_mosaic": 3,   # Your value
            },
            
            "loss_weights": {
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
            },
            
            "inference": {
                "iou": 0.5,
                "max_det": 150,  # SAR has fewer objects
            },
            
            "skip_epoch_val": skip_epoch_val,
        }
        
        self._save_config("stage2_sar_finetuning", config)
        
        model = YOLO(str(stage1_weights))
        
        print(f"🏋️ Starting Stage 2 fine-tuning...")
        print(f"⏰ Expected time: ~3-5 hours on RTX A2000")
        print(f"💡 Using your proven RTX A2000-safe settings\n")
        
        run_name = "stage2_sar_finetuning"
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device_arg,
            workers=self.workers_stage2,
            
            # Optimizer (lower LR for fine-tuning)
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
            
            # REDUCED Augmentation
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
            
            # Performance (RTX A2000-safe)
            cache=False,
            rect=False,
            single_cls=True,
            amp=True,
            freeze=freeze,
            
            # Project
            project=str(self.models_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            save_period=5,
            val=not skip_epoch_val,
            verbose=True,
        )
        
        best_model = self.models_dir / run_name / "weights" / "best.pt"
        
        # Validate on HERIDAL
        print(f"\n{'-'*80}")
        print("📊 Stage 2 Validation on HERIDAL")
        print(f"{'-'*80}")
        
        heridal_metrics = _run_validation(
            weights_path=best_model,
            data_yaml=Path("cfg/heridalyolo.yaml"),
            exp_dir=self.exp_dir,
            val_batch=2,  # Safe for validation
            val_workers=0,
            val_imgsz=imgsz,
            device_arg=self.device_arg,
            dataset_name="HERIDAL"
        )
        
        # Validate on SARD
        print(f"\n{'-'*80}")
        print("📊 Stage 2 Validation on SARD")
        print(f"{'-'*80}")
        
        sard_metrics = _run_validation(
            weights_path=best_model,
            data_yaml=Path("cfg/sardyolo.yaml"),
            exp_dir=self.exp_dir,
            val_batch=2,
            val_workers=0,
            val_imgsz=imgsz,
            device_arg=self.device_arg,
            dataset_name="SARD"
        )
        
        # Validate on VisDrone (to check if we degraded)
        print(f"\n{'-'*80}")
        print("📊 Stage 2 Validation on VisDrone (sanity check)")
        print(f"{'-'*80}")
        
        visdrone_metrics = _run_validation(
            weights_path=best_model,
            data_yaml=Path("cfg/visdrone_person.yaml"),
            exp_dir=self.exp_dir,
            val_batch=2,
            val_workers=0,
            val_imgsz=imgsz,
            device_arg=self.device_arg,
            dataset_name="VISDRONE"
        )
        
        print(f"\n✅ Stage 2 Complete!")
        print(f"\n📊 Results Summary:")
        print(f"   HERIDAL  → mAP50: {heridal_metrics['mAP50']:.3f} | Recall: {heridal_metrics['recall']:.3f}")
        print(f"   SARD     → mAP50: {sard_metrics['mAP50']:.3f} | Recall: {sard_metrics['recall']:.3f}")
        print(f"   VisDrone → mAP50: {visdrone_metrics['mAP50']:.3f} (sanity check)\n")
        
        # Copy to checkpoints
        self._copy_best_to_checkpoint(best_model, "stage2", f"sar_{epochs}ep")
        
        return best_model
    
    # =========================================================================
    # STAGE 3: Hard Negative Mining (Optional)
    # =========================================================================
    
    def stage3_hard_negative_mining(
        self,
        stage2_weights: Path,
        data_yaml: str = "cfg/sar_person_mix.yaml",  # Includes NTUT4K
        epochs: int = 10,
        batch: int = 6,
        imgsz: int = 800,
        patience: int = 5
    ) -> Path:
        """
        Stage 3 (Optional): Hard negative mining with NTUT4K
        
        Goal: Reduce false positives (6.58 → < 2.0)
        Strategy: Mix positive samples with hard negatives
        """
        print(f"\n{'='*80}")
        print("🎯 STAGE 3: Hard Negative Mining (NTUT4K)")
        print(f"{'='*80}")
        print(f"Starting from: {stage2_weights.name}")
        print(f"Goal: Reduce false positives (target: < 2.0 FP/image)")
        print(f"Strategy: Train with SARD + HERIDAL + NTUT4K (negatives)")
        print(f"{'='*80}\n")
        
        config = {
            "stage": "stage3_hard_negative_mining",
            "start_weights": str(stage2_weights),
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "workers": self.workers_stage2,
            
            # Very low LR - just adaptation
            "optimizer": "AdamW",
            "lr0": 0.001,  # Very low
            "lrf": 0.1,
            "patience": patience,
            
            # MINIMAL augmentation
            "augmentation": {
                "hsv_h": 0.005,
                "hsv_s": 0.3,
                "hsv_v": 0.2,
                "degrees": 0.0,
                "translate": 0.05,
                "scale": 0.9,
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
            }
        }
        
        self._save_config("stage3_hard_negative", config)
        
        model = YOLO(str(stage2_weights))
        
        print(f"🏋️ Starting Stage 3 training...")
        print(f"⏰ Expected time: ~1-2 hours\n")
        
        run_name = "stage3_hard_negative"
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device_arg,
            workers=self.workers_stage2,
            
            # Very low LR
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1.0,
            cos_lr=True,
            nbs=64,
            patience=patience,
            
            # Minimal augmentation
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            degrees=0.0,
            translate=0.05,
            scale=0.9,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            
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
            amp=True,
            
            # Project
            project=str(self.models_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            val=True,
            verbose=True,
        )
        
        best_model = self.models_dir / run_name / "weights" / "best.pt"
        
        # Test FP reduction on NTUT4K
        print(f"\n{'-'*80}")
        print("📊 Stage 3 Validation - False Positive Test (NTUT4K)")
        print(f"{'-'*80}")
        
        ntut4k_metrics = _run_validation(
            weights_path=best_model,
            data_yaml=Path("cfg/ntut4kyolo.yaml"),
            exp_dir=self.exp_dir,
            val_batch=2,
            val_workers=0,
            val_imgsz=imgsz,
            val_conf=0.30,  # Your standard conf
            device_arg=self.device_arg,
            dataset_name="NTUT4K"
        )
        
        # Re-validate on positive datasets
        print(f"\n{'-'*80}")
        print("📊 Stage 3 Validation - HERIDAL (check we didn't degrade)")
        print(f"{'-'*80}")
        
        heridal_metrics = _run_validation(
            weights_path=best_model,
            data_yaml=Path("cfg/heridalyolo.yaml"),
            exp_dir=self.exp_dir,
            val_batch=2,
            val_workers=0,
            val_imgsz=imgsz,
            device_arg=self.device_arg,
            dataset_name="HERIDAL"
        )
        
        print(f"\n✅ Stage 3 Complete!")
        print(f"\n📊 False Positive Reduction:")
        if ntut4k_metrics.get("mode") == "FP-only":
            fp_value = ntut4k_metrics.get("avg_fp_per_image", 0)
            print(f"   NTUT4K FP/image: {fp_value:.3f}")
            if fp_value < 2.0:
                print(f"   ✓ Target achieved! (< 2.0)")
            else:
                print(f"   ⚠️ Still above target (goal: < 2.0)")
        
        print(f"\n📊 HERIDAL Performance Check:")
        print(f"   mAP50: {heridal_metrics['mAP50']:.3f} | Recall: {heridal_metrics['recall']:.3f}\n")
        
        # Copy to checkpoints
        self._copy_best_to_checkpoint(best_model, "stage3", f"hard_neg_{epochs}ep")
        
        return best_model
    
    # =========================================================================
    # VALIDATION-ONLY MODE
    # =========================================================================
    
    def validate_model_all_datasets(
        self,
        weights_path: Path,
        imgsz: int = 800,
        conf: float = 0.25,
        batch: int = 2
    ):
        """
        Validate a model on all available datasets.
        Useful for comparing different checkpoints.
        """
        print(f"\n{'='*80}")
        print(f"🔎 COMPREHENSIVE VALIDATION")
        print(f"{'='*80}")
        print(f"Model: {weights_path.name}")
        print(f"Settings: imgsz={imgsz}, conf={conf}, batch={batch}")
        print(f"{'='*80}\n")
        
        datasets = [
            ("SARD", "cfg/sardyolo.yaml"),
            ("HERIDAL", "cfg/heridalyolo.yaml"),
            ("VISDRONE", "cfg/visdrone_person.yaml"),
            ("NTUT4K", "cfg/ntut4kyolo.yaml"),
        ]
        
        all_metrics = {}
        
        for name, yaml_path in datasets:
            yaml_file = Path(yaml_path)
            if not yaml_file.exists():
                print(f"⚠️ Skipping {name}: {yaml_path} not found\n")
                continue
            
            metrics = _run_validation(
                weights_path=weights_path,
                data_yaml=yaml_file,
                exp_dir=self.exp_dir,
                val_batch=batch,
                val_workers=0,
                val_imgsz=imgsz,
                val_conf=conf,
                val_iou=0.5,
                device_arg=self.device_arg,
                dataset_name=name
            )
            
            all_metrics[name] = metrics
            print()  # Blank line between datasets
        
        # Summary table
        print(f"\n{'='*80}")
        print(f"📊 VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Dataset':<12} {'Mode':<10} {'mAP50':<8} {'Recall':<8} {'Precision':<10}")
        print(f"{'-'*80}")
        
        for name, metrics in all_metrics.items():
            mode = metrics.get('mode', 'GT')
            if mode == 'GT':
                print(f"{name:<12} {mode:<10} {metrics['mAP50']:>7.3f} "
                      f"{metrics['recall']:>7.3f} {metrics['precision']:>9.3f}")
            else:  # FP-only
                fp = metrics.get('avg_fp_per_image', 0)
                print(f"{name:<12} {mode:<10} {'N/A':<8} {'N/A':<8} "
                      f"FP={fp:.3f}/img")
        
        print(f"{'='*80}\n")
        
        # Save combined metrics
        combined_file = self.logs_dir / f"validation_all_datasets_{weights_path.stem}.json"
        with open(combined_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"💾 Combined metrics saved: {combined_file.name}\n")
        
        return all_metrics
    
    # =========================================================================
    # COMPLETE PIPELINE
    # =========================================================================
    
    def run_full_pipeline(
        self,
        skip_stage1: bool = False,
        stage1_weights: Optional[Path] = None,
        skip_stage3: bool = True,
        stage1_epochs: int = 150,
        stage2_epochs: int = 50,
        stage3_epochs: int = 10,
        stage1_only: bool = False  # NEW: Stop after Stage 1
    ):
        """
        Run complete training pipeline.
        
        Args:
            skip_stage1: Use existing stage1 weights instead of training
            stage1_weights: Path to pre-trained stage 1 model
            skip_stage3: Skip optional hard negative mining
            stage1_epochs: VisDrone training epochs
            stage2_epochs: SAR fine-tuning epochs
            stage3_epochs: Hard negative mining epochs
            stage1_only: Only run Stage 1, then stop (for testing)
        """
        print(f"\n{'='*80}")
        print("🚀 COMPLETE SAR TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Base Model: {self.base_model}")
        print(f"Device: {self.device} ({self.device_arg})")
        print(f"Experiment: {self.exp_dir.name}")
        print(f"\n📋 Pipeline Plan:")
        
        if stage1_only:
            print(f"   Stage 1: Train {stage1_epochs} epochs (THEN STOP)")
            print(f"   Stage 2: SKIP")
            print(f"   Stage 3: SKIP")
        else:
            print(f"   Stage 1: {'SKIP (using existing)' if skip_stage1 else f'Train {stage1_epochs} epochs'}")
            print(f"   Stage 2: Train {stage2_epochs} epochs")
            print(f"   Stage 3: {'SKIP' if skip_stage3 else f'Train {stage3_epochs} epochs'}")
        print(f"{'='*80}\n")
        
        # Stage 1: VisDrone Foundation
        if skip_stage1 and stage1_weights:
            print(f"⏭️  Skipping Stage 1, using: {stage1_weights.name}\n")
            s1_model = stage1_weights
        else:
            s1_model = self.stage1_visdrone_foundation(
                epochs=stage1_epochs,
                batch=16,
                patience=30
            )
        
        # Stop here if stage1_only
        if stage1_only:
            print(f"\n{'='*80}")
            print("🛑 STAGE 1 COMPLETE - STOPPING AS REQUESTED")
            print(f"{'='*80}")
            print(f"📁 Stage 1 model saved to: {s1_model}")
            print(f"📦 Checkpoint copied to: {self.checkpoints_dir}")
            print(f"\n🔄 To continue with Stage 2, run:")
            print(f"   python complete_sar_training_pipeline.py \\")
            print(f"       --skip-stage1 \\")
            print(f"       --stage1-weights {s1_model} \\")
            print(f"       --stage2-epochs 50")
            print(f"{'='*80}\n")
            return s1_model
        
        # Stage 2: SAR Fine-tuning
        s2_model = self.stage2_sar_finetuning(
            stage1_weights=s1_model,
            epochs=stage2_epochs,
            batch=6,  # RTX A2000-safe
            patience=15
        )
        
        # Stage 3: Hard Negative Mining (Optional)
        if not skip_stage3:
            s3_model = self.stage3_hard_negative_mining(
                stage2_weights=s2_model,
                epochs=stage3_epochs,
                batch=6
            )
            final_model = s3_model
            final_stage = "stage3"
        else:
            final_model = s2_model
            final_stage = "stage2"
        
        # Final comprehensive validation
        print(f"\n{'='*80}")
        print("🎯 FINAL COMPREHENSIVE VALIDATION")
        print(f"{'='*80}\n")
        
        self.validate_model_all_datasets(
            weights_path=final_model,
            imgsz=800,
            conf=0.25,
            batch=2
        )
        
        # Final Summary
        print(f"\n{'='*80}")
        print("✅ TRAINING PIPELINE COMPLETE!")
        print(f"{'='*80}")
        print(f"📁 Experiment directory: {self.exp_dir}")
        print(f"🏆 Final model: {final_model.name}")
        print(f"📦 Checkpoints saved to: {self.checkpoints_dir}")
        print(f"\n📊 Next steps:")
        print(f"   1. Test on live video:")
        print(f"      python test_inference.py --weights {final_model}")
        print(f"   2. Run confidence sweeps:")
        print(f"      python sweep_confidence.py --weights {final_model}")
        print(f"   3. Test different resolutions (640, 800, 960)")
        print(f"   4. Optimize for deployment (TensorRT, quantization)")
        print(f"   5. Test 90° top-down perspectives")
        print(f"{'='*80}\n")
        
        return final_model


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete SAR Training Pipeline - YOLOv8m",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (Stage 1 + 2)
  python complete_sar_training_pipeline.py
  
  # Full pipeline with Stage 3 (hard negative mining)
  python complete_sar_training_pipeline.py --enable-stage3
  
  # Skip Stage 1, use existing weights
  python complete_sar_training_pipeline.py --skip-stage1 --stage1-weights path/to/stage1_best.pt
  
  # Validation only
  python complete_sar_training_pipeline.py --val-only --weights path/to/model.pt
  
  # Custom epochs
  python complete_sar_training_pipeline.py --stage1-epochs 100 --stage2-epochs 30
        """
    )
    
    # Pipeline control
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                       help="Base model (yolov8m.pt, yolov8l.pt, etc.)")
    parser.add_argument("--experiment", type=str, default="sar_complete",
                       help="Experiment name")
    
    # Stage control
    parser.add_argument("--stage1-only", action="store_true",
                       help="Only run Stage 1 (VisDrone foundation), then stop")
    parser.add_argument("--skip-stage1", action="store_true",
                       help="Skip Stage 1 if you have pre-trained weights")
    parser.add_argument("--stage1-weights", type=str, default=None,
                       help="Path to Stage 1 weights (if skipping Stage 1)")
    parser.add_argument("--enable-stage3", action="store_true",
                       help="Enable Stage 3 hard negative mining")
    
    # Epoch control
    parser.add_argument("--stage1-epochs", type=int, default=150,
                       help="Stage 1 epochs (default: 150)")
    parser.add_argument("--stage2-epochs", type=int, default=50,
                       help="Stage 2 epochs (default: 50)")
    parser.add_argument("--stage3-epochs", type=int, default=10,
                       help="Stage 3 epochs (default: 10)")
    
    # Validation-only mode
    parser.add_argument("--val-only", action="store_true",
                       help="Only validate, no training")
    parser.add_argument("--weights", type=str, default=None,
                       help="Model weights for validation-only mode")
    parser.add_argument("--val-imgsz", type=int, default=800,
                       help="Validation image size")
    parser.add_argument("--val-conf", type=float, default=0.25,
                       help="Validation confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CompleteSARTrainer(
        base_model=args.model,
        experiment_name=args.experiment
    )
    
    # Validation-only mode
    if args.val_only:
        if not args.weights:
            print("❌ Error: --weights required for validation-only mode")
            exit(1)
        
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"❌ Error: Weights not found: {weights_path}")
            exit(1)
        
        trainer.validate_model_all_datasets(
            weights_path=weights_path,
            imgsz=args.val_imgsz,
            conf=args.val_conf,
            batch=2
        )
        exit(0)
    
    # Training mode
    stage1_weights = Path(args.stage1_weights) if args.stage1_weights else None
    
    if args.skip_stage1 and not stage1_weights:
        print("❌ Error: --stage1-weights required when using --skip-stage1")
        exit(1)
    
    if stage1_weights and not stage1_weights.exists():
        print(f"❌ Error: Stage 1 weights not found: {stage1_weights}")
        exit(1)
    
    # Run pipeline
    trainer.run_full_pipeline(
        skip_stage1=args.skip_stage1,
        stage1_weights=stage1_weights,
        skip_stage3=not args.enable_stage3,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage1_only=args.stage1_only  # NEW: Stop after Stage 1
    )