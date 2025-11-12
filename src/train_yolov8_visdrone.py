# src/train_yolov8_visdrone.py
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

from src.config import PROJECT_ROOT, DEVICE


def setup_experiment_dir(experiment_name: str) -> Path:
    """Create timestamped experiment directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = PROJECT_ROOT / "experiments" / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "weights").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    print(f"[setup] Experiment directory: {exp_dir}")
    return exp_dir


def save_training_config(exp_dir: Path, config: dict):
    """Save training configuration to JSON."""
    config_file = exp_dir / "training_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[config] Saved to {config_file}")


def plot_training_metrics(results_csv: Path, exp_dir: Path):
    """Plot comprehensive training metrics in English."""
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Remove whitespace

    run_name = exp_dir.name  # e.g. yolov8m_visdrone_optimized_20251112_084721

    # Set professional style
    plt.style.use("seaborn-v0_8-darkgrid")

    # === FIGURE 1: Loss Curves ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training and Validation Losses", fontsize=16, fontweight="bold")

    # Box Loss
    axes[0, 0].plot(
        df["epoch"], df["train/box_loss"], label="Train Box Loss", linewidth=2
    )
    axes[0, 0].plot(
        df["epoch"], df["val/box_loss"], label="Val Box Loss", linewidth=2
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Bounding Box Regression Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Class Loss
    axes[0, 1].plot(
        df["epoch"], df["train/cls_loss"], label="Train Class Loss", linewidth=2
    )
    axes[0, 1].plot(
        df["epoch"], df["val/cls_loss"], label="Val Class Loss", linewidth=2
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Classification Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # DFL Loss (Distribution Focal Loss)
    axes[1, 0].plot(
        df["epoch"], df["train/dfl_loss"], label="Train DFL Loss", linewidth=2
    )
    axes[1, 0].plot(
        df["epoch"], df["val/dfl_loss"], label="Val DFL Loss", linewidth=2
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Distribution Focal Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Total Loss (if available)
    has_total_loss = False
    if "train/loss" in df.columns:
        axes[1, 1].plot(
            df["epoch"], df["train/loss"], label="Total Train Loss", linewidth=2
        )
        has_total_loss = True
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Total Training Loss")
    if has_total_loss:
        axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = exp_dir / "plots" / f"loss_curves_{run_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[plot] Saved:", out_path.name)

    # === FIGURE 2: Detection Metrics ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Detection Performance Metrics", fontsize=16, fontweight="bold")

    # Precision
    axes[0, 0].plot(
        df["epoch"],
        df["metrics/precision(B)"],
        label="Precision",
        color="green",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].set_title("Precision @ IoU=0.5 (Person Class)")
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Recall
    axes[0, 1].plot(
        df["epoch"],
        df["metrics/recall(B)"],
        label="Recall",
        color="blue",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].set_title("Recall @ IoU=0.5 (Person Class)")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # mAP@0.5
    axes[1, 0].plot(
        df["epoch"],
        df["metrics/mAP50(B)"],
        label="mAP@0.5",
        color="red",
        linewidth=2,
        marker="s",
        markersize=3,
    )
    axes[1, 0].axhline(
        y=0.55,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="Target (0.55-0.65)",
    )
    axes[1, 0].axhline(y=0.65, color="orange", linestyle="--", linewidth=1.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mAP@0.5")
    axes[1, 0].set_title("Mean Average Precision @ IoU=0.5")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # mAP@0.5:0.95
    axes[1, 1].plot(
        df["epoch"],
        df["metrics/mAP50-95(B)"],
        label="mAP@0.5:0.95",
        color="purple",
        linewidth=2,
        marker="^",
        markersize=3,
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("mAP@0.5:0.95")
    axes[1, 1].set_title("Mean Average Precision @ IoU=0.5:0.95")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    out_path = exp_dir / "plots" / f"detection_metrics_{run_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[plot] Saved:", out_path.name)

    # === FIGURE 3: Learning Rate & Summary ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Dynamics", fontsize=16, fontweight="bold")

    # Learning Rate
    if "lr/pg0" in df.columns:
        axes[0].plot(df["epoch"], df["lr/pg0"], label="LR (param group 0)", linewidth=2)
    if "lr/pg1" in df.columns:
        axes[0].plot(df["epoch"], df["lr/pg1"], label="LR (param group 1)", linewidth=2)
    if "lr/pg2" in df.columns:
        axes[0].plot(df["epoch"], df["lr/pg2"], label="LR (param group 2)", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_title("Learning Rate Schedule (Cosine Decay)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    # Metrics Summary Table (based on best mAP@0.5)
    axes[1].axis("off")
    best_row_idx = df["metrics/mAP50(B)"].idxmax()
    best_epoch = int(df["epoch"].iloc[best_row_idx])

    summary_data = [
        ["Metric", "Best Value", "Epoch"],
        [
            "Precision",
            f"{df['metrics/precision(B)'].iloc[best_row_idx]:.4f}",
            str(best_epoch),
        ],
        [
            "Recall",
            f"{df['metrics/recall(B)'].iloc[best_row_idx]:.4f}",
            str(best_epoch),
        ],
        [
            "mAP@0.5",
            f"{df['metrics/mAP50(B)'].iloc[best_row_idx]:.4f}",
            str(best_epoch),
        ],
        [
            "mAP@0.5:0.95",
            f"{df['metrics/mAP50-95(B)'].iloc[best_row_idx]:.4f}",
            str(best_epoch),
        ],
        [
            "Box Loss",
            f"{df['val/box_loss'].iloc[best_row_idx]:.4f}",
            str(best_epoch),
        ],
        [
            "Class Loss",
            f"{df['val/cls_loss'].iloc[best_row_idx]:.4f}",
            str(best_epoch),
        ],
    ]
    table = axes[1].table(
        cellText=summary_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    axes[1].set_title(
        "Best Model Performance Summary", fontsize=12, fontweight="bold", pad=20
    )

    plt.tight_layout()
    out_path = exp_dir / "plots" / f"training_dynamics_{run_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[plot] Saved:", out_path.name)

    # Confusion Matrix (if available)
    confusion_matrix_path = results_csv.parent / "confusion_matrix.png"
    if confusion_matrix_path.exists():
        import shutil

        dst_cm = exp_dir / "plots" / f"confusion_matrix_{run_name}.png"
        shutil.copy(confusion_matrix_path, dst_cm)
        print("[plot] Copied:", dst_cm.name)


def save_final_report(exp_dir: Path, results_csv: Path):
    """Generate comprehensive training report."""
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    best_row_idx = df["metrics/mAP50(B)"].idxmax()
    best_epoch = int(df["epoch"].iloc[best_row_idx])
    best_metrics = df.iloc[best_row_idx]
    run_name = exp_dir.name

    report = f"""
=============================================================================
              YOLOV8 AERIAL PERSON DETECTION - TRAINING REPORT
                      OPTIMIZED RUN (YOLOv8m + 800px)
=============================================================================

EXPERIMENT DETAILS
------------------
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Run Name: {run_name}
Dataset: VisDrone (~6K images, person class only)
Model: YOLOv8m (Medium - 25M parameters)
Device: {DEVICE}
Input Size: 800px (multi-scale / strong aug)
Batch Size: 6
Total Epochs: {len(df)}

BEST MODEL PERFORMANCE (Epoch {best_epoch})
--------------------------------------------
Detection Metrics:
  • Precision @ IoU=0.5:     {best_metrics['metrics/precision(B)']:.4f}
  • Recall @ IoU=0.5:        {best_metrics['metrics/recall(B)']:.4f}
  • mAP @ IoU=0.5:           {best_metrics['metrics/mAP50(B)']:.4f}
  • mAP @ IoU=0.5:0.95:      {best_metrics['metrics/mAP50-95(B)']:.4f}

Loss Values:
  • Box Loss (val):          {best_metrics['val/box_loss']:.4f}
  • Classification Loss:     {best_metrics['val/cls_loss']:.4f}
  • DFL Loss:                {best_metrics['val/dfl_loss']:.4f}

PROJECT TARGETS vs ACHIEVED
----------------------------
  Target mAP@0.5:    0.55 - 0.65
  Achieved:          {best_metrics['metrics/mAP50(B)']:.4f} {"✓ PASSED" if best_metrics['metrics/mAP50(B)'] >= 0.55 else "✗ BELOW TARGET"}
  
  Target FPS:        ≥ 15 FPS (to be tested in inference stage)

TRAINING DYNAMICS
------------------
  • Total Training Time:     Check ultralytics logs
  • Final Learning Rate:     {df['lr/pg0'].iloc[-1]:.6f}
  • Early Stopping:          {"Yes" if len(df) < 100 else "No (completed all epochs)"}

OPTIMIZATION CHANGES FROM BASELINE
-----------------------------------
  ✓ Model: YOLOv8n (3M) → YOLOv8m (25M params)  ← 8x more capacity
  ✓ Resolution: 640px → 800px (more pixels per person)
  ✓ Epochs: 50 → 100
  ✓ Optimizer: SGD → AdamW (better for small objects)
  ✓ Learning rate: 0.01 → 0.02
  ✓ IoU threshold: 0.7 → 0.5 (more tolerant in aerial scenes)

AUGMENTATION STRATEGY (AGGRESSIVE)
-----------------------------------
  ✓ Strong HSV Color Jitter (h=0.04, s=0.9, v=0.7)
      → robust to extreme lighting and neon clothing
  ✓ High-resolution training @ 800px
      → better for small person detection at high altitude
  ✓ Mosaic 100% + MixUp 20% + Copy-Paste 30%
      → maximum context variation and small object focus
  ✓ Enhanced geometric transforms (rotation, translation, perspective)
      → robust to varying drone angles and movement
  ✓ IoU threshold 0.5 for difficult aerial scenarios

NEXT STEPS
----------
1. Test inference speed on RTX A2000 (target: ≥15 FPS @ 800px)
2. If FPS too low: export to TensorRT or fallback to YOLOv8s
3. Add nature/wilderness dataset for fine-tuning (empty-world prior)
4. Integrate DeepSORT/ByteTrack for multi-object tracking
5. Evaluate on UAVDT test set:
   - Target IDF1 ≥ 0.55
   - Target MOTA ≥ 0.35
   - Target ID-switches ≤ 5%

FILES GENERATED
---------------
  • Best weights:     {exp_dir}/weights/best_{run_name}.pt
  • Last weights:     {exp_dir}/weights/last_{run_name}.pt
  • Training curves:  {exp_dir}/plots/
  • CSV logs:         {exp_dir}/logs/results_{run_name}.csv
  • Metrics log:      {exp_dir}/logs/all_metrics_{run_name}.txt

=============================================================================
"""

    report_file = exp_dir / f"training_report_{run_name}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"[report] Saved to {report_file}")


def save_all_metrics_text(
    exp_dir: Path, results_csv: Path, val_metrics: dict, best_weights_path: Path
):
    """
    Save a plain-text file that contains:
      - all per-epoch training/validation metrics (from results.csv)
      - the final validation metrics of the best model
    """
    run_name = exp_dir.name
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    out_path = exp_dir / "logs" / f"all_metrics_{run_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "=====================================================================\n"
            "                YOLOv8 Training & Test Metrics (Full Log)\n"
            "                    OPTIMIZED RUN (YOLOv8m @ 800px)\n"
            "=====================================================================\n\n"
        )
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Experiment Directory: {exp_dir}\n")
        f.write(f"Best Weights: {best_weights_path}\n")
        f.write(f"Results CSV: {results_csv}\n\n")

        f.write("---- Per-epoch training & validation metrics (from results.csv) ----\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        f.write("---- Final validation metrics on best model ----\n\n")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                f.write(f"{key:12s}: {value:.6f}\n")
            else:
                f.write(f"{key:12s}: {value}\n")

        f.write("\nEnd of metrics log.\n")

    print(f"[metrics] Saved full metrics log to {out_path}")


def main():
    experiment_name = "yolov8m_visdrone_optimized"
    exp_dir = setup_experiment_dir(experiment_name)
    run_name = exp_dir.name

    data_cfg = PROJECT_ROOT / "cfg" / "visdrone_person.yaml"
    print(f"[train] Using data config: {data_cfg}")

    # Map DEVICE to ultralytics 'device' argument
    if DEVICE.type == "cuda":
        device_arg = DEVICE.index if DEVICE.index is not None else 0
    else:
        device_arg = "cpu"
    print(f"[train] Using device: {DEVICE} (ultralytics arg: {device_arg})")

    # Training configuration (for logging only)
    train_config = {
        "model": "yolov8m.pt",
        "dataset": "VisDrone (person only)",
        "epochs": 100,
        "imgsz": 800,
        "batch": 6,
        "device": str(DEVICE),
        "augmentation": {
            "hsv": {"h": 0.04, "s": 0.9, "v": 0.7},
            "geometric": {
                "degrees": 15.0,
                "scale": 0.9,
                "translate": 0.3,
                "perspective": 0.002,
            },
            "mosaic": 1.0,
            "mixup": 0.2,
            "copy_paste": 0.3,
        },
        "optimizer": {
            "name": "AdamW",
            "lr0": 0.02,
            "lrf": 0.01,
            "cos_lr": True,
            "warmup_epochs": 5,
            "weight_decay": 0.0005,
        },
        "detection": {"iou": 0.5},
        "early_stopping": {"patience": 20},
        "target_metrics": {
            "mAP@0.5": "0.55-0.65",
            "FPS": "≥15",
            "IDF1": "≥0.55 (tracking stage)",
            "MOTA": "≥0.35 (tracking stage)",
        },
    }
    save_training_config(exp_dir, train_config)

    # Initialize YOLOv8m
    model = YOLO("yolov8m.pt")

    print("\n" + "=" * 80)
    print("  🚀 OPTIMIZED TRAINING RUN")
    print("  Model: YOLOv8m (25M params) | Resolution: 800px | Epochs: 100")
    print("  Optimizer: AdamW | IoU=0.5 | Mosaic=1.0 | Strong Augmentation")
    print("=" * 80 + "\n")

    # NOTE: 'accumulate' is NOT a valid argument in this Ultralytics version,
    # so we simply omit it here.
    results = model.train(
        data=str(data_cfg),
        epochs=100,
        imgsz=800,
        batch=6,
        device=device_arg,
        workers=2,
        amp=True,  # mixed precision
        # Project structure
        project=str(exp_dir / "runs"),
        name="train",
        exist_ok=True,
        # Optimization / scheduler
        optimizer="AdamW",
        lr0=0.02,
        lrf=0.01,
        cos_lr=True,
        patience=20,
        warmup_epochs=5,
        warmup_momentum=0.5,
        weight_decay=0.0005,
        # No extra caching (800px + big model)
        cache=False,
        # Augmentation
        hsv_h=0.04,
        hsv_s=0.9,
        hsv_v=0.7,
        degrees=15.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        scale=0.9,
        translate=0.3,
        perspective=0.002,
        # Detection / training behaviour
        iou=0.5,
        close_mosaic=15,
        classes=[0],  # person only
        # Logging
        verbose=True,
        plots=True,
    )

    print("\n" + "=" * 80)
    print("  ✓ TRAINING FINISHED")
    print("=" * 80 + "\n")

    # Copy weights to experiment directory
    import shutil

    train_dir = exp_dir / "runs" / "train"
    best_weights_run_path = train_dir / "weights" / "best.pt"
    last_weights_run_path = train_dir / "weights" / "last.pt"

    best_dst = exp_dir / "weights" / f"best_{run_name}.pt"
    last_dst = exp_dir / "weights" / f"last_{run_name}.pt"

    if best_weights_run_path.exists():
        shutil.copy(best_weights_run_path, best_dst)
        if last_weights_run_path.exists():
            shutil.copy(last_weights_run_path, last_dst)
        print(f"[weights] Copied best/last weights to {exp_dir / 'weights'}")
    else:
        print(
            "[weights][WARN] best.pt not found in run directory. Skipping weight copy."
        )

    # Copy and plot results
    results_csv_run = train_dir / "results.csv"
    if results_csv_run.exists():
        dst_results_csv = exp_dir / "logs" / f"results_{run_name}.csv"
        shutil.copy(results_csv_run, dst_results_csv)
        print("[logs] Copied results.csv ->", dst_results_csv.name)

        # Generate custom plots
        plot_training_metrics(dst_results_csv, exp_dir)
    else:
        print("[logs][WARN] results.csv not found. Skipping custom plots.")
        dst_results_csv = None

    # Validate best model (if available)
    if best_dst.exists() and dst_results_csv is not None:
        print("\n[validation] Running final validation on best model...")
        best_model = YOLO(best_dst)
        val_results = best_model.val(data=str(data_cfg), split="val", plots=True)

        # Save validation metrics
        val_metrics = {
            "precision": float(val_results.box.p[0]),
            "recall": float(val_results.box.r[0]),
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "fitness": float(val_results.fitness),
        }

        val_metrics_path = exp_dir / "logs" / f"validation_metrics_{run_name}.json"
        with open(val_metrics_path, "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)
        print(f"\n[validation] Final Metrics:")
        print(f"  • Precision:     {val_metrics['precision']:.4f}")
        print(f"  • Recall:        {val_metrics['recall']:.4f}")
        print(f"  • mAP@0.5:       {val_metrics['mAP50']:.4f}")
        print(f"  • mAP@0.5:0.95:  {val_metrics['mAP50-95']:.4f}")
        print(f"  • Metrics saved to: {val_metrics_path}")

        # Training report and full metrics log
        save_final_report(exp_dir, dst_results_csv)
        save_all_metrics_text(exp_dir, dst_results_csv, val_metrics, best_dst)
    else:
        print(
            "\n[validation][WARN] best weights or results.csv missing. "
            "Skipping final validation and report."
        )

    print(f"\n{'=' * 80}")
    print(f"✓ All results saved to: {exp_dir}")
    print(f"  Target mAP@0.5: 0.55–0.65 – check training_report for final score.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
