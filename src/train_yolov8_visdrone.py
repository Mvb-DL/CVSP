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

    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "weights").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    print(f"[setup] Experiment directory: {exp_dir}")
    return exp_dir


def save_training_config(exp_dir: Path, config: dict):
    """Save training configuration to JSON."""
    config_file = exp_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[config] Saved to {config_file}")


def plot_training_metrics(results_csv: Path, exp_dir: Path):
    """Plot comprehensive training metrics in English."""
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training and Validation Losses', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Bounding Box Regression Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Distribution Focal Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    if 'train/loss' in df.columns:
        axes[1, 1].plot(df['epoch'], df['train/loss'], label='Total Train Loss', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Total Training Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(exp_dir / "plots" / "loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[plot] Saved: loss_curves.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detection Performance Metrics', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(
        df['epoch'], df['metrics/precision(B)'],
        label='Precision', color='green', linewidth=2, marker='o', markersize=3
    )
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision @ IoU=0.5 (Person Class)')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(
        df['epoch'], df['metrics/recall(B)'],
        label='Recall', color='blue', linewidth=2, marker='o', markersize=3
    )
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall @ IoU=0.5 (Person Class)')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].plot(
        df['epoch'], df['metrics/mAP50(B)'],
        label='mAP@0.5', color='red', linewidth=2, marker='s', markersize=3
    )
    axes[1, 0].axhline(y=0.55, color='orange', linestyle='--',
                       linewidth=1.5, label='Target (0.55-0.65)')
    axes[1, 0].axhline(y=0.65, color='orange', linestyle='--', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP@0.5')
    axes[1, 0].set_title('Mean Average Precision @ IoU=0.5')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(
        df['epoch'], df['metrics/mAP50-95(B)'],
        label='mAP@0.5:0.95', color='purple', linewidth=2, marker='^', markersize=3
    )
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP@0.5:0.95')
    axes[1, 1].set_title('Mean Average Precision @ IoU=0.5:0.95')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(exp_dir / "plots" / "detection_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[plot] Saved: detection_metrics.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Dynamics', fontsize=16, fontweight='bold')
    
    if 'lr/pg0' in df.columns:
        axes[0].plot(df['epoch'], df['lr/pg0'], label='LR (param group 0)', linewidth=2)
    if 'lr/pg1' in df.columns:
        axes[0].plot(df['epoch'], df['lr/pg1'], label='LR (param group 1)', linewidth=2)
    if 'lr/pg2' in df.columns:
        axes[0].plot(df['epoch'], df['lr/pg2'], label='LR (param group 2)', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_title('Learning Rate Schedule (Cosine Decay)')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    
    axes[1].axis('off')
    best_row_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = int(df['epoch'].iloc[best_row_idx])

    summary_data = [
        ['Metric', 'Best Value', 'Epoch'],
        ['Precision', f"{df['metrics/precision(B)'].iloc[best_row_idx]:.4f}", str(best_epoch)],
        ['Recall', f"{df['metrics/recall(B)'].iloc[best_row_idx]:.4f}", str(best_epoch)],
        ['mAP@0.5', f"{df['metrics/mAP50(B)'].iloc[best_row_idx]:.4f}", str(best_epoch)],
        ['mAP@0.5:0.95', f"{df['metrics/mAP50-95(B)'].iloc[best_row_idx]:.4f}", str(best_epoch)],
        ['Box Loss', f"{df['val/box_loss'].iloc[best_row_idx]:.4f}", str(best_epoch)],
        ['Class Loss', f"{df['val/cls_loss'].iloc[best_row_idx]:.4f}", str(best_epoch)],
    ]
    table = axes[1].table(
        cellText=summary_data, cellLoc='left', loc='center',
        colWidths=[0.4, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1].set_title('Best Model Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(exp_dir / "plots" / "training_dynamics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[plot] Saved: training_dynamics.png")
    
    confusion_matrix_path = results_csv.parent / "confusion_matrix.png"
    if confusion_matrix_path.exists():
        import shutil
        shutil.copy(confusion_matrix_path, exp_dir / "plots" / "confusion_matrix.png")
        print("[plot] Copied: confusion_matrix.png")


def save_final_report(model, exp_dir: Path, results_csv: Path):
    """Generate comprehensive training report."""
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    best_row_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = int(df['epoch'].iloc[best_row_idx])
    best_metrics = df.iloc[best_row_idx]
    
    report = f"""
=============================================================================
              YOLOV8 AERIAL PERSON DETECTION - TRAINING REPORT
=============================================================================

EXPERIMENT DETAILS
------------------
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: VisDrone (~6K images, person class only)
Model: YOLOv8n (Nano)
Device: {DEVICE}
Input Size: 640 (multi-scale training enabled)
Batch Size: 8
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
  • Early Stopping:          {"Yes" if len(df) < 5 else "No (completed all epochs)"}

AUGMENTATION STRATEGY
---------------------
  ✓ Moderate HSV Color Jitter (h=0.02, s=0.8, v=0.5) 
      → robust to bright / neon clothing without overfitting to extreme colors
  ✓ Multi-scale training around 640px 
      → simulates altitude variation
  ✓ Mosaic (70%) + MixUp (15%) 
      → robust to complex backgrounds and context changes
  ✓ Light Motion Blur (1%) 
      → drone movement and slight camera shake
  ✓ Perspective warping and geometric transforms 
      → varying viewing angles and camera poses

NEXT STEPS
----------
1. Test inference speed on RTX A2000 (target: ≥15 FPS @ 640px)
2. Add nature/wilderness dataset for fine-tuning (empty-world prior)
3. Integrate DeepSORT/ByteTrack for multi-object tracking
4. Evaluate on UAVDT test set:
   - Target IDF1 ≥ 0.55
   - Target MOTA ≥ 0.35
   - Target ID-switches ≤ 5%

FILES GENERATED
---------------
  • Best weights:     {exp_dir}/weights/best.pt
  • Last weights:     {exp_dir}/weights/last.pt
  • Training curves:  {exp_dir}/plots/
  • CSV logs:         {exp_dir}/logs/results.csv
  • This report:      {exp_dir}/training_report.txt

=============================================================================
"""
    
    report_file = exp_dir / "training_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n{report}")
    print(f"[report] Saved to {report_file}")


def main():

    experiment_name = "yolov8n_visdrone_test"  
    exp_dir = setup_experiment_dir(experiment_name)
    
    data_cfg = PROJECT_ROOT / "cfg" / "visdrone_person.yaml"
    print(f"[train] Using data config: {data_cfg}")

    if DEVICE.type == "cuda":
        device_arg = DEVICE.index if DEVICE.index is not None else 0
    else:
        device_arg = "cpu"
    print(f"[train] Using device: {DEVICE} (ultralytics arg: {device_arg})")

    train_config = {
        "model": "yolov8n.pt",
        "dataset": "VisDrone (person only)",
        "epochs": 5,  # TEST MODE
        "imgsz": 640,
        "multi_scale": True,
        "batch": 8,
        "device": str(DEVICE),
        "augmentation": {
            "hsv": {"h": 0.02, "s": 0.8, "v": 0.5},
            "geometric": {"degrees": 10.0, "scale": 0.6, "translate": 0.2, "perspective": 0.001},
            "mosaic": 0.7,
            "mixup": 0.15,
            "blur": 0.01,
        },
        "optimizer": {
            "lr0": 0.01,
            "lrf": 0.01,
            "cos_lr": True,
            "warmup_epochs": 1,  
        },
        "target_metrics": {
            "mAP@0.5": "0.55-0.65",
            "FPS": "≥15",
            "IDF1": "≥0.55 (tracking stage)",
            "MOTA": "≥0.35 (tracking stage)",
        }
    }
    save_training_config(exp_dir, train_config)

    model = YOLO("yolov8n.pt")

    print("\n" + "="*80)
    print("STARTING TEST TRAINING (5 EPOCHS)")
    print("="*80 + "\n")
    
    results = model.train(
        data=str(data_cfg),
        epochs=5,  # ← TEST MODE
        imgsz=640,
        multi_scale=True,
        batch=8,
        device=device_arg,
        workers=4,
        
        project=str(exp_dir / "runs"),
        name="train",
        exist_ok=True,
        
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        patience=3,  
        warmup_epochs=1,  
        warmup_momentum=0.5,
        
        cache="disk",
        
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        degrees=10.0,
        flipud=0.0, fliplr=0.5,
        mosaic=0.7,
        mixup=0.15,
        scale=0.6,
        translate=0.2,
        perspective=0.001,
        blur=0.01,
        
        label_smoothing=0.05,
        close_mosaic=2, 
        
        classes=[0],  

        verbose=True,
        plots=True,  
    )

    print("\n" + "="*80)
    print("TRAINING FINISHED")
    print("="*80 + "\n")
    
    import shutil
    train_dir = exp_dir / "runs" / "train"
    best_weights_run_path = train_dir / "weights" / "best.pt"
    last_weights_run_path = train_dir / "weights" / "last.pt"

    if best_weights_run_path.exists():
        shutil.copy(best_weights_run_path, exp_dir / "weights" / "best.pt")
        if last_weights_run_path.exists():
            shutil.copy(last_weights_run_path, exp_dir / "weights" / "last.pt")
        print(f"[weights] Copied to {exp_dir / 'weights'}")
    else:
        print("[weights][WARN] best.pt not found in run directory. Skipping weight copy.")

    results_csv_run = train_dir / "results.csv"
    if results_csv_run.exists():
        dst_results_csv = exp_dir / "logs" / "results.csv"
        shutil.copy(results_csv_run, dst_results_csv)
        print("[logs] Copied results.csv")
    
        plot_training_metrics(dst_results_csv, exp_dir)
    else:
        print("[logs][WARN] results.csv not found. Skipping custom plots.")
        dst_results_csv = None
    
    best_weights_path = exp_dir / "weights" / "best.pt"
    if best_weights_path.exists() and dst_results_csv is not None:
        print("\n[validation] Running final validation on best model...")
        best_model = YOLO(best_weights_path)
        val_results = best_model.val(data=str(data_cfg), split="val", plots=True)
        
        val_metrics = {
            "precision": float(val_results.box.p[0]),
            "recall": float(val_results.box.r[0]),
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "fitness": float(val_results.fitness),
        }
        
        with open(exp_dir / "logs" / "validation_metrics.json", 'w') as f:
            json.dump(val_metrics, f, indent=2)
        
        print(f"\n[validation] Final Metrics:")
        print(f"  • Precision:     {val_metrics['precision']:.4f}")
        print(f"  • Recall:        {val_metrics['recall']:.4f}")
        print(f"  • mAP@0.5:       {val_metrics['mAP50']:.4f}")
        print(f"  • mAP@0.5:0.95:  {val_metrics['mAP50-95']:.4f}")
        
        save_final_report(best_model, exp_dir, dst_results_csv)
    else:
        print("\n[validation][WARN] best.pt or results.csv missing. Skipping final validation and report.")
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {exp_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()