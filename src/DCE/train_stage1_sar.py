#!/usr/bin/env python3
"""
Stage-1 Training: DCE-YOLOv8m Joint Training (VisDrone + SAR + Background)
Optimiert für Paperspace Resumption und stabile, niedrige Learning Rate.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# --- Resumption-Logik (Paperspace/Limit) ---
def get_resume_path(project_name, run_name):
    """Prüft, ob ein unterbrochenes Training fortgesetzt werden kann."""
    run_dir = Path(project_name) / run_name
    last_pt = run_dir / 'weights' / 'last.pt'
    if last_pt.exists():
        print(f"🔄 Checkpoint gefunden: {last_pt}. Das Training wird fortgesetzt.")
        return True 
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-yaml", default="cfg/yolov8m_dce.yaml")
    parser.add_argument("--pretrained", default="visdrone_pretrained.pt")
    parser.add_argument("--data", default="cfg/sar_composite.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4) 
    parser.add_argument("--imgsz", type=int, default=800)
    
    args = parser.parse_args()

    PROJECT_NAME = "experiments/stage1_sar"
    RUN_NAME = "dce_yolov8m_composite_run"
    
    resume_flag = get_resume_path(PROJECT_NAME, RUN_NAME)

    # 1. Modell initialisieren
    if resume_flag:
        # Fortsetzen: Lade direkt den letzten Checkpoint
        model = YOLO(Path(PROJECT_NAME) / RUN_NAME / 'weights' / 'last.pt')
    else:
        # Neues Training: Lade Architektur aus YAML und Gewichte
        model = YOLO(args.model_yaml)
        if Path(args.pretrained).exists():
            print(f"   Lade initiale Gewichte von {args.pretrained}...")
            model.load(args.pretrained)

    # 2. Training Starten
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=PROJECT_NAME,
        name=RUN_NAME,
        
        # --- RESUMPTION (Paperspace Fix) ---
        resume=resume_flag, 
        
        # --- STABILITÄT (LR Fix für NaN-Werte) ---
        optimizer='AdamW',
        lr0=0.001,          # Sehr niedrig (stabilisiert DCE-Architektur)
        warmup_epochs=5.0,  # Längeres Warmup
        patience=30,        # Etwas mehr Geduld bei Misch-Daten
        
        # --- ROBUSTHEIT (SAR Augmentations für Camouflage/Neon) ---
        mosaic=1.0,         
        mixup=0.15,
        degrees=15.0,
        copy_paste=0.3,
        
        hsv_h=0.015,
        hsv_s=0.7,          # Erhöhte Sättigung: Robust gegen Camouflage/Grelles Licht
        hsv_v=0.4,          # Erhöhte Helligkeit: Robust gegen starke Schatten/Sonneneinstrahlung
        
        # System
        device=0,
        save=True,
        plots=True,
        exist_ok=True
    )

    print("✅ Training beendet/abgebrochen. Checkpoint unter", Path(PROJECT_NAME) / RUN_NAME)

if __name__ == "__main__":
    main()