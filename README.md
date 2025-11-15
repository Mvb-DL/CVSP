# CHANGELOG — SAR Person Detection (YOLOv8)

> Lückenlose Verlauf-/Änderungsdoku der Trainings & Validierungen (ohne Begründungen), inkl. Datensätze, Parameterstände und Ergebnisse.  
> Zeitraum: Nov 2025

---

## 0) Infrastruktur & Umgebung

**Lokal (Windows, RTX A2000 8GB)**
- Python 3.13.4, Torch 2.9.0+cu126, Ultralytics 8.3.227
- Default: `batch=6`, `workers=2`, `imgsz=800`
- Datenträger: **D:\data** (SSD) für alle Datasets (ab Stage‑3 konsequent)
- Stabilitäts-Patch (Val/Low-VRAM):  
  `PYTORCH_ALLOC_CONF=max_split_size_mb:64,expandable_segments:False`

**Paperspace (Quadro RTX 5000 ~16GB)**
- Python 3.11.7, Torch 2.1.1+cu121, Ultralytics 8.3.227/228
- Default: `workers=4`, `imgsz=800`

**Wichtige Skripte**
- `src/train_yolov8_visdrone.py` (Stage‑1)
- `src/finetuning_yolov8_visdrone.py` (Stage‑2)
- `src/prepare_external_sar_datasets.py` (Konvertierung → YOLOv8)
- `src/stage3_train_sar.py` (Stage‑3 + Val‑Only‑Mode, FP‑only Fallback)
- `src/config.py` (globale Pfade/Device; **DATA_ROOT=D:/data**)

---

## 1) Stage‑1 — Basismodell VisDrone (reine Personendetektion)

**Modell/Setup**
- YOLOv8n → **YOLOv8m (25M Params)**
- `imgsz`: 640 → **800**
- Optimizer: SGD → **AdamW**
- Epochen: 5 → **50** → **70**
- Eval/IoU: **0.5**
- Aug (stark): `hsv_h=0.04, hsv_s=0.9, hsv_v=0.7, degrees=15, translate=0.3, perspective=0.002, mosaic=1.0, mixup=0.2, copy_paste=0.3`

**Runs & Ergebnisse (VisDrone val)**
- **50 Ep (12.11.2025, RTX 5000)**  
  P=**0.638**, R=**0.460**, mAP50=**0.5105**, mAP50‑95=**0.1993**; ~18.2 ms/img
- **70 Ep (12.11.2025, RTX 5000)**  
  Best(Ep66): P≈0.6805, R≈0.4995, mAP50=**0.5453**, mAP50‑95=**0.2023**  
  Final: P=0.6633, R=0.4896, mAP50=**0.5348**, mAP50‑95=**0.1966**

**Hinweise**
- Entfernt: ungültiges Argument `accumulate` (Ultralytics-Fehler).  
- Log „yolo11n.pt“ Download stammt von AMP/Font‑Check; **Train‑Backbone bleibt v8m**.  
- Einmalige „NMS time limit exceeded“-Warnung in frühem Val‑Durchlauf.

---

## 2) Stage‑2 — VisDrone Re‑Finetuning (12 Epochen)

**Startgewichte**
- `best` aus Stage‑1 (70‑Ep Run)

**Parameter**
- Epochen: **12**
- `lr0=0.005`, `lrf=0.01`, `cos_lr=True`, `warmup_epochs=1`, `weight_decay=5e-4`
- Aug (entschärft): `hsv_h=0.02, hsv_s=0.8, hsv_v=0.55, degrees=10, mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=1`
- Eval IoU: **0.6**, `max_det=150`

**Ergebnis (VisDrone val, 12.11.2025, RTX 5000)**
- P=**0.681**, R=**0.509**, mAP50=**0.557**, mAP50‑95=**0.218**

---

## 3) Datensätze (SAR‑Erweiterung) & Vorbereitung (ab 13.11.2025)

**Quellen & Ausgabe (unter D:\data)**
- **SARD** (Roboflow YOLO, Person=0) → **D:\data\SARDYOLO** → YAML: `cfg/sardyolo.yaml`
- **HERIDAL** (Roboflow YOLO, Person=0) → **D:\data\HERIDALYOLO** → YAML: `cfg/heridalyolo.yaml`
- **NTUT4K** (i. d. R. **negativ**, keine Labels) → **D:\data\NTUT4KYOLO** → YAML: `cfg/ntut4kyolo.yaml`
- **Zenodo 7740081** → keine Bilder gefunden

**Konvertierung (`src/prepare_external_sar_datasets.py`)**
- Automatische Formatdetektion: *roboflow_yolo*, *plain_yolo*, *coco*, *voc*, *negative*
- Filtert nur **person** (Klasse 0); negative Bilder → leere Labeldateien
- Kombi‑YAMLs:
  - `cfg/sar_person_mix.yaml` (SARD + HERIDAL + NTUT4K)
  - `cfg/sar_pos_only.yaml` (nur **SARD + HERIDAL**)

**Bekannte Warnung**
- Roboflow Mixed (Boxes/Segments): Segmente werden verworfen, es werden **nur Boxes** genutzt.

---

## 4) Stage‑3 — SAR Finetuning & Validierungen (lokal, A2000)

**Skript `src/stage3_train_sar.py`**
- Train‑Args: `--data`, `--weights`, `--epochs` (Default **20**), `--batch` (Default **6**), `--imgsz` (Default **800**)
- Val‑Only: `--val-only`, `--val-batch`, `--val-workers`, `--val-imgsz`, `--val-plots`
- Optim: AdamW, `lr0=0.003`, `lrf=0.01`, `cos_lr=True`, `nbs=64`, `weight_decay=5e-4`, `warmup_epochs=1`, `patience=8`
- Aug: `hsv_h=0.03, hsv_s=0.8, hsv_v=0.6, degrees=8, scale=0.85, translate=0.2, mosaic=0.15, mixup=0.0, copy_paste=0.1, close_mosaic=3`
- Eval/NMS: `iou=0.5`, `max_det=150`, `classes=[0]`
- No‑label‑Fallback (Val): **FP‑only** Kennzahlen

**Stabilität (Val/OOM)**
- Val‑Only mit `--val-batch 2`, `--val-workers 0`, plus ENV Patch (s. oben).

### 4.1 Kurz‑Finetuning „Positiv‑Mix“ (SARD + HERIDAL), 12 Epochen — 13.11.2025
- **Val (HERIDAL, light)**: P≈**0.747**, R≈**0.628**, mAP50≈**0.701**, mAP50‑95≈**0.295**
- **Voll‑Val (HERIDAL) nach Training**: P≈**0.760–0.766**, R≈**0.656–0.664**, mAP50≈**0.714–0.719**, mAP50‑95≈**0.284–0.293**
- **Val (VisDrone)**: P≈**0.510**, R≈**0.303**, mAP50≈**0.315**, mAP50‑95≈**0.117**
- **Val (NTUT4K, FP‑only)**: frühe Messung `avg_fp_per_image ≈ 6.58`

### 4.2 Kurz‑Rewarm (SARD + HERIDAL), **4 Epochen** — 14.11.2025
- **Train‑Val (HERIDAL)**: P=**0.792**, R=**0.608**, mAP50=**0.696**, mAP50‑95=**0.280**
- **Externe Val (HERIDAL, 2276 imgs)**: P=**0.826**, R=**0.583**, mAP50=**0.728**, mAP50‑95=**0.343**
- **NTUT4K (FP‑only, aktualisiert)**: `avg_fp_per_image ≈ 0.0122` (10 Predictions / 819 Bilder)

---

## 5) Schwellen‑Sweeps (Conf‑Ablenkungen)

**SARD (val)**
- **conf=0.25:** P=**0.790**, R=**0.662**, mAP50=**0.769**, mAP50‑95=**0.364**, ~34.5 ms/img  
- **conf=0.30:** P=**0.832**, R=**0.641**, mAP50=**0.767**, mAP50‑95=**0.367**, ~40.4 ms/img  
- **conf=0.40:** P=**0.919**, R=**0.545**, mAP50=**0.743**, mAP50‑95=**0.368**, ~35.0 ms/img

**HERIDAL (val)**
- **conf=0.25:** P=**0.625**, R=**0.362**, mAP50=**0.484**, mAP50‑95=**0.224**, ~25.1 ms/img  
- **conf=0.30:** P=**0.677**, R=**0.285**, mAP50=**0.476**, mAP50‑95=**0.227**, ~25.2 ms/img  
- **conf=0.40:** P=**0.775**, R=**0.171**, mAP50=**0.474**, mAP50‑95=**0.233**, ~25.4 ms/img

**NTUT4K (negativ, No‑Label)**
- **conf=0.30:** keine mAP; **FP‑only** – repräsentativ `avg_fp_per_image ≈ 0.0122`

---

## 6) Aktuelle Presets (Betrieb)

- **Default/Robust:** `conf=0.30`, `iou=0.50`, `max_det=150`, `imgsz=800`
- **High‑Recall (Suchmodus):** `conf=0.25`, `iou=0.50`, `max_det=150`, `imgsz=800`
- Optional „Slow‑but‑Sharp“ Val‑Prüfung: `imgsz=896/960` (nur Val/Inference)

---

## 7) Konfigurationsstände (kompakt)

**`src/config.py` (Windows lokal)**
- `DATA_ROOT = Path("D:/data").resolve()`  
- `VISDRONE_ROOT = DATA_ROOT / "VisdroneYOLO"`  
- `EXTERNAL_ROOT = DATA_ROOT / "external"`  
- `SARMIX_ROOT = DATA_ROOT / "SARmix"`  
- `DEVICE = cuda:0` (A2000)  
- Warn‑Log falls Pfad nicht existiert

**`src/stage3_train_sar.py` (aktuell)**
- Train‑Args: `--data`, `--weights`, `--epochs`, `--batch`, `--imgsz`
- Val‑Only: `--val-only`, `--val-batch`, `--val-workers`, `--val-imgsz`, `--val-plots`
- Trainer: Optim/Aug/Eval wie oben; JSON‑Metriken + FP‑only Fallback

**YAMLs**
- `cfg/sar_pos_only.yaml` (SARDYOLO + HERIDALYOLO)  
- `cfg/sar_person_mix.yaml` (SARD + HERIDAL + NTUT4K)  
- `cfg/sardyolo.yaml`, `cfg/heridalyolo.yaml`, `cfg/ntut4kyolo.yaml` (Einzelsets)

---

## 8) Bekannte Warnungen & Hinweise
- Roboflow mixed (Boxes/Segments): Segmente werden **ignoriert** → nur Boxen.
- NTUT4K: **keine Labels** → keine mAP; **FP‑only**‑Bewertung.
- AMP/Assets‑Downloads in Logs (z. B. „yolo11n.pt“, „Arial.ttf“) **ändern das Train‑Backbone nicht**.
- Einmalige NMS‑Timeout‑Warnung (früher Lauf).

---

| Modell                         | SARD mAP50 |    SARD R | HERIDAL mAP50 | HERIDAL R | VISDRONE mAP50 |     Vis R | NTUT4K avg FP/img |
| ------------------------------ | ---------: | --------: | ------------: | --------: | -------------: | --------: | ----------------: |
| stage2_best                    |      0.427 |     0.080 |         0.364 |     0.226 |      **0.625** |     0.479 |          **7.92** |
| stage3_4ep_best                |  **0.795** |     0.665 |         0.491 |     0.362 |          0.470 | **0.001** |         **0.012** |
| stage3_sard_heridal_best       |      0.785 | **0.681** |     **0.602** | **0.561** |          0.483 |     0.235 |              6.58 |
| stage3_sar_recallpush_3ep_best |      0.784 |     0.666 |         0.563 |     0.478 |          0.472 |     0.211 |              5.74 |


*Stand: 14.11.2025*
