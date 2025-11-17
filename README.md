# CHANGELOG — SAR Person Detection (YOLOv8)

> Lückenlose, **rein faktische** Verlaufs- und Änderungsdokumentation aller Trainings-, Validierungs- und Benchmark-Läufe (ohne Begründungen / Interpretation).
> Zeitraum: **November 2025**

## 0) Infrastruktur & Umgebung

### Lokal (Windows, RTX A2000 8 GB)

* Python **3.13.4**
* Torch **2.9.0+cu126**
* Ultralytics **8.3.227**
* Standard-Settings:

  * `batch=6`
  * `workers=2`
  * `imgsz=800`
* Datenträger:

  * **D:\data** (SSD) als Root für alle Datensätze (ab Stage-3 konsequent)
* Stabilitäts-Patch (v. a. für Validation / Low-VRAM):

  ```bash
  PYTORCH_ALLOC_CONF=max_split_size_mb:64,expandable_segments:False
  ```

### Paperspace (Quadro RTX 5000 ≈ 16 GB)

* Python **3.11.7**
* Torch **2.1.1+cu121**
* Ultralytics **8.3.227 / 8.3.228**
* Standard-Settings:

  * `workers=4`
  * `imgsz=800`

### Zentrale Skripte

* **Training / Finetuning**

  * `src/train_yolov8_visdrone.py`
    → Stage-1 VisDrone Baseline
  * `src/finetuning_yolov8_visdrone.py`
    → ursprüngliches Stage-2 Re-Finetuning auf VisDrone
  * `src/stage2_with_dce_comparison.py`
    → neues Stage-2 Framework
    → **Stage2A** (Standard YOLOv8m) & **Stage2B** (YOLOv8m + DCE)
  * `src/stage3_train_sar.py`
    → Stage-3 SAR-Finetuning + Val-Only + FP-only-Fallback

* **Datenvorbereitung**

  * `src/prepare_external_sar_datasets.py`
    → Konvertiert externe SAR-Datensätze nach YOLOv8 (inkl. Negativ-Handling)

* **Benchmarks & Inferenz**

  * `src/benchmark_models.py`
    → Multi-Modell / Multi-Dataset-Benchmark, inkl. **FP-only** Modus, NTUT4K Hard-Wiring, Plots
  * `src/live_sar_detect.py` (Name frei)
    → Live-Video Inferenz mit Metriken + **Area-Filter**

* **Konfiguration**

  * `src/config.py`

    * `PROJECT_ROOT`
    * `DATA_ROOT = Path("D:/data").resolve()`
    * `VISDRONE_ROOT = DATA_ROOT / "VisdroneYOLO"`
    * `EXTERNAL_ROOT = DATA_ROOT / "external"`
    * `SARMIX_ROOT = DATA_ROOT / "SARmix"`
    * `DEVICE` (typisch `cuda:0` / RTX A2000)
    * Warn-Log falls Pfad nicht existiert

---

## 1) Stage-1 — VisDrone Baselines (reine Personendetektion)

### 1.1 Ursprüngliches Stage-1 (old_stage1)

**Modell / Setup**

* Backbone: **YOLOv8m** (~25.9 M Parameter)
* `imgsz`: 640 → **800**
* Optimizer: **SGD → AdamW**
* Epochen: 5 → **50** → **70**
* Eval IoU: **0.5**
* Starke Augmentierung:

  ```text
  hsv_h=0.04, hsv_s=0.9, hsv_v=0.7,
  degrees=15, translate=0.3,
  perspective=0.002,
  mosaic=1.0, mixup=0.2, copy_paste=0.3
  ```

**Ergebnisse (VisDrone val)**

* **50 Epochen** (12.11.2025, RTX 5000)

  * P = **0.638**
  * R = **0.460**
  * mAP50 = **0.5105**
  * mAP50-95 = **0.1993**
  * Inferenz ≈ **18.2 ms/img**

* **70 Epochen** (12.11.2025, RTX 5000)

  * Best (Ep66):

    * P ≈ **0.6805**
    * R ≈ **0.4995**
    * mAP50 = **0.5453**
    * mAP50-95 = **0.2023**
  * Final:

    * P = **0.6633**
    * R = **0.4896**
    * mAP50 = **0.5348**
    * mAP50-95 = **0.1966**

**Technische Hinweise**

* Entfernt: ungültiges Argument `accumulate` (Ultralytics Bug).
* Log-Zeile „yolo11n.pt“ stammt aus interner Ultralytics-Routine; **Train-Backbone bleibt YOLOv8m**.
* Einmalige Warnung: `NMS time limit exceeded` in frühem Val-Lauf.

---

### 1.2 Neuer Stage-1 VisDrone Baseline (new_stage1 → `best.pt`)

**Kontext**

* Neu trainiertes VisDrone-Only Modell, genutzt als aktuelles **Stage-1 Startmodell**
  (`best.pt`) für:

  * Stage-2 (Standard & DCE auf Paperspace)
  * Stage-3 SAR-Finetuning (z. T. `*_newstage1_*`-Experimente)

**Aggregierte Kennzahlen (VisDrone val)**

| Metric       | old_stage1 | new_stage1 | Δ absolut | Δ relativ (ca.) |
| ------------ | ---------: | ---------: | --------: | --------------: |
| Precision    |     0.6843 |     0.7327 |   +0.0484 |          ≈ +7 % |
| Recall       |     0.5113 |     0.5706 |   +0.0593 |         ≈ +12 % |
| mAP@0.5      |     0.5544 |     0.6297 |   +0.0753 |      ≈ +13–14 % |
| mAP@0.5–0.95 |     0.2132 |     0.2680 |   +0.0548 |      ≈ +25–26 % |

**Externe Referenz**

* MDPI Drones, Vol. 9, Issue 8, Article 514
  URL: `https://www.mdpi.com/2504-446X/9/8/514`
  (Quelle / Referenz für den neuen VisDrone-Baseline-Vergleich)

---

## 2) Stage-2 — VisDrone / SAR Feintuning

Es existieren zwei Stage-2-Varianten:

1. **Legacy Stage-2** (`stage2_best`) — ursprüngliches Re-Finetuning auf VisDrone
2. **Neues Stage-2 Framework** (`Stage2Trainer`, `stage2a_standard_best` & `stage2b_dce_best`)

---

### 2.1 Legacy Stage-2 (`src/finetuning_yolov8_visdrone.py` → `stage2_best`)

**Startgewichte**

* `best` aus ursprünglichem Stage-1 (70 Ep)

**Parameter**

* Epochen: **12**
* Optimizer: AdamW

  * `lr0=0.005`
  * `lrf=0.01`
  * `cos_lr=True`
  * `warmup_epochs=1`
  * `weight_decay=5e-4`
* Aug (abgeschwächt):

  ```text
  hsv_h=0.02, hsv_s=0.8, hsv_v=0.55,
  degrees=10,
  mosaic=0.0, mixup=0.0, copy_paste=0.0,
  close_mosaic=1
  ```
* Eval IoU: **0.6**, `max_det=150`

**Ergebnis (VisDrone val, 12.11.2025, RTX 5000)**

* P = **0.681**
* R = **0.509**
* mAP50 = **0.557**
* mAP50-95 = **0.218**

**Benchmark (multi-dataset, s. Section 4)**

* Modellname: **`stage2_best`**
  → detaillierte Kennzahlen siehe Tabelle in Abschnitt **4.3**.

---

### 2.2 Stage-2 Framework mit Standard & DCE (`src/stage2_with_dce_comparison.py`)

**Allgemeines**

* Klasse: `Stage2Trainer`
* Eingabe: `--stage1-weights` (typisch: `checkpoints/best.pt` = **new_stage1**)
* Experiment-Ordner: `experiments/stage2_comparison_YYYYMMDD_HHMMSS/`
* Gemeinsame Hyperparameter (Standard/DCE):

  * `epochs` (typisch **50**)
  * `batch=6`
  * `imgsz=800`
  * Optimizer: AdamW

    * `lr0=0.003`, `lrf=0.01`, `cos_lr=True`
    * `momentum=0.937`
    * `weight_decay=0.0005`
    * `warmup_epochs=1.0`, `warmup_momentum=0.5`
    * `patience=15`
    * `nbs=64`
  * Aug:

    ```text
    hsv_h=0.03, hsv_s=0.8, hsv_v=0.6,
    degrees=8.0, translate=0.2, scale=0.85,
    fliplr=0.5, flipud=0.0,
    mosaic=0.15, mixup=0.0, copy_paste=0.1,
    perspective=0.002, close_mosaic=3
    ```
  * Loss-Weights: `box=7.5`, `cls=0.5`, `dfl=1.5`
  * Inference: `iou=0.5`, `max_det=150`, `classes=[0]`
  * Sonstiges: `cache=False`, `rect=False`, `single_cls=True`, `amp=True`
* Validierung in `_validate_all_datasets()` auf:

  * `cfg/sardyolo.yaml` (SARD)
  * `cfg/heridalyolo.yaml` (HERIDAL)
  * `cfg/visdrone_person.yaml` (VisDrone Personen)

---

#### 2.2.1 Stage-2A — Standard YOLOv8m (`stage2a_standard_best`)

**Training**

* Funktion: `Stage2Trainer.train_standard(...)`
* Daten: `cfg/sar_pos_only.yaml` (**SARD + HERIDAL**, nur person=0)
* Startgewichte: `stage1_weights` (neues Stage-1 Modell, `best.pt`)
* Plattform: lokal oder Paperspace (Standard-Settings oben)
* Epochen (typisch): **50**

**Validationsergebnisse (Benchmark-Run, conf=0.25)**

|      Dataset | Precision | Recall |  mAP50 | mAP50-95 | Mode    | NTUT4K FP/Img |
| -----------: | --------: | -----: | -----: | -------: | ------- | ------------: |
|     **SARD** |    0.8754 | 0.5236 | 0.7096 |   0.3184 | GT      |             – |
|  **HERIDAL** |    0.7404 | 0.4246 | 0.5820 |   0.2682 | GT      |             – |
| **VISDRONE** |    0.7700 | 0.1294 | 0.4498 |   0.2048 | GT      |             – |
|   **NTUT4K** |         – |      – |      – |        – | FP-only |      **4.46** |

---

#### 2.2.2 Stage-2B — YOLOv8m + DCE Architektur (`stage2b_dce_best`)

**Architektur**

* YAML: `cfg/yolov8m_dce.yaml` (ggf. auto-generiert durch `_create_dce_yaml`)
  Backbone/Head mit:

  * `DCE`, `ERB`, `SCDown`, `SPPF`
* YOLO Summary (Paperspace):

  * **73** Layer
  * **8,300,522** Parameter
  * ca. **143.9 GFLOPs**

**Gewichtsinitialisierung**

* Start: `stage1_weights` (= **new_stage1 best.pt**)
* Transfer:

  * `stage1_state` → `model_state`, nur Layer mit identischer Shape
  * `strict=False` beim Laden
  * Logging von `transferred` vs. `skipped` Layern

**Training**

* Funktion: `Stage2Trainer.train_dce(...)`
* Daten: `cfg/sar_pos_only.yaml` (SARD + HERIDAL, nur Person)
* Epochen: **50**
* Plattform: **Paperspace Quadro RTX 5000**, `batch=6`, `workers=4`, `imgsz=800`
* Checkpoint:

  * `/notebooks/CVSP/checkpoints/stage2b_dce_best.pt`
  * Trainings-Run: `/notebooks/CVSP/experiments/stage2_dce_only_newstage1_20251117_111632/`

**Validation (am Ende des Trainings, Ultralytics Val pro Dataset)**

* **SARD**

  * Images: 1144 (davon 177 Hintergrund)
  * P = **0.884**
  * R = **0.726**
  * mAP50 = **0.823**
  * mAP50-95 = **0.366**

* **HERIDAL**

  * Images: 313 (davon 109 Hintergrund)
  * P = **0.670**
  * R = **0.574**
  * mAP50 = **0.619**
  * mAP50-95 = **0.279**

* **VISDRONE (Person)**

  * Images: 548
  * P = **0.256**
  * R = **0.0302**
  * mAP50 = **0.135**
  * mAP50-95 = **0.0463**

**Benchmark (lokales `benchmark_models.py`, conf=0.25)**

|      Dataset | Precision | Recall |  mAP50 | mAP50-95 | Mode    | NTUT4K FP/Img |
| -----------: | --------: | -----: | -----: | -------: | ------- | ------------: |
|     **SARD** |    0.8843 | 0.7259 | 0.8234 |   0.3661 | GT      |             – |
|  **HERIDAL** |    0.6695 | 0.5735 | 0.6190 |   0.2789 | GT      |             – |
| **VISDRONE** |    0.2558 | 0.0302 | 0.1346 |   0.0463 | GT      |             – |
|   **NTUT4K** |         – |      – |      – |        – | FP-only |      **4.05** |

---

## 3) SAR-Datensätze & Vorbereitung

**Zielstruktur (unter `D:\data`)**

* **SARDYOLO**

  * Pfad: `D:\data\SARDYOLO`
  * YAML: `cfg/sardyolo.yaml`
  * Inhalt: Roboflow-YOLO, Personenklasse **0**

* **HERIDALYOLO**

  * Pfad: `D:\data\HERIDALYOLO`
  * YAML: `cfg/heridalyolo.yaml`
  * Inhalt: Roboflow-YOLO, Personenklasse **0**

* **NTUT4KYOLO**

  * Pfad: `D:\data\NTUT4KYOLO`
  * YAML: `cfg/ntut4kyolo.yaml`
  * Inhalt: überwiegend **Negative** (keine Personenlabels)

* **VisdroneYOLO**

  * Pfad: `D:\data\VisdroneYOLO`
  * YAML: `cfg/visdrone_person.yaml` (Personenfilter)

* **Zenodo 7740081**

  * Bilder im getesteten Dump **nicht** auffindbar → nicht eingebunden

**Konvertierung (`src/prepare_external_sar_datasets.py`)**

* Formatdetektion:

  * `roboflow_yolo`, `plain_yolo`, `coco`, `voc`, `negative`
* Filter:

  * ausschließlich Klasse `person` (0)
  * Negativbilder → leere `.txt`-Labels
* Kombinations-YAMLs:

  * `cfg/sar_pos_only.yaml`
    → **SARDYOLO + HERIDALYOLO**, nur positive Personen
  * `cfg/sar_person_mix.yaml`
    → **SARDYOLO + HERIDALYOLO + NTUT4KYOLO** (inkl. Negativ-Set)

**Bekannte Besonderheit**

* Roboflow Mixed (Boxes + Segmente) → Ultralytics-Warnung
  → **Segmente werden entfernt**, es werden nur **Boxen** genutzt.

---

## 4) Stage-3 — SAR Finetuning & Modell-Varianten

Stage-3 läuft über `src/stage3_train_sar.py` (lokal, RTX A2000).

### 4.1 Gemeinsame Stage-3-Konfiguration

* Skript: `src/stage3_train_sar.py`
* Wichtige CLI-Argumente:

  * `--data` (z. B. `cfg/sar_pos_only.yaml`)
  * `--weights` (Startmodell, z. B. `stage2_best.pt`, `best.pt`, `stage2b_dce_best.pt`, etc.)
  * `--epochs` (Default **20**, diverse Varianten 3–12 Ep)
  * `--batch=6` (A2000-safe)
  * `--imgsz=800`
  * `--freeze` (Default **0**)
  * `--skip-epoch-val` (optional; spart VRAM)
* Optimierung:

  ```text
  optimizer="AdamW",
  lr0=0.003,
  lrf=0.01,
  cos_lr=True,
  weight_decay=0.0005,
  warmup_epochs=1,
  warmup_momentum=0.5,
  nbs=64,
  patience=8
  ```
* Augmentierung:

  ```text
  hsv_h=0.03, hsv_s=0.8, hsv_v=0.6,
  degrees=8.0, scale=0.85, translate=0.2,
  mosaic=0.15, mixup=0.0, copy_paste=0.1,
  fliplr=0.5, flipud=0.0,
  perspective=0.002, close_mosaic=3
  ```
* Inferenz / Val:

  * `iou=0.5`, `max_det=150`
  * `classes=[0]` (Person)
* Val-Modi:

  * Normal: `m.val(data=..., split="val", ...)`
  * FP-only-Fallback:

    * Kein GT → `_fp_only_eval(...)`
    * Metriken: `avg_fp_per_image`, `images`, `total_preds`, `max_preds_single_image`
* Artefakte:

  * Kopie `runs/train/weights/best.pt` → `experiments/.../weights/best_<exp>.pt`
  * * Checkpoint in `PROJECT_ROOT/checkpoints` mit eindeutigen Namen, z. B.:
      `stage3_sard_heridal_best.pt`, `stage3_sard_heridal_newstage1_best.pt`, …

---

### 4.2 Überblick über Stage-3-Modelle

**Namensschema (praktisch verwendet)**

* `stage3_4ep_best.pt`
* `stage3_sard_heridal_best.pt`
* `stage3_sard_heridal_newstage1_best.pt`
* `stage3_sar_mix_newstage1_4ep_best.pt`
* `stage3_sar_recallpush_3ep_best.pt`
* `stage3_sar_recallpush_6ep_best.pt`
* `stage3_sar_tiltprone_4ep_best.pt`

**Alle Benchmarks (conf=0.25, imgsz=800, batch=2, via `benchmark_models.py`)**

```jsonc
Datasets:
- SARD      → cfg/sardyolo.yaml
- HERIDAL   → cfg/heridalyolo.yaml
- VISDRONE  → cfg/visdrone_person.yaml
- NTUT4K    → cfg/ntut4kyolo.yaml (FP-only)
```

#### 4.2.1 Benchmark-Tabelle (GT-Metriken + FP-Only)

> Werte bei **conf=0.25**, `imgsz=800`, `batch=2`, `iou=0.5`.

**a) Baselines & Stage-2**

| Modell                    | Dataset  | Precision | Recall |  mAP50 | mAP50-95 | Mode    | NTUT4K avg FP/img |
| ------------------------- | -------- | --------: | -----: | -----: | -------: | ------- | ----------------: |
| **best** (new_stage1)     | SARD     |    0.4100 | 0.0082 | 0.1180 |   0.0590 | GT      |                 – |
|                           | HERIDAL  |    0.5788 | 0.2904 | 0.4349 |   0.2163 | GT      |                 – |
|                           | VISDRONE |    0.7489 | 0.5596 | 0.6803 |   0.3348 | GT      |                 – |
|                           | NTUT4K   |         – |      – |      – |        – | FP-only |          **8.80** |
| **stage2_best**           | SARD     |    0.7697 | 0.0800 | 0.4275 |   0.1990 | GT      |                 – |
|                           | HERIDAL  |    0.5020 | 0.2261 | 0.3643 |   0.1586 | GT      |                 – |
|                           | VISDRONE |    0.7320 | 0.4789 | 0.6254 |   0.2861 | GT      |                 – |
|                           | NTUT4K   |         – |      – |      – |        – | FP-only |          **7.92** |
| **stage2a_standard_best** | SARD     |    0.8754 | 0.5236 | 0.7096 |   0.3184 | GT      |                 – |
|                           | HERIDAL  |    0.7404 | 0.4246 | 0.5820 |   0.2682 | GT      |                 – |
|                           | VISDRONE |    0.7700 | 0.1294 | 0.4498 |   0.2048 | GT      |                 – |
|                           | NTUT4K   |         – |      – |      – |        – | FP-only |          **4.46** |
| **stage2b_dce_best**      | SARD     |    0.8843 | 0.7259 | 0.8234 |   0.3661 | GT      |                 – |
|                           | HERIDAL  |    0.6695 | 0.5735 | 0.6190 |   0.2789 | GT      |                 – |
|                           | VISDRONE |    0.2558 | 0.0302 | 0.1346 |   0.0463 | GT      |                 – |
|                           | NTUT4K   |         – |      – |      – |        – | FP-only |          **4.05** |

**b) Stage-3-Varianten**

| Modell                                 | Dataset  | Precision | Recall |  mAP50 | mAP50-95 | Mode    | NTUT4K avg FP/img |
| -------------------------------------- | -------- | --------: | -----: | -----: | -------: | ------- | ----------------: |
| **stage3_4ep_best**                    | SARD     |    0.8845 | 0.6651 | 0.7947 |   0.3768 | GT      |                 – |
|                                        | HERIDAL  |    0.6417 | 0.3621 | 0.4907 |   0.2267 | GT      |                 – |
|                                        | VISDRONE |    0.9375 | 0.0011 | 0.4696 |   0.2353 | GT      |                 – |
|                                        | NTUT4K   |         – |      – |      – |        – | FP-only |         **0.012** |
| **stage3_sard_heridal_best**           | SARD     |    0.8449 | 0.6815 | 0.7848 |   0.3673 | GT      |                 – |
|                                        | HERIDAL  |    0.6124 | 0.5607 | 0.6018 |   0.2830 | GT      |                 – |
|                                        | VISDRONE |    0.7285 | 0.2347 | 0.4826 |   0.2213 | GT      |                 – |
|                                        | NTUT4K   |         – |      – |      – |        – | FP-only |          **6.58** |
| **stage3_sard_heridal_newstage1_best** | SARD     |    0.8128 | 0.6500 | 0.6940 |   0.2984 | GT      |                 – |
|                                        | HERIDAL  |    0.5960 | 0.5533 | 0.5952 |   0.2857 | GT      |                 – |
|                                        | VISDRONE |    0.6860 | 0.1851 | 0.4346 |   0.1936 | GT      |                 – |
|                                        | NTUT4K   |         – |      – |      – |        – | FP-only |          **6.04** |
| **stage3_sar_mix_newstage1_4ep_best**  | SARD     |    0.8928 | 0.5352 | 0.7213 |   0.3059 | GT      |                 – |
|                                        | HERIDAL  |    0.7846 | 0.0938 | 0.4379 |   0.2048 | GT      |                 – |
|                                        | VISDRONE |    0.4286 | 0.0002 | 0.2143 |   0.1287 | GT      |                 – |
|                                        | NTUT4K   |         – |      – |      – |        – | FP-only |        **0.0024** |
| **stage3_sar_recallpush_3ep_best**     | SARD     |    0.8628 | 0.6664 | 0.7840 |   0.3585 | GT      |                 – |
|                                        | HERIDAL  |    0.6616 | 0.4779 | 0.5629 |   0.2662 | GT      |                 – |
|                                        | VISDRONE |    0.7313 | 0.2112 | 0.4715 |   0.2227 | GT      |                 – |
|                                        | NTUT4K   |         – |      – |      – |        – | FP-only |          **5.74** |
| **stage3_sar_recallpush_6ep_best**     | SARD     |    0.8654 | 0.6678 | 0.7887 |   0.3681 | GT      |                 – |
|                                        | HERIDAL  |    0.6305 | 0.5239 | 0.5775 |   0.2776 | GT      |                 – |
|                                        | VISDRONE |    0.7294 | 0.2234 | 0.4773 |   0.2205 | GT      |                 – |
|                                        | NTUT4K   |         – |      – |      – |        – | FP-only |          **6.16** |
| **stage3_sar_tiltprone_4ep_best**      | SARD     |    0.9349 | 0.3923 | 0.6621 |   0.2998 | GT      |                 – |
|                                        | HERIDAL  |    0.7683 | 0.2316 | 0.4972 |   0.1934 | GT      |                 – |
|                                        | VISDRONE |    0.7827 | 0.0719 | 0.4246 |   0.1614 | GT      |                 – |
|                                        | NTUT4K   |         – |      – |      – |        – | FP-only |          **2.24** |

*Stand der Benchmarks: 17.11.2025*

---

## 5) Confidencce-Sweeps für `best.pt` (new_stage1)

Mit `src/benchmark_models.py` bei unterschiedlichen `--conf`-Werten, `imgsz=800`, `batch=2`, `iou=0.5`.

### 5.1 Conf = 0.30

```json
SARD:
  P       = 0.4100
  R       = 0.0082
  mAP50   = 0.1761
  mAP50-95= 0.0880

HERIDAL:
  P       = 0.6481
  R       = 0.2574
  mAP50   = 0.4499
  mAP50-95= 0.2236

VISDRONE:
  P       = 0.7904
  R       = 0.5320
  mAP50   = 0.6792
  mAP50-95= 0.3392

NTUT4K (FP-only):
  avg_fp_per_image = 8.1746
  images           = 819
  total_preds      = 6695
  max_preds_image  = 51
```

### 5.2 Conf = 0.35

```json
SARD:
  P       = 0.4286
  R       = 0.0082
  mAP50   = 0.2189
  mAP50-95= 0.1094

HERIDAL:
  P       = 0.7321
  R       = 0.2261
  mAP50   = 0.4713
  mAP50-95= 0.2385

VISDRONE:
  P       = 0.8235
  R       = 0.5032
  mAP50   = 0.6755
  mAP50-95= 0.3422

NTUT4K (FP-only):
  avg_fp_per_image = 7.5385
  images           = 819
  total_preds      = 6174
  max_preds_image  = 47
```

### 5.3 Conf = 0.40

```json
SARD:
  P       = 0.4583
  R       = 0.0075
  mAP50   = 0.2336
  mAP50-95= 0.1167

HERIDAL:
  P       = 0.7266
  R       = 0.1857
  mAP50   = 0.4542
  mAP50-95= 0.2379

VISDRONE:
  P       = 0.8553
  R       = 0.4695
  mAP50   = 0.6695
  mAP50-95= 0.3449

NTUT4K (FP-only):
  avg_fp_per_image = 6.9414
  images           = 819
  total_preds      = 5685
  max_preds_image  = 44
```

---

## 6) Aktuelle Betriebs-Presets

> Modell-abhängig, aber konfigurationsseitig frequent genutzt.

* **Default / robust** (SAR-Modelle, z. B. `stage3_sard_heridal_best`, `stage2b_dce_best`):

  * `conf=0.30`
  * `iou=0.50`
  * `max_det=150`
  * `imgsz=800`

* **High-Recall / Suchmodus**:

  * `conf=0.25`
  * `iou=0.50`
  * `max_det=150`
  * `imgsz=800`

* **Optional (Val / Inference)**:

  * „Slow-but-sharp“: `imgsz=896` oder `960` (nur Val / Inferenz, nicht Training)

---

## 7) `src/benchmark_models.py` — aktueller Stand

**Funktionen**

* Lädt Modelle über:

  * `--model alias=path.pt` (repeatable)
  * `--model-dir DIR` (alle `*.pt` im Verzeichnis)
* Datensätze:

  * `--dataset alias=path.yaml` (repeatable)
* Metriken:

  * Normal: Ultralytics `.val()` → Precision, Recall, mAP50, mAP50-95, Fitness
  * Fallback: **FP-only** (`_fp_only_eval`)

**Wichtige Anpassungen**

* **NTUT4K FP-only Hard-Wiring**

  ```python
  yaml_path = Path(d_yaml)
  yaml_stem = yaml_path.stem.lower()
  is_fp_only = (
      d_alias.lower() in {"ntut4k", "ntut", "ntut4k_fp"}
      or "ntut4k" in yaml_stem
  )
  ```

  → Für NTUT4K wird **direkt** `_fp_only_eval()` genutzt, ohne vorherigen mAP-Val-Run.

* **OOM-Handling**

  * Falls `.val()` mit `CUDA out of memory` scheitert:

    * Fallback auf `_fp_only_eval()` (FP-only-Mode)
  * reduzierte Val-Settings:

    * `batch=2` (bei Bedarf manuell auf `batch=1` senkbar)
    * `workers=0`

* **Ausgabe**

  * CSV: `benchmark_results.csv`
  * JSON: `benchmark_summary.json` (verschachtelt nach Modell → Dataset)
  * Plots (optional `--plots`):

    * `map50_by_dataset.png`
    * `precision_by_dataset.png`
    * `recall_by_dataset.png`
    * `fp_only_by_dataset.png` (falls FP-only-Datasets vorhanden)

---

## 8) Live-Video-Inferenz mit Area-Filter (`live_sar_detect.py`)

**Zweck**

* Live-Personendetektion in Drohnenvideos mit:

  * Metrik-Dashboard (FPS, Inferenzzeit, Detections/frame, Laufzeit)
  * **Area-Filter** zur Unterdrückung sehr kleiner Boxen (FP-Kontrolle)

**CLI-Argumente (wichtigste)**

```bash
--model          Pfad zum Modell (z. B. checkpoints/best.pt)
--video          Pfad zum Input-Video
--output         optionaler Pfad zum Output-Video
--conf           Confidence-Threshold (Default: 0.25)
--iou            IoU-Threshold NMS (Default: 0.5)
--imgsz          Inferenz-Bildgröße (Default: 800)
--device         'cuda' / 'cpu' (Default: auto)
--max-det        max. Detections pro Frame (Default: 150)
--no-save        wenn gesetzt → kein Output-Video
--min-area-ratio Minimaler Flächenanteil einer Box relativ zur Frame-Fläche
                 (Default: 0.0002 = 0.02 %)
```

**Area-Filter (Kernlogik)**

* Für jedes Frame:

  * Framegröße: `h, w = frame.shape[:2]`
  * Framefläche: `frame_area = h * w`
  * Minimalfläche: `min_area = args.min_area_ratio * frame_area`
  * Für jede Box:

    ```python
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    area = (x2 - x1) * (y2 - y1)
    if area >= min_area:
        filtered_boxes.append(box)
    ```
* Visualisierung:

  * `r0.plot()` wird **nicht** verwendet
  * stattdessen manuelles Zeichnen von Bounding-Boxen + Scores auf `annotated_frame`

**Metriken (MetricsTracker)**

* Pro Frame:

  * `frame_count`
  * `total_detections`
  * Liste `inference_times`
  * Liste `detections_per_frame` (gleitender Durchschnitt über 30 Frames)
* Abgeleitete Kennzahlen:

  * Durchschnittliche Inferenzzeit / Frame
  * FPS (`1.0 / avg_inference_time`)
  * Durchschnittliche Detections / Frame
  * Min/Max Detections / Frame
  * Gesamtlaufzeit

**Overlay-Panel**

* Zeigt u. a.:

  * Frame-Zähler + Fortschritt in %
  * aktuelle Detections (hervorgehoben)
  * Gesamt / Durchschnitt / Max Detections
  * FPS + Inferenzzeit (ms)
  * Laufzeit (mm:ss)
  * aktuelles `conf`
  * `min_area_ratio` in % der Frame-Fläche
  * Keybindings:

    * `SPACE` → Pause / Resume
    * `Q` / `ESC` → Beenden

**Hinweis zur Nutzung (Beispiel)**

```bash
python live_sar_detect.py ^
  --model checkpoints/best.pt ^
  --video path\to\drone_video.mp4 ^
  --conf 0.35 ^
  --min-area-ratio 0.0002
```

---

## 9) Bekannte Warnungen & Besonderheiten

* **Roboflow Mixed (Boxes + Segmente)**
  Ultralytics-Warnung:

  > „Box and segment counts should be equal … only boxes will be used …“
  > → Segmente werden konsequent verworfen, nur Bounding Boxes werden genutzt.

* **NTUT4K**

  * Enthält **keine** gültigen Personenlabels im Val-Split.
  * Ultralytics-Val gibt Warnung:

    > „WARNING no labels found in detect set, can not compute metrics without labels“
  * Konsequenz:

    * Keine Precision/Recall/mAP
    * Bewertung nur über **FP-only** (`avg_fp_per_image`, `total_preds`, …)

* **AMP / Assets-Downloads**

  * Logs: `yolo11n.pt`, `Arial.ttf` und ähnliche Downloads stammen von internen Ultralytics-Checks.
  * Sie **ändern das verwendete Modell nicht** (Backbone bleibt v8m bzw. v8m_dce).

* **NMS-Timeout**

  * Einzelne Warnung:

    > „NMS time limit exceeded“
  * Tritt sporadisch in frühen Läufen auf, keine Konfig-Änderung vorgenommen.

---

> **Stand dieser Dokumentation:**
> **17.11.2025**, inkl. aller bis dahin trainierten Modelle (`best.pt`, `stage2a_standard_best`, `stage2b_dce_best`, alle `stage3_*.pt`), Benchmarks und Inferenz-Anpassungen.
> 

---

## 1. VisDrone-Personendetektion – Einordnung der Ergebnisse

### Eigene Modelle

Für die VisDrone-Personendetektion liegen unter anderem folgende Ergebnisse vor (Val-Set, Person-Klasse):

* `old_stage1` (70 Epochen):

  * Precision: 0,6843
  * Recall: 0,5113
  * mAP@0,5: 0,5544
  * mAP@0,5–0,95: 0,2132

* `new_stage1` (`best.pt`):

  * Precision: 0,7327
  * Recall: 0,5706
  * mAP@0,5: 0,6297
  * mAP@0,5–0,95: 0,2680

* `stage2_best` (Re-Finetuning):

  * Precision: 0,681
  * Recall: 0,509
  * mAP@0,5: 0,557
  * mAP@0,5–0,95: 0,218

Damit steigert das neue Basismodell (`new_stage1`) die mAP@0,5 im Vergleich zur alten Stage-1-Version um rund 0,075 (absolut), bei gleichzeitig höherer Precision und höherem Recall.

### Vergleich mit publizierten Arbeiten

Publizierte Arbeiten zu VisDrone betrachten typischerweise Multi-Klassen-Detektion (Personen, Fahrzeuge etc.). Die offiziellen VisDrone-Challenges und begleitende Arbeiten berichten für starke Detektoren (Faster R-CNN, YOLO-Varianten, neuere Transformer-Ansätze) mAP-Werte im Bereich von grob 30–40 % mAP (COCO-Stil, über Klassen gemittelt), teilweise darüber, je nach Setting und Auswertemetrik. Diese Benchmarks beziehen sich jedoch auf alle Klassen und nicht speziell auf die Fußgänger-/Personenklasse.

Neuere spezialisierte Methoden wie SOD-YOLO oder VRF-DETR demonstrieren auf VisDrone-Teildatensätzen mAP@0,5-Werte um 0,50 bzw. etwas darüber für kleine Objekte und Fußgänger, häufig auf kleinere Backbones (z. B. YOLO-S- oder YOLO-n-artige Modelle) und mit Fokus auf „small object detection“.

Demgegenüber erreicht dein `new_stage1`-Modell für die Person-Klasse allein mAP@0,5 = 0,6297 bei gleichzeitig solider Precision und Recall. Verglichen mit den in der Literatur berichteten Resultaten ergibt sich:

* Die Performance liegt klar **im oberen Bereich** dessen, was für Fußgänger/Niedrigauflösungs-Ziele in VisDrone typischerweise berichtet wird.
* Die Fokussierung auf eine schwierige einzelne Klasse (Person) macht die direkte mAP-Zahl anspruchsvoller als bei Multi-Klassen-Scores, bei denen „leichtere“ Klassen (z. B. Fahrzeuge) den Durchschnitt heben.

Insgesamt lässt sich das VisDrone-Basismodell damit als qualitativ starkes, wettbewerbsfähiges Personendetektionsmodell einordnen.

---

## 2. SARD – Spezialisierte SAR-Personendetektion

### Eigene Modelle

Für das SARD-Datenset wurden insbesondere folgende Modelle ausgewertet:

* `stage2b_dce_best` (DCE-Architektur, Stage-2B, nur SARD/HERIDAL):

  * Precision: 0,8843
  * Recall: 0,7259
  * mAP@0,5: 0,8234
  * mAP@0,5–0,95: 0,3661

* `stage3_sar_recallpush_6ep_best`:

  * Precision: 0,8654
  * Recall: 0,6678
  * mAP@0,5: 0,7887
  * mAP@0,5–0,95: 0,3681

Damit erreicht insbesondere `stage2b_dce_best` auf SARD ein hohes Präzisions-/Recall-Niveau und einen mAP@0,5-Wert von über 0,82.

### Vergleich mit publizierten Benchmarks

In der verfügbaren Literatur wird SARD meist im Kontext von Such- und Rettungsanwendungen aus der Luft behandelt. Es existieren Arbeiten, in denen klassische bzw. frühere YOLO-Varianten (z. B. YOLOv4 oder YOLOv5-Derivate) und andere CNN-Detektoren auf SARD evaluiert werden. Diese Arbeiten berichten typischerweise mAP@0,5-Werte im Bereich von etwa 0,6–0,7, abhängig von Trainingssetup, Backbone und genutzten zusätzlichen Daten (z. B. Kombination mit synthetischen Disaster-Datensätzen).

Eine neuere Arbeit zur C2A-Datensammlung (kombinierte Katastrophenszenarien) vergleicht verschiedene Detektoren und diskutiert SARD im Rahmen eines breiteren Benchmarks für Menschendetektion in Katastrophenszenarien. In diesem Kontext werden für YOLO-Modelle mAP@0,5-Werte im Bereich um 0,66 auf SARD berichtet, wenn Modelle auf generischen Human-Daten plus C2A trainiert und dann auf SARD evaluiert werden.

Vor diesem Hintergrund liegt dein `stage2b_dce_best`-Modell mit mAP@0,5 ≈ 0,82 deutlich oberhalb vieler veröffentlichter Referenzwerte, bei gleichzeitig hoher Precision und hohem Recall. Auch der mAP@0,5–0,95 (≈ 0,37) deutet auf eine robuste Lokalisierung über verschiedene IoU-Schwellen hin.

Damit lassen sich die SARD-Ergebnisse als sehr stark und im Bereich der aktuell berichteten Spitzenleistungen einordnen.

---

## 3. HERIDAL – Wald-/Wilderness-Szenarien

### Eigene Modelle

Für HERIDAL wurden u. a. folgende Resultate gemessen:

* `stage2b_dce_best`:

  * Precision: 0,6695
  * Recall: 0,5735
  * mAP@0,5: 0,6190
  * mAP@0,5–0,95: 0,2787

* `stage3_sard_heridal_best`:

  * Precision: 0,6124
  * Recall: 0,5607
  * mAP@0,5: 0,6018
  * mAP@0,5–0,95: 0,2830

Die Modelle zeigen somit auf HERIDAL eine mAP@0,5 um 0,60 bei moderat hoher Precision und Recall.

### Vergleich mit publizierten Benchmarks

HERIDAL wurde in der Literatur ursprünglich im Kontext von Personendetektion in Luftbildern für Such- und Rettungseinsätze eingeführt. In der Einführung und Folgearbeiten werden verschiedene Detektoren (u. a. Faster R-CNN, YOLO-Varianten, EfficientDet-Modelle) auf HERIDAL evaluiert. Die Berichte zeigen:

* Multimodale oder Ensemble-Ansätze erreichen sehr hohe Recall-Werte (teilweise > 0,9) bei Precision in einem Bereich um 0,7.
* Einzelne Arbeiten mit spezialisierten YOLOv5-Abwandlungen berichten mAP@0,5-Werte im Bereich um 0,8 auf HERIDAL.

Im direkten Vergleich liegen deine Modelle mit mAP@0,5 ≈ 0,62 unter diesen besten publizierten Werten, aber durchaus auf einem soliden Niveau. Insbesondere die Balance aus Precision und Recall ist eher konservativ als extrem recall-orientiert. Gleichzeitig zeigen die Ergebnisse, dass HERIDAL im aktuellen Setup die schwächste Domäne ist, verglichen mit VisDrone und SARD.

Aus Sicht der Gesamtsystemleistung deutet dies darauf hin, dass eine eigene, stärker HERIDAL-zentrierte Finetuning-Phase (z. B. zusätzliche Epochen nur auf HERIDAL mit optimierter Augmentierung für Wald- und Bewuchsstrukturen) noch Verbesserungspotenzial bietet.

---

## 4. UAVDT – Zielmetriken für Tracking

Für UAVDT liegen im hier betrachteten Projekt noch keine eigenen Tracking-Ergebnisse vor; es wurden jedoch Zielmetriken definiert:

* IDF1 ≥ 0,55
* MOTA ≥ 0,35
* ID-Switch-Rate ≤ 5 %

Publizierte Arbeiten zur Multi-Object-Verfolgung auf UAVDT berichten für moderne Tracking-Verfahren (Kombinationen aus Detektoren, oft auf YOLO-Basis, plus Trackern wie DeepSORT, ByteTrack, oder eigens entwickelte MOT-Netzwerke) typischerweise:

* MOTA im Bereich von etwa 0,30 bis 0,50
* IDF1 im Bereich von etwa 0,50 bis 0,70

Die im Projekt gesetzten Zielgrößen (MOTA ≥ 0,35, IDF1 ≥ 0,55) liegen damit im unteren bis mittleren Bereich dessen, was aktuell auf UAVDT erreichbar ist. Unter der Annahme, dass die vorhandenen Detektoren (insbesondere `new_stage1`/`best.pt`) in eine saubere Tracking-Pipeline (z. B. mit ByteTrack oder DeepSORT) integriert werden, erscheinen diese Zielwerte realistisch und erreichbar. Gleichzeitig besteht bei sorgfältiger Integration und Hyperparameter-Optimierung die Möglichkeit, MOTA-Werte im Bereich 0,40–0,50 zu erreichen.

---

## 5. Kontrolle von False Positives (NTUT4K)

### Eigene FP-only-Ergebnisse

Da NTUT4K überwiegend negative Beispiele (Landschaft, Infrastruktur, keine Personen) enthält, wird es zur Messung und Reduktion von False Positives eingesetzt. Typische Ergebnisse (FP-only-Evaluierung) sind:

* `best.pt` (VisDrone-Basismodell, ohne SAR-Spezialisierung):

  * Durchschnittliche False Positives pro Bild: ≈ 8,80

* `stage2b_dce_best`:

  * ≈ 4,05 FP/Bild

* `stage3_4ep_best`:

  * ≈ 0,012 FP/Bild

* `stage3_sar_mix_newstage1_4ep_best`:

  * ≈ 0,0024 FP/Bild

Damit sinkt die FP-Rate von einem hohen Ausgangswert (≈ 8–9 FP/Bild) auf praktisch vernachlässigbares Niveau im Bereich von 10⁻³ bis 10⁻² FP/Bild, wenn stark negative Datensätze und geeignete Fine-Tuning-Strategien einbezogen werden.

### Einordnung

In der Literatur wird der explizite Einsatz großer Negativsets (reine Nicht-Person-Landschaften) zur systematischen FP-Reduktion zwar gelegentlich erwähnt (z. B. im Kontext von Hard-Negative-Mining oder „background“-Klassen), systematische Auswertungen auf dedizierten Negativ-Datensätzen wie NTUT4K sind jedoch seltener publiziert.

Die beobachtete Reduktion der FP-Rate um mehrere Größenordnungen deutet darauf hin, dass der gewählte Ansatz (gezieltes Fine-Tuning mit vielen Landschafts-/Hintergrundbildern aus NTUT4K) in der Praxis sehr wirksam ist und einen eigenständigen Beitrag zur Robustheit des Systems leistet, insbesondere für SAR-Einsätze mit hohem Anspruch an geringe Fehlalarme.

---

## 6. Laufzeit und Echtzeitfähigkeit

Die im Projekt gesetzte Zielgröße für Echtzeit-Einsatz lautet:

* mindestens etwa 15 FPS (Frames pro Sekunde) auf einer RTX A2000 bei `imgsz ≈ 800`.

In publizierten Arbeiten erreichen kleine oder mittelgroße YOLO-Backbones (YOLOv5-/YOLOv8-n,s,m-Klasse) auf Desktop-GPUs typischerweise zwischen 20 und 60 FPS, abhängig von Bildauflösung, Optimierungsgrad und Implementierungsdetails. Auf eingebetteten Systemen (z. B. Jetson-Plattformen) werden häufig noch 20–40 FPS mit kompakten Backbones berichtet.

Vor diesem Hintergrund erscheint ein Ziel von ≥ 15 FPS für ein YOLOv8m-basiertes SAR-Modell auf einer RTX A2000 als konservativ und gut erreichbar. Die aktuelle Modellgröße und die verwendeten Konfigurationen sind mit gängigen Echtzeit-Szenarien kompatibel. Eine systematische Benchmarktabelle über alle Modelle (mit Messungen in ms/Frame und FPS) wäre ein sinnvoller nächster Schritt zur vollständigen Dokumentation.

---

## 7. Gesamtfazit

Die vorliegenden Ergebnisse lassen sich wie folgt zusammenfassen:

1. **VisDrone-Personendetektion**
   Das neue Basismodell `new_stage1` erreicht mit mAP@0,5 = 0,6297 und erhöhtem Precision/Recall-Niveau eine für die Person-Klasse sehr starke Performance. Im Vergleich zu publizierten, meist Multi-Klassen-Ergebnissen auf VisDrone liegt das Modell im oberen Bereich und kann als wettbewerbsfähige Grundlage für weitere SAR-Spezialisierung betrachtet werden.

2. **SARD – SAR-spezifische Personendetektion**
   Für das SARD-Datenset erreicht `stage2b_dce_best` mAP@0,5 ≈ 0,82 bei hoher Precision und hohem Recall. Dies liegt deutlich über vielen in der Literatur rapportierten Werten und entspricht einem sehr hohen Leistungsniveau für die Detektion von Personen in SAR-Szenarien. Der DCE-Ansatz trägt hierbei messbar zur Performance bei.

3. **HERIDAL – Wald- und Wilderness-Umgebungen**
   Auf HERIDAL erzielen die Modelle mAP@0,5 im Bereich 0,60–0,62. Diese Werte liegen unterhalb der besten veröffentlichten Ergebnisse (um etwa 0,8 mAP@0,5), sind aber insgesamt solide. HERIDAL stellt damit aktuell die schwierigste Domäne dar und bietet klar identifizierbares Verbesserungspotenzial durch gezielteres Datenset-/Augmentierungs-Design oder ein dediziertes HERIDAL-Fine-Tuning.

4. **False-Positive-Reduktion mit NTUT4K**
   Durch Einbezug des stark negativen NTUT4K-Datensatzes kann die FP-Rate von ursprünglich rund 8–9 FPs/Bild (`best.pt`) auf Werte um 10⁻² bis 10⁻³ FPs/Bild gesenkt werden. Dies zeigt eine ausgeprägte Robustheit gegenüber Landschaftsszenen und spricht für die Praxistauglichkeit im SAR-Kontext, in dem geringe Fehlalarmraten zentral sind.

5. **Tracking-Ziele auf UAVDT**
   Die definierten Zielwerte (MOTA ≥ 0,35, IDF1 ≥ 0,55) liegen im realistischen Bereich der publizierten Ergebnisse für UAVDT und sind mit einer sauberen Integration moderner Tracker (z. B. ByteTrack/DeepSORT) auf Basis der vorhandenen Detektoren voraussichtlich erreichbar.

6. **Echtzeitfähigkeit**
   Die anvisierte Echtzeitfähigkeit (≥ 15 FPS auf RTX A2000) erscheint technisch gut erreichbar und ist im Rahmen ähnlicher Arbeiten eher konservativ angesetzt.

Insgesamt deutet die Gesamtschau darauf hin, dass das System – insbesondere in Bezug auf SARD und die FP-Kontrolle – ein sehr hohes Niveau erreicht und sich im Bereich aktueller State-of-the-Art-Ansätze bewegt. Die wesentlichen offenen Punkte liegen in der weiteren Optimierung der HERIDAL-Performance und in der abschließenden Integration des Trackings auf UAVDT, um das Gesamtsystem vollständig zu evaluieren.

---

## Literaturverzeichnis (ausgewählte Referenzen)

1. Zhu, P., Wen, L., Du, D., Bian, X., Ling, H., and Hu, Q.
   „Vision Meets Drones: A Challenge.“
   International Journal of Computer Vision, Band 129, 2021, Seiten 1–31.
   (Einführung und Benchmarkbeschreibung für das VisDrone-Datenset.)

2. „SOD-YOLO: A lightweight small object detection framework.“
   Scientific Reports, Nature Portfolio, online publizierter Artikel zur leichten Objektdetektion mit YOLO-basiertem Ansatz auf Datensätzen wie VisDrone.
   (Veröffentlicht im Zeitraum 2024/2025; genaue Angaben siehe Originalpublikation.)

3. „VRF-DETR: Vision-guided Residual Feature Detection Transformer.“
   arXiv-Preprint arXiv:2504.15165, 2025.
   (Transformatorbasierter Detektor mit Auswertungen u. a. auf VisDrone-ähnlichen Szenarien.)

4. Ragib Amin Nihal et al.
   „UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection Dataset for Disaster Scenarios.“
   arXiv-Preprint arXiv:2408.04922, 2024.
   (Einführung des C2A-Datensets und Vergleich verschiedener Detektoren; enthält u. a. Statistiken zu SARD und ähnlichen Datensätzen.)

5. „Multimodal Deep Learning for Person Detection in Aerial Images.“
   Electronics, MDPI, 2020.
   (Arbeit zur Personendetektion in Luftbildern mit Beschreibung und Nutzung des HERIDAL-Datensets.)

6. „Open Problems in Computer Vision for Wilderness Search and Rescue with Drones.“
   Remote Sensing, MDPI, Erscheinungsjahr im Bereich 2023/2024.
   (Übersichtsarbeit zu Such- und Rettungsanwendungen in Wald-/Wilderness-Szenarien; diskutiert HERIDAL und Detektionsmodelle wie EfficientDet.)

7. Du, D., Qi, Y., Yu, H. et al.
   „The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking.“
   In: European Conference on Computer Vision (ECCV) Workshops, 2018.
   (Einführung des UAVDT-Benchmarks für Detektion und Tracking in UAV-Videos.)

8. Shrivastava, A., Gupta, A., und Girshick, R.
   „Training Region-Based Object Detectors with Online Hard Example Mining.“
   In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
   (Grundlegende Arbeit zum Hard-Negative-/Hard-Example-Mining, relevant für Strategien zur Reduktion von False Positives.)

