---

# CHANGELOG — SAR Person Detection (YOLOv8)

> Lückenlose, **rein faktische** Verlaufs- und Änderungsdokumentation aller Trainings-, Validierungs- und Benchmark-Läufe (ohne Begründungen / Interpretation).
> Zeitraum: **November 2025**

---

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
