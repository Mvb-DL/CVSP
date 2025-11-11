# src/run_benchmarks.py
import argparse
import time
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

from src.config import VISDRONE_ROOT, OKUTAMA_ROOT, UAVDT_ROOT, OUTPUT_ROOT, DEVICE
from src.models import load_yolo_model, load_faster_rcnn, load_detr


BENCHMARK_DIR = OUTPUT_ROOT / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

PERSON_CLASS_COCO = 1 

def discover_visdrone_split() -> Tuple[Optional[str], Optional[Path]]:
    root = VISDRONE_ROOT
    if not root.exists():
        print(f"[visdrone][WARN] VISDRONE_ROOT does not exist: {root}")
        return None, None

    candidates: List[Tuple[str, Path]] = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        img_dir = sub / "img"
        if img_dir.exists():
            candidates.append((sub.name, img_dir))

    if not candidates:
        print(f"[visdrone][WARN] No subfolder with 'img' found under: {root}")
        return None, None

    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if "test" in n:
            s += 10
        if "dev" in n:
            s += 5
        if "val" in n:
            s += 3
        if "train" in n:
            s += 1
        return s

    best_name, best_img_dir = max(candidates, key=lambda x: score(x[0]))
    print(f"[visdrone] Using split '{best_name}' at: {best_img_dir}")
    return best_name, best_img_dir


VISDRONE_SPLIT_NAME, VISDRONE_IMG_DIR = discover_visdrone_split()


def get_all_image_paths_for_dataset(dataset_key: str) -> List[Path]:

    if dataset_key == "visdrone":
        if VISDRONE_IMG_DIR is None:
            print("[visdrone][WARN] No VisDrone img directory discovered.")
            return []
        return sorted(VISDRONE_IMG_DIR.rglob("*.jpg"))

    elif dataset_key == "uavdt":
        img_dir = UAVDT_ROOT / "test" / "img"
        return sorted(img_dir.rglob("*.jpg")) if img_dir.exists() else []

    elif dataset_key == "okutama":
        base_dir = OKUTAMA_ROOT / "TestSetFrames"
        return sorted(base_dir.rglob("*.jpg")) if base_dir.exists() else []

    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")


def get_image_paths_for_dataset(dataset_key: str, num_images: int) -> List[Path]:

    all_paths = get_all_image_paths_for_dataset(dataset_key)
    if num_images is not None and num_images > 0 and len(all_paths) > num_images:
        return all_paths[:num_images]
    return all_paths


def benchmark_yolo(model, image_paths: List[Path], score_thresh: float) -> Dict:

    if not image_paths:
        return {}


    warmup_n = min(2, len(image_paths))
    print(f"[YOLO] Warmup on {warmup_n} images ...")
    for p in image_paths[:warmup_n]:
        _ = model(str(p))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("[YOLO] Start timing ...")
    t0 = time.time()
    total_dets = 0

    for img_path in image_paths:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        results = model(str(img_path))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()

        dt_ms = (t_end - t_start) * 1000.0
        print(f"  [YOLO] {img_path.name}: {dt_ms:.1f} ms")

        for r in results:
            if r.boxes is None:
                continue
            cls = r.boxes.cls
            conf = r.boxes.conf
            if cls is None:
                continue

            mask = (cls == 0) & (conf >= score_thresh)
            total_dets += int(mask.sum().item())

    total_time = time.time() - t0
    n = len(image_paths)
    return {
        "num_images": n,
        "total_time_sec": total_time,
        "avg_time_ms": total_time / n * 1000.0,
        "fps": n / total_time,
        "total_dets": total_dets,
        "avg_dets_per_image": total_dets / n if n > 0 else 0.0,
    }


def benchmark_faster_rcnn(model, image_paths: List[Path], score_thresh: float) -> Dict:

    if not image_paths:
        return {}

    transform = T.Compose([T.ToTensor()])


    warmup_n = min(2, len(image_paths))
    print(f"[FRCNN] Warmup on {warmup_n} images ...")
    for img_path in image_paths[:warmup_n]:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = transform(img_rgb).to(DEVICE)
        with torch.no_grad():
            _ = model([inp])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("[FRCNN] Start timing ...")
    t0 = time.time()
    total_dets = 0

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = transform(img_rgb).to(DEVICE)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        with torch.no_grad():
            outputs = model([inp])[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()

        dt_ms = (t_end - t_start) * 1000.0
        print(f"  [FRCNN] {img_path.name}: {dt_ms:.1f} ms")

        labels = outputs["labels"]
        scores = outputs["scores"]
        mask = (labels == PERSON_CLASS_COCO) & (scores >= score_thresh)
        total_dets += int(mask.sum().item())

    total_time = time.time() - t0
    n = len(image_paths)
    return {
        "num_images": n,
        "total_time_sec": total_time,
        "avg_time_ms": total_time / n * 1000.0,
        "fps": n / total_time,
        "total_dets": total_dets,
        "avg_dets_per_image": total_dets / n if n > 0 else 0.0,
    }


def benchmark_detr(
    pipe,
    image_paths: List[Path],
    score_thresh: float,
    batch_size: int = 4,
) -> Dict:

    if not image_paths:
        return {}

    print("[DETR] Using Hugging Face pipeline for object detection.")

    warmup_n = min(2, len(image_paths))
    print(f"[DETR] Warmup on {warmup_n} images ...")
    for img_path in image_paths[:warmup_n]:
        _ = pipe(str(img_path))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f"[DETR] Start timing with batch_size={batch_size} ...")
    t0 = time.time()
    total_dets = 0
    n = len(image_paths)

    for i in range(0, n, batch_size):
        batch_paths = image_paths[i: i + batch_size]
        batch_strs = [str(p) for p in batch_paths]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        outputs_batch = pipe(batch_strs) 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()

        dt_ms = (t_end - t_start) * 1000.0
        avg_ms_per_image = dt_ms / len(batch_paths)
        for p in batch_paths:
            print(f"  [DETR] {p.name}: {avg_ms_per_image:.1f} ms (approx, batched)")

        for img_out in outputs_batch:
            for det in img_out:
                score = float(det.get("score", 0.0))
                label = str(det.get("label", "")).lower()

                if score < score_thresh:
                    continue

                if "person" not in label and label not in {"label_1", "1"}:
                    continue

                total_dets += 1

    total_time = time.time() - t0
    return {
        "num_images": n,
        "total_time_sec": total_time,
        "avg_time_ms": total_time / n * 1000.0,
        "fps": n / total_time,
        "total_dets": total_dets,
        "avg_dets_per_image": total_dets / n if n > 0 else 0.0,
    }


MODEL_SPECS = [
    {"name": "yolo11n", "label": "YOLOv11n",         "type": "yolo",        "weights": "yolo11n.pt"},
    {"name": "yolov8n", "label": "YOLOv8n",          "type": "yolo",        "weights": "yolov8n.pt"},
    {"name": "fasterrcnn_resnet50_fpn", "label": "Faster R-CNN",  "type": "fasterrcnn", "weights": None},
    {"name": "detr_resnet50",           "label": "DETR-ResNet50", "type": "detr",       "weights": None},
]

_vis_label_split = VISDRONE_SPLIT_NAME if VISDRONE_SPLIT_NAME is not None else "unknown"

DATASET_SPECS = [
    {"key": "visdrone", "label": f"Visdrone ({_vis_label_split})", "split": _vis_label_split},
    {"key": "uavdt",    "label": "UAVDT (test)",                   "split": "test"},
    {"key": "okutama",  "label": "Okutama (TestSetFrames)",        "split": "test"},
]


def save_results_csv(rows: List[Dict], csv_path: Path):
    if not rows:
        print("[bench] No results to save.")
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[bench] Results saved to: {csv_path}")


def plot_grouped_bar(
    rows: List[Dict],
    value_key: str,
    ylabel: str,
    title: str,
    filename: str,
):
    import numpy as np

    datasets = [spec["key"] for spec in DATASET_SPECS]
    dataset_labels = [spec["label"] for spec in DATASET_SPECS]
    models = [spec["name"] for spec in MODEL_SPECS]
    model_labels = [spec["label"] for spec in MODEL_SPECS]

    values = [[0.0 for _ in datasets] for _ in models]

    for i_m, m in enumerate(models):
        for i_d, d in enumerate(datasets):
            for r in rows:
                if r["model_name"] == m and r["dataset_key"] == d:
                    values[i_m][i_d] = r[value_key]
                    break

    x = np.arange(len(datasets))
    width = 0.18

    plt.figure(figsize=(10, 6))

    for i_m, m_label in enumerate(model_labels):
        offsets = (i_m - (len(models) - 1) / 2) * width
        bar_positions = x + offsets
        bars = plt.bar(bar_positions, values[i_m], width=width, label=m_label)

        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xticks(x, dataset_labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = BENCHMARK_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[bench] Plot saved: {out_path}")

def compute_dataset_image_counts() -> List[Dict]:
    """
    Counts how many images are available in the evaluated splits:
      - Visdrone: discovered split's img/*.jpg
      - UAVDT:   test/img/*.jpg
      - Okutama: TestSetFrames/**/*.jpg
    """
    summary: List[Dict] = []

    if VISDRONE_IMG_DIR is not None:
        vis_count = sum(1 for _ in VISDRONE_IMG_DIR.rglob("*.jpg"))
        vis_label = f"Visdrone ({VISDRONE_SPLIT_NAME})"
        vis_split = VISDRONE_SPLIT_NAME
    else:
        vis_count = 0
        vis_label = "Visdrone (missing)"
        vis_split = "missing"

    summary.append({
        "dataset_key": "visdrone",
        "dataset_label": vis_label,
        "split": vis_split,
        "num_images": vis_count,
    })

    uavdt_img_dir = UAVDT_ROOT / "test" / "img"
    uavdt_count = sum(1 for _ in uavdt_img_dir.rglob("*.jpg")) if uavdt_img_dir.exists() else 0
    summary.append({
        "dataset_key": "uavdt",
        "dataset_label": "UAVDT (test)",
        "split": "test",
        "num_images": uavdt_count,
    })

    oku_test_dir = OKUTAMA_ROOT / "TestSetFrames"
    oku_count = sum(1 for _ in oku_test_dir.rglob("*.jpg")) if oku_test_dir.exists() else 0
    summary.append({
        "dataset_key": "okutama",
        "dataset_label": "Okutama (TestSetFrames)",
        "split": "test",
        "num_images": oku_count,
    })

    return summary


def plot_dataset_overview(dataset_rows: List[Dict]):
    """
    Simple bar chart: datasets vs. number of images in the evaluated split.
    """
    import numpy as np

    if not dataset_rows:
        print("[bench] No dataset overview to plot.")
        return

    labels = [r["dataset_label"] for r in dataset_rows]
    counts = [r["num_images"] for r in dataset_rows]

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, counts)

    for bar, val in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Number of images")
    plt.title(
        "Benchmark datasets (images in evaluated splits)\n"
        "Detectors pre-trained on COCO 2017 (118k images, 80 classes)"
    )
    plt.tight_layout()

    out_path = BENCHMARK_DIR / "dataset_overview.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[bench] Dataset overview plot saved: {out_path}")


def print_dataset_overview(dataset_rows: List[Dict]):
    if not dataset_rows:
        return
    print("\nDataset overview (images in the evaluated splits):")
    for r in dataset_rows:
        print(f"  - {r['dataset_label']}: {r['num_images']} images")
    print("  All detectors are pre-trained on COCO 2017 (118k train / 80 classes).\n")


def summarize_results(rows: List[Dict]):
    """
    Print a short textual summary:
    - For each model: FPS, avg time/image and avg detections/image for each dataset.
    - For each dataset: fastest model (highest FPS).
    """
    if not rows:
        print("\n[bench] No results for summary.")
        return

    print("\n========== Benchmark summary (person detection baseline) ==========")

    rows_by_model = defaultdict(list)
    for r in rows:
        rows_by_model[r["model_name"]].append(r)

    for model_name, mrows in rows_by_model.items():
        label = mrows[0]["model_label"]
        print(f"\nModel: {label} ({model_name})")
        for r in mrows:
            print(
                f"  - {r['dataset_label']}: "
                f"FPS={r['fps']:.1f}, "
                f"avg time/img={r['avg_time_ms']:.1f} ms, "
                f"avg detections/img={r['avg_dets_per_image']:.1f}"
            )

    print("\nFastest model per dataset (by FPS):")
    rows_by_dataset = defaultdict(list)
    for r in rows:
        rows_by_dataset[r["dataset_key"]].append(r)

    for ds_key, drows in rows_by_dataset.items():
        best = max(drows, key=lambda x: x["fps"])
        print(
            f"  - {best['dataset_label']}: "
            f"{best['model_label']} with {best['fps']:.1f} FPS"
        )
    print("===================================================================\n")


def run_benchmarks(num_images: int, score_thresh: float):

    dataset_rows = compute_dataset_image_counts()
    plot_dataset_overview(dataset_rows)
    print_dataset_overview(dataset_rows)

    all_rows: List[Dict] = []

    for model_spec in MODEL_SPECS:
        print("\n======================================================")
        print(f"[bench] Starting benchmark for model: {model_spec['label']}")
        print("======================================================")


        try:
            if model_spec["type"] == "yolo":
                model = load_yolo_model(model_spec["weights"])
            elif model_spec["type"] == "fasterrcnn":
                model = load_faster_rcnn()
            elif model_spec["type"] == "detr":
                model = load_detr()
            else:
                raise ValueError(f"Unknown model type: {model_spec['type']}")
        except Exception as e:
            print(f"[bench][WARN] Model '{model_spec['label']}' could not be loaded and will be skipped:")
            print(f"               {e}")
            continue

        for ds in DATASET_SPECS:
            print(f"\n[bench] Dataset: {ds['label']}")
            image_paths = get_image_paths_for_dataset(ds["key"], num_images=num_images)
            if not image_paths:
                print("[bench][WARN] No images found, skipping this dataset.")
                continue

            if model_spec["type"] == "yolo":
                metrics = benchmark_yolo(model, image_paths, score_thresh=score_thresh)
            elif model_spec["type"] == "fasterrcnn":
                metrics = benchmark_faster_rcnn(model, image_paths, score_thresh=score_thresh)
            else:  # detr
                metrics = benchmark_detr(
                    model,
                    image_paths,
                    score_thresh=score_thresh,
                    batch_size=4,
                )

            if not metrics:
                print("[bench][WARN] No metrics returned, skipping.")
                continue

            row = {
                "phase": "baseline",
                "device": str(DEVICE),
                "model_name": model_spec["name"],
                "model_label": model_spec["label"],
                "model_type": model_spec["type"],
                "weights": model_spec["weights"] or "",
                "dataset_key": ds["key"],
                "dataset_label": ds["label"],
                "dataset_split": ds["split"],
                "num_images": metrics["num_images"],
                "total_time_sec": metrics["total_time_sec"],
                "avg_time_ms": metrics["avg_time_ms"],
                "fps": metrics["fps"],
                "total_dets": metrics["total_dets"],
                "avg_dets_per_image": metrics["avg_dets_per_image"],
                "person_only": 1, 
                "score_thresh": score_thresh,
            }
            all_rows.append(row)

    csv_path = BENCHMARK_DIR / "results.csv"
    save_results_csv(all_rows, csv_path)

    if all_rows:
        score_thresh = all_rows[0]["score_thresh"]

        title_suffix = f"(person-only, score ≥ {score_thresh:.2f})"

        plot_grouped_bar(
            all_rows,
            value_key="fps",
            ylabel="Frames per second (FPS)",
            title=f"FPS per model and dataset {title_suffix}",
            filename="fps_by_model_and_dataset.png",
        )

        plot_grouped_bar(
            all_rows,
            value_key="avg_dets_per_image",
            ylabel="Average detections per image",
            title=f"Average detected persons per image {title_suffix}",
            filename="avg_dets_by_model_and_dataset.png",
        )

        summarize_results(all_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: YOLOv11, YOLOv8, Faster R-CNN, DETR on 3 UAV datasets (PERSON detection only)."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=-1,
        help="Number of images per dataset for baseline tests. "
             "If <= 0, all available images are used.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Score threshold for detections (0-1).",
    )
    args = parser.parse_args()

    run_benchmarks(num_images=args.num_images, score_thresh=args.score_thresh)


if __name__ == "__main__":
    main()
