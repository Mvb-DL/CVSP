# src/benchmark_models.py
from __future__ import annotations
import argparse, json, csv, os, pathlib  # os und pathlib ergänzen!
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt

# >>> Windows-Fix für Linux-Checkpoints mit PosixPath <<<
if os.name == "nt":
    # YOLO-Checkpoints, die unter Linux gespeichert wurden, enthalten oft pathlib.PosixPath.
    # Auf Windows wirft das beim Unpickling sonst UnsupportedOperation.
    pathlib.PosixPath = pathlib.WindowsPath  # monkeypatch

from ultralytics import YOLO


# ---------- Minimal-YAML-Leser für val-Pfade ----------
def _read_yaml_minimal(yaml_path: Path):
    txt = yaml_path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    root = None
    vals: List[str] = []
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
    out_dirs: List[Path] = []
    for v in vals:
        p = Path(v)
        if not p.is_absolute() and root is not None:
            p = root / v
        out_dirs.append(p)
    if not out_dirs and root is not None:
        out_dirs = [root / "images" / "val"]
    return out_dirs

def _fp_only_eval(m: YOLO, sources: List[Path], imgsz: int, conf: float, iou: float, device_arg) -> Dict:
    total_imgs, total_preds, max_per_img = 0, 0, 0
    for src in sources:
        if not src.exists():
            continue
        gen = m.predict(
            source=str(src), imgsz=int(imgsz), conf=float(conf), iou=float(iou),
            stream=True, device=device_arg, verbose=False
        )
        for r in gen:
            n = 0 if r.boxes is None else int(len(r.boxes))
            total_imgs += 1
            total_preds += n
            max_per_img = max(max_per_img, n)
    avg_fp = total_preds / max(total_imgs, 1)
    return {
        "avg_fp_per_image": float(avg_fp),
        "images": int(total_imgs),
        "total_preds": int(total_preds),
        "max_preds_single_image": int(max_per_img),
    }

def _device_arg():
    import torch
    if torch.cuda.is_available():
        return 0
    return "cpu"

def parse_kv_list(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Use alias=path format, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out

def collect_models_from_dirs(dirs: List[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for d in dirs or []:
        for p in Path(d).glob("*.pt"):
            alias = p.stem
            base = alias
            i = 1
            # ensure unique alias
            while alias in models:
                alias = f"{base}_{i}"
                i += 1
            models[alias] = str(p.resolve())
    return models

def main():
    ap = argparse.ArgumentParser(description="Benchmark multiple YOLO models across multiple datasets")
    ap.add_argument("--model", action="append",
                    help='Repeatable: alias=PATH_TO_WEIGHTS.pt (optional if --model-dir is used)')
    ap.add_argument("--model-dir", action="append",
                    help='Repeatable: directory to load all *.pt (alias = filename)')
    ap.add_argument("--dataset", action="append", required=True,
                    help='Repeatable: alias=PATH_TO_DATA.yaml')
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("experiments") / "benchmarks")
    args = ap.parse_args()

    models = {}
    # from dirs
    if args.model_dir:
        models.update(collect_models_from_dirs(args.model_dir))
    # explicit alias=path
    if args.model:
        models.update(parse_kv_list(args.model))
    if not models:
        raise SystemExit("No models provided. Use --model-dir or --model alias=path")

    datasets = parse_kv_list(args.dataset)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir / f"bm_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    device_arg = _device_arg()
    rows = []
    summary = {}

    for m_alias, m_path in models.items():
        m = YOLO(m_path)
        summary[m_alias] = {}
        for d_alias, d_yaml in datasets.items():
            print(f"[RUN] {m_alias} on {d_alias}")

            yaml_path = Path(d_yaml)
            yaml_stem = yaml_path.stem.lower()

            # -----------------------------------------
            # 1) Hard-wire: NTUT4K ist FP-only Dataset
            # -----------------------------------------
            is_fp_only = (
                d_alias.lower() in {"ntut4k", "ntut", "ntut4k_fp"}
                or "ntut4k" in yaml_stem
            )

            if is_fp_only:
                print(f"[INFO] {d_alias}: configured as FP-only (no GT labels) -> running FP-only eval")
                sources = _read_yaml_minimal(yaml_path)
                fp = _fp_only_eval(
                    m,
                    sources,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device_arg=device_arg,
                )
                metrics = {
                    "model": m_alias,
                    "dataset": d_alias,
                    "precision": None,
                    "recall": None,
                    "mAP50": None,
                    "mAP50-95": None,
                    "fitness": None,
                    "avg_fp_per_image": fp["avg_fp_per_image"],
                    "images": fp["images"],
                    "total_preds": fp["total_preds"],
                    "max_preds_single_image": fp["max_preds_single_image"],
                    "mode": "FP-only",
                }

            else:
                # -----------------------------------------
                # 2) Normaler GT-Val-Flow mit Fallback
                # -----------------------------------------
                try:
                    val = m.val(
                        data=d_yaml,
                        split="val",
                        imgsz=args.imgsz,
                        batch=args.batch,
                        workers=0,
                        conf=args.conf,
                        iou=args.iou,
                        device=device_arg,
                        plots=False,
                    )
                    p = getattr(val.box, "p", None)
                    if p is None or (hasattr(p, "size") and p.size == 0):
                        raise ValueError("No GT labels in val set")

                    metrics = {
                        "model": m_alias,
                        "dataset": d_alias,
                        "precision": float(val.box.p[0]),
                        "recall": float(val.box.r[0]),
                        "mAP50": float(val.box.map50),
                        "mAP50-95": float(val.box.map),
                        "fitness": float(val.fitness),
                        "mode": "GT",
                    }
                except Exception as e:
                    print(f"[WARN] {d_alias}: GT-val failed ({e}) -> FP-only")
                    sources = _read_yaml_minimal(yaml_path)
                    fp = _fp_only_eval(
                        m,
                        sources,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        device_arg=device_arg,
                    )
                    metrics = {
                        "model": m_alias,
                        "dataset": d_alias,
                        "precision": None,
                        "recall": None,
                        "mAP50": None,
                        "mAP50-95": None,
                        "fitness": None,
                        "avg_fp_per_image": fp["avg_fp_per_image"],
                        "images": fp["images"],
                        "total_preds": fp["total_preds"],
                        "max_preds_single_image": fp["max_preds_single_image"],
                        "mode": "FP-only",
                    }

            rows.append(metrics)
            summary[m_alias][d_alias] = metrics

    # Save CSV
    csv_path = outdir / "benchmark_results.csv"
    fields = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[SAVE] {csv_path}")

    # Plots (only GT-metrics)
    if args.plots:
        import numpy as np

        gt_rows = [r for r in rows if r.get("mode") == "GT" and r.get("mAP50") is not None]
        if gt_rows:
            datasets_unique = sorted({r["dataset"] for r in gt_rows})
            models_unique = list(models.keys())
            x = np.arange(len(datasets_unique))
            width = 0.8 / max(1, len(models_unique))

            # mAP@0.5
            fig1 = plt.figure(figsize=(10, 5))
            for i, m_alias in enumerate(models_unique):
                vals = [next((r["mAP50"] for r in gt_rows if r["dataset"] == d and r["model"] == m_alias), 0.0)
                        for d in datasets_unique]
                plt.bar(x + i*width, vals, width=width, label=m_alias)
            plt.xticks(x + width*(len(models_unique)-1)/2, datasets_unique, rotation=0)
            plt.ylabel("mAP@0.5")
            plt.title("mAP@0.5 über Datensätze")
            plt.legend()
            plt.tight_layout()
            fig1.savefig(outdir / "map50_by_dataset.png", dpi=150)

            # Precision
            fig2 = plt.figure(figsize=(10, 5))
            for i, m_alias in enumerate(models_unique):
                vals = [next((r["precision"] for r in gt_rows if r["dataset"] == d and r["model"] == m_alias), 0.0)
                        for d in datasets_unique]
                plt.bar(x + i*width, vals, width=width, label=m_alias)
            plt.xticks(x + width*(len(models_unique)-1)/2, datasets_unique)
            plt.ylabel("Precision")
            plt.title("Precision über Datensätze")
            plt.legend(); plt.tight_layout()
            fig2.savefig(outdir / "precision_by_dataset.png", dpi=150)

            # Recall
            fig3 = plt.figure(figsize=(10, 5))
            for i, m_alias in enumerate(models_unique):
                vals = [next((r["recall"] for r in gt_rows if r["dataset"] == d and r["model"] == m_alias), 0.0)
                        for d in datasets_unique]
            plt.bar(x + i*width, vals, width=width, label=m_alias)
            plt.xticks(x + width*(len(models_unique)-1)/2, datasets_unique)
            plt.ylabel("Recall")
            plt.title("Recall über Datensätze")
            plt.legend(); plt.tight_layout()
            fig3.savefig(outdir / "recall_by_dataset.png", dpi=150)

        # FP-only Balken (falls vorhanden)
        fp_rows = [r for r in rows if r.get("mode") == "FP-only"]
        if fp_rows:
            ds_fp = sorted({r["dataset"] for r in fp_rows})
            models_unique = list(models.keys())
            x = np.arange(len(ds_fp))
            width = 0.8 / max(1, len(models_unique))
            fig4 = plt.figure(figsize=(10, 5))
            for i, m_alias in enumerate(models_unique):
                vals = [next((r["avg_fp_per_image"] for r in fp_rows if r["dataset"] == d and r["model"] == m_alias), 0.0)
                        for d in ds_fp]
                plt.bar(x + i*width, vals, width=width, label=m_alias)
            plt.xticks(x + width*(len(models_unique)-1)/2, ds_fp)
            plt.ylabel("Durchschn. FP/Bild")
            plt.title("FP-only (ohne Labels)")
            plt.legend(); plt.tight_layout()
            fig4.savefig(outdir / "fp_only_by_dataset.png", dpi=150)

    with open(outdir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] {outdir}")

if __name__ == "__main__":
    main()
