# src/baseline_tests.py
import time
import argparse
from pathlib import Path
from typing import List

import torch

from src.config import VISDRON_ROOT, OKUTAMA_ROOT, UAVDT_ROOT
from src.models import load_yolo_model
from src.datasets.visdron import get_visdron_sample_images
from src.datasets.uavdt import get_uavdt_sample_images
from src.datasets.okutama import get_okutama_sample_images


def _time_inference_on_images(model, image_paths: List[Path], warmup: int = 2):
    if not image_paths:
        print("[baseline] Keine Bilder erhalten.")
        return

    print(f"[baseline] Warmup auf {min(warmup, len(image_paths))} Bildern ...")
    for img in image_paths[:warmup]:
        _ = model(str(img))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("[baseline] Starte Messung ...")
    t0 = time.time()
    total_dets = 0

    for img in image_paths:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        results = model(str(img))  
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()

        dt_ms = (t_end - t_start) * 1000.0
        print(f"  {img.name}: {dt_ms:.1f} ms")

        for r in results:
            total_dets += 0 if r.boxes is None else len(r.boxes)

    total_time = time.time() - t0
    n = len(image_paths)
    print("\n=== Baseline Summary ===")
    print(f"Anzahl Bilder:    {n}")
    print(f"Gesamtzeit:       {total_time:.2f} s")
    print(f"Ø Zeit/Bild:      {total_time / n * 1000.0:.1f} ms")
    print(f"Ø FPS:            {n / total_time:.1f}")
    print(f"Detektionen ges.: {total_dets}")


def main():
    parser = argparse.ArgumentParser(description="Phase-1 Basistests auf UAV-Datensätzen")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["visdron", "uavdt", "okutama"],
        required=True,
        help="Welcher Datensatz soll getestet werden?",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Anzahl Bilder für den Test",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics-Gewichte (z.B. yolo11n.pt, yolov8n.pt, Pfad zu .pt-Datei)",
    )
    args = parser.parse_args()

    if args.dataset == "visdron":
        root = VISDRON_ROOT
        print(f"[baseline] Nutze Visdrone aus {root} (Split: val)")
        image_paths = get_visdron_sample_images(root, split="val", max_images=args.num_images)

    elif args.dataset == "uavdt":
        root = UAVDT_ROOT
        print(f"[baseline] Nutze UAVDT aus {root} (Split: test)")
        image_paths = get_uavdt_sample_images(root, split="test", max_images=args.num_images)

    else:  
        root = OKUTAMA_ROOT
        print(f"[baseline] Nutze Okutama aus {root} (Split: train)")
        image_paths = get_okutama_sample_images(root, split="train", max_images=args.num_images)

    if not image_paths:
        raise RuntimeError(
            "Es wurden keine Bilder gefunden. "
            "Bitte prüfen, ob die Datensätze korrekt entpackt sind "
            "und die Ordnernamen (Visdrone/Okutama/UAVDT) passen."
        )

    print(f"[baseline] Anzahl ausgewählter Bilder: {len(image_paths)}")
    model = load_yolo_model(args.weights)
    _time_inference_on_images(model, image_paths)


if __name__ == "__main__":
    main()
