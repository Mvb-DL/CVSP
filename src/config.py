from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

VISDRONE_ROOT = DATA_ROOT / "Visdrone"
UAVDT_ROOT = DATA_ROOT / "UAVDT"
OKUTAMA_ROOT = DATA_ROOT / "Okutama"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        gpu_index = n - 1 
        dev = torch.device(f"cuda:{gpu_index}")
        try:
            name = torch.cuda.get_device_name(gpu_index)
        except Exception:
            name = "Unknown CUDA device"
        print(f"[config] Using device: {dev} ({name}, {n} CUDA device(s) available)")
        return dev
    else:
        dev = torch.device("cpu")
        print("[config] No CUDA devices available, using CPU.")
        return dev


DEVICE = _select_device()
