from pathlib import Path
import torch
import platform

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# >>> Alle DATASETS liegen jetzt auf der externen SSD:
DATA_ROOT    = Path("D:/data").resolve()   # <— Windows: forward slashes sind sicher
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"

# Bestehende Datasets (jetzt unter D:/data)
VISDRONE_ROOT = DATA_ROOT / "VisdroneYOLO"
UAVDT_ROOT    = DATA_ROOT / "UAVDT"
OKUTAMA_ROOT  = DATA_ROOT / "Okutama"

# Externals + gemischtes SAR-Ziel (werden ebenfalls auf D:/data erzeugt)
EXTERNAL_ROOT = DATA_ROOT / "external"
SARMIX_ROOT   = DATA_ROOT / "SARmix"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        gpu_index = 0  # lokal: RTX A2000 = cuda:0
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
IS_WINDOWS = platform.system().lower().startswith("win")

# Sanity logs (kein mkdir für DATA_ROOT, weil externe SSD)
for p in [VISDRONE_ROOT, EXTERNAL_ROOT]:
    if not p.exists():
        print(f"[config][WARN] Dataset path not found: {p}")
