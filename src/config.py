from pathlib import Path
import torch

# Projekt-Root ist der Ordner eine Ebene über src/
# Annahme: config.py liegt in einem Ordner (z.B. src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Basis-Pfade ---
DATA_ROOT = PROJECT_ROOT / "data" 
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Output-Ordner sicherstellen
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- 1. Original-Datensätze (werden oft nicht mehr direkt verwendet, aber als Root behalten) ---
VISDRONE_ROOT = DATA_ROOT / "Visdrone"
UAVDT_ROOT = DATA_ROOT / "UAVDT"
OKUTAMA_ROOT = DATA_ROOT / "Okutama"

# --- 2. HINZUGEFÜGTE SAR-Datensätze (MÜSSEN zur Pfadstruktur passen!) ---

# WICHTIG: Die Pfade MÜSSEN zu deiner tatsächlichen Ordnerstruktur passen!
# Basierend auf deiner YAML:
SAR_ROOT = Path("/notebooks/CVSP/data") # Neuer Root-Pfad für SAR-Daten

# Konvention für YOLO-Struktur: <DATASETNAME>YOLO
VISDRONE_YOLO_ROOT = Path("D:/data/VisdroneYOLO") 
SARD_YOLO_ROOT = SAR_ROOT / "SARDYOLO" 
HERIDAL_YOLO_ROOT = SAR_ROOT / "HERIDALYOLO"
NTUT4K_YOLO_ROOT = Path("D:/data/NTUT4KYOLO_reduced") # Pfad zum reduzierten Set


# --- Gerätewahl (Unverändert und Korrekt) ---

def _select_device() -> torch.device:
    """Wählt das Rechen-Device (CUDA oder CPU)."""
    if torch.cuda.is_available():
        gpu_index = 0
        dev = torch.device(f"cuda:{gpu_index}")
        try:
            name = torch.cuda.get_device_name(gpu_index)
            n = torch.cuda.device_count()
        except Exception:
            name = "Unknown CUDA device"
            n = "?"
        print(f"[config] Using device: {dev} ({name}, {n} CUDA device(s) available)")
        return dev
    else:
        dev = torch.device("cpu")
        print("[config] No CUDA devices available, using CPU.")
        return dev


DEVICE = _select_device()