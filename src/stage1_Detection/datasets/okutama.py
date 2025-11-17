# src/datasets/okutama.py
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, List


@dataclass
class OkutamaFrame:
    image_path: Path
    split: str       # 'train' oder 'test'
    drone: str       # 'Drone1' oder 'Drone2'
    time_of_day: str # 'Morning', 'Noon', ...
    sequence: str    # z.B. '1.1.8'


def iter_okutama_frames(root: Path, split: str = "train") -> Iterator[OkutamaFrame]:
    """
    root: Pfad zu 'data/Okutama'
    Struktur:
      root/TrainSetFrames/
        Drone1/Morning/Extracted-Frames-1280x720/1.1.1/*.jpg
        Drone2/Noon/Extracted-Frames-1280x720/2.2.2/*.jpg
        Labels/...
      root/TestSetFrames/
        gleiche Logik
    """
    split_map = {
        "train": "TrainSetFrames",
        "test": "TestSetFrames",
        "TrainSetFrames": "TrainSetFrames",
        "TestSetFrames": "TestSetFrames",
    }
    split_folder = split_map.get(split, split)
    base = root / split_folder
    if not base.exists():
        raise FileNotFoundError(f"Okutama-Basisordner nicht gefunden: {base}")

    # Ebene 1: Drone1, Drone2, Labels
    for drone_dir in sorted(base.iterdir()):
        if not drone_dir.is_dir():
            continue
        if drone_dir.name.lower() == "labels":
            # Label-Ordner ignorieren
            continue

        drone_name = drone_dir.name  # 'Drone1' oder 'Drone2'

        # Ebene 2: Morning, Noon, ...
        for tod_dir in sorted(drone_dir.iterdir()):
            if not tod_dir.is_dir():
                continue
            time_of_day = tod_dir.name  # 'Morning' / 'Noon'

            # Ebene 3: Extracted-Frames-...
            for frames_dir in sorted(tod_dir.glob("Extracted-Frames-*")):
                if not frames_dir.is_dir():
                    continue

                # Ebene 4: Sequenzen wie 1.1.8, 2.2.10, ...
                for seq_dir in sorted(frames_dir.iterdir()):
                    if not seq_dir.is_dir():
                        continue
                    sequence_name = seq_dir.name

                    # Ebene 5: eigentliche Bilder
                    for img_path in sorted(seq_dir.glob("*.jpg")):
                        yield OkutamaFrame(
                            image_path=img_path,
                            split="train" if split_folder == "TrainSetFrames" else "test",
                            drone=drone_name,
                            time_of_day=time_of_day,
                            sequence=sequence_name,
                        )


def get_okutama_sample_images(root: Path, split: str = "train", max_images: int = 50) -> List[Path]:
    images: List[Path] = []
    for frame in iter_okutama_frames(root, split):
        images.append(frame.image_path)
        if len(images) >= max_images:
            break
    return images
