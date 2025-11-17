
from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Set



def iter_zip_names(zip_path: Path) -> List[str]:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return zf.namelist()
    except Exception as e:
        print(f"  [ERROR] Konnte ZIP {zip_path} nicht lesen: {e}")
        return []


def iter_tar_names(tar_path: Path) -> List[str]:
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            return [m.name for m in tf.getmembers()]
    except Exception as e:
        print(f"  [ERROR] Konnte TAR {tar_path} nicht lesen: {e}")
        return []


def all_archive_names(archive_path: Path) -> List[str]:
    if archive_path.suffix.lower() == ".zip":
        return iter_zip_names(archive_path)
    if archive_path.suffix.lower() in {".tar", ".tgz", ".gz", ".bz2", ".xz"} or \
       any(str(archive_path).lower().endswith(ext) for ext in (".tar.gz", ".tar.bz2", ".tar.xz")):
        return iter_tar_names(archive_path)
    return []


def collect_path_segments(names: Iterable[str]) -> Set[str]:

    segs: Set[str] = set()
    for n in names:
        n = n.strip().strip("/") 
        if not n:
            continue
        parts = [p for p in n.split("/") if p]
        for p in parts:
            segs.add(p.lower())
    return segs


def find_child_segments(names: Iterable[str], parent_segment: str) -> Set[str]:

    parent_lower = parent_segment.lower()
    childs: Set[str] = set()
    for n in names:
        n = n.strip().strip("/")
        if not n:
            continue
        parts = [p for p in n.split("/") if p]
        for i, p in enumerate(parts):
            if p.lower() == parent_lower and i + 1 < len(parts):
                childs.add(parts[i + 1].lower())
    return childs

def detect_visdron_dirs(data_root: Path) -> List[Path]:
    return [
        p for p in data_root.iterdir()
        if p.is_dir() and any(k in p.name.lower() for k in ("visdron", "vis-drone", "visdrone"))
    ]


def detect_visdron_archives(data_root: Path) -> List[Path]:
    archives: List[Path] = []
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith(".zip") or name.endswith(".tar") or name.endswith(".tar.gz") or
                name.endswith(".tgz") or name.endswith(".tar.bz2")):
            continue
        if (("vis" in name and "drone" in name) or "visdron" in name or "visdrone" in name):
            archives.append(p)
    return archives


def check_visdron_dir_structure(vis_root: Path):
    print(f"\n[Visdron] Prüfe Verzeichnisstruktur unter: {vis_root}")
    expected_splits = ["train", "val", "test dev", "test challenge"]
    expected_subdirs = ["img", "ann", "meta"]

    for split in expected_splits:
        split_dir = vis_root / split
        if split_dir.exists():
            print(f"  [OK] Split '{split}' gefunden: {split_dir}")
            for sub in expected_subdirs:
                sub_dir = split_dir / sub
                if sub_dir.exists():
                    count = sum(1 for _ in sub_dir.iterdir())
                    print(f"      [OK] Unterordner '{sub}' vorhanden ({count} Einträge)")
                else:
                    print(f"      [WARN] Unterordner '{sub}' fehlt in {split_dir}")
        else:
            print(f"  [WARN] Split '{split}' nicht gefunden (erwartet z.B. {vis_root / split})")


def check_visdron_archive_structure(archive_path: Path):
    print(f"\n[Visdron] Prüfe Archiv: {archive_path}")
    names = all_archive_names(archive_path)
    if not names:
        print("  [WARN] Keine Einträge im Archiv oder Fehler beim Lesen.")
        return

    segments = collect_path_segments(names)
    print(f"  [INFO] Gefundene Segmente (Ausschnitt): {sorted(list(segments))[:15]} ...")

    expected_splits = ["train", "val", "test dev", "test challenge"]
    expected_subdirs = ["img", "ann", "meta"]

    for split in expected_splits:
        child_subdirs = find_child_segments(names, split)
        if child_subdirs:
            print(f"  [OK] Split '{split}' im Archiv gefunden. Unterordner: {sorted(child_subdirs)}")
            for sub in expected_subdirs:
                if sub in child_subdirs:
                    print(f"      [OK] '{sub}' unter '{split}' vorhanden")
                else:
                    print(f"      [WARN] '{sub}' unter '{split}' NICHT gefunden")
        else:
            print(f"  [WARN] Split '{split}' im Archiv NICHT gefunden.")


def run_visdron_checks(data_root: Path):
    print("\n================ Visdron / VisDrone Check ================")
    dirs = detect_visdron_dirs(data_root)
    if not dirs:
        print("[INFO] Kein Visdron-Verzeichnis direkt unter data/ gefunden.")
    else:
        for d in dirs:
            check_visdron_dir_structure(d)

    archives = detect_visdron_archives(data_root)
    if not archives:
        print("[INFO] Keine Visdron-Archive (*.zip, *.tar, ...) gefunden.")
    else:
        for a in archives:
            check_visdron_archive_structure(a)


def detect_uavdt_dirs(data_root: Path) -> List[Path]:
    return [
        p for p in data_root.iterdir()
        if p.is_dir() and "uavdt" in p.name.lower()
    ]


def detect_uavdt_archives(data_root: Path) -> List[Path]:
    archives: List[Path] = []
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith(".zip") or name.endswith(".tar") or name.endswith(".tar.gz") or
                name.endswith(".tgz") or name.endswith(".tar.bz2")):
            continue
        if "uavdt" in name:
            archives.append(p)
    return archives


def check_uavdt_dir_structure(uav_root: Path):
    print(f"\n[UAVDT] Prüfe Verzeichnisstruktur unter: {uav_root}")
    expected_splits = ["train", "test"]
    expected_subdirs = ["img", "ann", "meta"]

    for split in expected_splits:
        split_dir = uav_root / split
        if split_dir.exists():
            print(f"  [OK] Split '{split}' gefunden: {split_dir}")
            for sub in expected_subdirs:
                sub_dir = split_dir / sub
                if sub_dir.exists():
                    count = sum(1 for _ in sub_dir.iterdir())
                    print(f"      [OK] Unterordner '{sub}' vorhanden ({count} Einträge)")
                else:
                    print(f"      [WARN] Unterordner '{sub}' fehlt in {split_dir}")
        else:
            print(f"  [WARN] Split '{split}' nicht gefunden (erwartet z.B. {uav_root / split})")


def check_uavdt_archive_structure(archive_path: Path):
    print(f"\n[UAVDT] Prüfe Archiv: {archive_path}")
    names = all_archive_names(archive_path)
    if not names:
        print("  [WARN] Keine Einträge im Archiv oder Fehler beim Lesen.")
        return

    segments = collect_path_segments(names)
    print(f"  [INFO] Gefundene Segmente (Ausschnitt): {sorted(list(segments))[:15]} ...")

    expected_splits = ["train", "test"]
    expected_subdirs = ["img", "ann", "meta"]

    for split in expected_splits:
        child_subdirs = find_child_segments(names, split)
        if child_subdirs:
            print(f"  [OK] Split '{split}' im Archiv gefunden. Unterordner: {sorted(child_subdirs)}")
            for sub in expected_subdirs:
                if sub in child_subdirs:
                    print(f"      [OK] '{sub}' unter '{split}' vorhanden")
                else:
                    print(f"      [WARN] '{sub}' unter '{split}' NICHT gefunden")
        else:
            print(f"  [WARN] Split '{split}' im Archiv NICHT gefunden.")


def run_uavdt_checks(data_root: Path):
    print("\n====================== UAVDT Check =======================")
    dirs = detect_uavdt_dirs(data_root)
    if not dirs:
        print("[INFO] Kein UAVDT-Verzeichnis direkt unter data/ gefunden.")
    else:
        for d in dirs:
            check_uavdt_dir_structure(d)

    archives = detect_uavdt_archives(data_root)
    if not archives:
        print("[INFO] Keine UAVDT-Archive (*.zip, *.tar, ...) gefunden.")
    else:
        for a in archives:
            check_uavdt_archive_structure(a)


def detect_okutama_dirs(data_root: Path) -> List[Path]:
    return [
        p for p in data_root.iterdir()
        if p.is_dir() and "okutama" in p.name.lower()
    ]


def check_okutama_frames_dir(frames_dir: Path, label: str):
    print(f"  [Okutama] Prüfe entpackten Ordner {label}: {frames_dir}")
    if not frames_dir.exists():
        print(f"    [WARN] Ordner existiert nicht.")
        return

    for name in ("Drone1", "Drone2", "Labels"):
        sub = frames_dir / name
        if sub.exists():
            print(f"    [OK] {name}/ gefunden: {sub}")
        else:
            print(f"    [WARN] {name}/ NICHT gefunden unter {frames_dir}")

    jpg_count = sum(1 for _ in frames_dir.rglob("*.jpg"))
    print(f"    [INFO] Anzahl .jpg-Dateien unterhalb von {frames_dir}: {jpg_count}")


def check_okutama_frames_archive(archive_path: Path, label: str):
    print(f"  [Okutama] Prüfe Archiv {label}: {archive_path}")
    names = all_archive_names(archive_path)
    if not names:
        print("    [WARN] Keine Einträge im Archiv oder Fehler beim Lesen.")
        return

    segs = collect_path_segments(names)
    print(f"    [INFO] Gefundene Segmente (Ausschnitt): {sorted(list(segs))[:15]} ...")

    for expected in ("trainsetframes", "testsetframes"):
        if expected in segs:
            print(f"    [OK] Segment '{expected}' im Archiv gefunden.")

    for name in ("drone1", "drone2", "labels", "extracted-frames-1280x720"):
        if name in segs:
            print(f"    [OK] Segment '{name}' im Archiv gefunden.")
        else:
            print(f"    [WARN] Segment '{name}' im Archiv NICHT gefunden.")


def run_okutama_checks(data_root: Path):
    print("\n===================== Okutama Check ======================")
    dirs = detect_okutama_dirs(data_root)
    if not dirs:
        print("[INFO] Kein Okutama-Verzeichnis direkt unter data/ gefunden.")
        return

    for ok_root in dirs:
        print(f"[Okutama] Basisordner: {ok_root}")

        train_dir = ok_root / "TrainSetFrames"
        test_dir = ok_root / "TestSetFrames"

        if train_dir.exists():
            check_okutama_frames_dir(train_dir, "TrainSetFrames (entpackt)")
        else:
            print("  [INFO] TrainSetFrames (entpackt) nicht gefunden.")

        if test_dir.exists():
            check_okutama_frames_dir(test_dir, "TestSetFrames (entpackt)")
        else:
            print("  [INFO] TestSetFrames (entpackt) nicht gefunden.")

        train_zip = ok_root / "TrainSetFrames.zip"
        test_zip = ok_root / "TestSetFrames.zip"

        if train_zip.exists():
            check_okutama_frames_archive(train_zip, "TrainSetFrames.zip")
        else:
            print("  [INFO] TrainSetFrames.zip nicht gefunden.")

        if test_zip.exists():
            check_okutama_frames_archive(test_zip, "TestSetFrames.zip")
        else:
            print("  [INFO] TestSetFrames.zip nicht gefunden.")



def main():
    parser = argparse.ArgumentParser(description="Prüft die Struktur der UAV-Datasets (Visdron, UAVDT, Okutama).")
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Wurzelordner der Datasets relativ zum Projektroot (Standard: 'data').",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / args.root

    print("==========================================================")
    print(f"[INFO] Projektroot: {project_root}")
    print(f"[INFO] Dataset-Root: {data_root}")
    print("==========================================================")

    if not data_root.exists():
        print("[ERROR] Dataset-Root existiert nicht. Bitte prüfen, ob der Ordnername stimmt.")
        return

    run_visdron_checks(data_root)
    run_uavdt_checks(data_root)
    run_okutama_checks(data_root)

    print("\n[INFO] check_dataset abgeschlossen.")


if __name__ == "__main__":
    main()
