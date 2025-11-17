
from __future__ import annotations

from pathlib import Path
import tarfile
import zipfile

from src.config import DATA_ROOT, VISDRONE_ROOT, UAVDT_ROOT, OKUTAMA_ROOT



def extract_tar(tar_path: Path, target_root: Path) -> bool:
    """
    Entpackt eine TAR-Datei nach target_root.
    Gibt True zurück, wenn kein Fehler aufgetreten ist (Entpacken versucht).
    """
    print(f"[prepare] Entpacke TAR: {tar_path} -> {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(target_root)
        print(f"[prepare] Fertig: {tar_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Konnte {tar_path} nicht entpacken: {e}")
        return False


def extract_zip(zip_path: Path, target_root: Path) -> bool:
    """
    Entpackt eine ZIP-Datei nach target_root.
    Gibt True zurück, wenn kein Fehler aufgetreten ist (Entpacken versucht).
    """
    print(f"[prepare] Entpacke ZIP: {zip_path} -> {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_root)
        print(f"[prepare] Fertig: {zip_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Konnte {zip_path} nicht entpacken: {e}")
        return False


def _find_tar_in_root(root: Path, keyword: str) -> Path | None:
    """
    Sucht im data-Root nach einer TAR-Datei, deren Name das keyword enthält.
    Unterstützt .tar, .tar.gz, .tgz, .tar.bz2.
    """
    keyword = keyword.lower()
    candidates: list[Path] = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if keyword in name and (
            name.endswith(".tar")
            or name.endswith(".tar.gz")
            or name.endswith(".tgz")
            or name.endswith(".tar.bz2")
        ):
            candidates.append(p)

    if candidates:
        return candidates[0]
    return None


def prepare_visdrone():
    print("\n===== Visdrone vorbereiten =====")


    if (VISDRONE_ROOT / "train" / "img").exists():
        print(f"[prepare][Visdrone] train/img existiert bereits unter {VISDRONE_ROOT} – scheint entpackt zu sein.")
        return

    tar_path = _find_tar_in_root(DATA_ROOT, "visdrone")
    if tar_path is None:

        direct = DATA_ROOT / "Visdrone.tar"
        if direct.exists():
            tar_path = direct

    if tar_path is None:
        print("[prepare][Visdrone] Keine passende TAR-Datei gefunden (z.B. Visdrone.tar).")
        return

    print(f"[prepare][Visdrone] Nutze Archiv: {tar_path}")
    ok = extract_tar(tar_path, VISDRONE_ROOT)

    if ok and (VISDRONE_ROOT / "train" / "img").exists():
        try:
            tar_path.unlink()
            print(f"[prepare][Visdrone] Archiv gelöscht: {tar_path}")
        except OSError as e:
            print(f"[prepare][Visdrone][WARN] Konnte Archiv nicht löschen: {e}")
        print("[prepare][Visdrone] Entpacken erfolgreich. Struktur train/img gefunden.")
    else:
        print("[prepare][Visdrone][WARN] Nach dem Entpacken kein train/img gefunden – Archiv wird NICHT gelöscht.")



def prepare_uavdt():
    print("\n===== UAVDT vorbereiten =====")

    if (UAVDT_ROOT / "train" / "img").exists() and (UAVDT_ROOT / "test" / "img").exists():
        print(f"[prepare][UAVDT] train/img und test/img existieren bereits unter {UAVDT_ROOT} – scheint entpackt zu sein.")
        return

    tar_path = _find_tar_in_root(DATA_ROOT, "uavdt")
    if tar_path is None:
        direct = DATA_ROOT / "UAVDT.tar"
        if direct.exists():
            tar_path = direct

    if tar_path is None:
        print("[prepare][UAVDT] Keine passende TAR-Datei gefunden (z.B. UAVDT.tar).")
        return

    print(f"[prepare][UAVDT] Nutze Archiv: {tar_path}")
    ok = extract_tar(tar_path, UAVDT_ROOT)

    if ok and (UAVDT_ROOT / "train" / "img").exists() and (UAVDT_ROOT / "test" / "img").exists():
        try:
            tar_path.unlink()
            print(f"[prepare][UAVDT] Archiv gelöscht: {tar_path}")
        except OSError as e:
            print(f"[prepare][UAVDT][WARN] Konnte Archiv nicht löschen: {e}")
        print("[prepare][UAVDT] Entpacken erfolgreich. train/img und test/img gefunden.")
    else:
        print("[prepare][UAVDT][WARN] Nach dem Entpacken train/img oder test/img nicht gefunden – Archiv wird NICHT gelöscht.")



def prepare_okutama():
    print("\n===== Okutama vorbereiten =====")

    if not OKUTAMA_ROOT.exists():
        print(f"[prepare][Okutama] Basisordner {OKUTAMA_ROOT} existiert nicht.")
        return

    train_zip = OKUTAMA_ROOT / "TrainSetFrames.zip"
    train_target = OKUTAMA_ROOT / "TrainSetFrames"

    train_already = train_target.exists() and any(train_target.rglob("*.jpg"))

    if train_already:
        print(f"[prepare][Okutama] TrainSetFrames scheint bereits entpackt zu sein: {train_target}")
    elif train_zip.exists():
        print(f"[prepare][Okutama] Entpacke TrainSetFrames.zip nach {train_target}")
        ok = extract_zip(train_zip, train_target)
        # nur löschen, wenn danach wirklich Bilder da sind
        if ok and any(train_target.rglob("*.jpg")):
            try:
                train_zip.unlink()
                print(f"[prepare][Okutama] Archiv gelöscht: {train_zip}")
            except OSError as e:
                print(f"[prepare][Okutama][WARN] Konnte TrainSetFrames.zip nicht löschen: {e}")
        else:
            print("[prepare][Okutama][WARN] Nach dem Entpacken von TrainSetFrames.zip keine JPGs gefunden – Archiv wird NICHT gelöscht.")
    else:
        print(f"[prepare][Okutama][WARN] TrainSetFrames.zip nicht gefunden unter {OKUTAMA_ROOT}")


    test_zip = OKUTAMA_ROOT / "TestSetFrames.zip"
    test_target = OKUTAMA_ROOT / "TestSetFrames"

    test_already = test_target.exists() and any(test_target.rglob("*.jpg"))

    if test_already:
        print(f"[prepare][Okutama] TestSetFrames scheint bereits entpackt zu sein: {test_target}")
    elif test_zip.exists():
        print(f"[prepare][Okutama] Entpacke TestSetFrames.zip nach {test_target}")
        ok = extract_zip(test_zip, test_target)
        if ok and any(test_target.rglob("*.jpg")):
            try:
                test_zip.unlink()
                print(f"[prepare][Okutama] Archiv gelöscht: {test_zip}")
            except OSError as e:
                print(f"[prepare][Okutama][WARN] Konnte TestSetFrames.zip nicht löschen: {e}")
        else:
            print("[prepare][Okutama][WARN] Nach dem Entpacken von TestSetFrames.zip keine JPGs gefunden – Archiv wird NICHT gelöscht.")
    else:
        print(f"[prepare][Okutama][WARN] TestSetFrames.zip nicht gefunden unter {OKUTAMA_ROOT}")

    train_jpg = sum(1 for _ in train_target.rglob("*.jpg")) if train_target.exists() else 0
    test_jpg = sum(1 for _ in test_target.rglob("*.jpg")) if test_target.exists() else 0
    print(f"[prepare][Okutama] JPGs in TrainSetFrames: {train_jpg}, in TestSetFrames: {test_jpg}")


def main():
    print(f"[prepare] DATA_ROOT = {DATA_ROOT}")
    prepare_visdrone()
    prepare_uavdt()
    prepare_okutama()
    print("\n[prepare] Vorbereitung abgeschlossen. Archive wurden, falls möglich, gelöscht.")


if __name__ == "__main__":
    main()
