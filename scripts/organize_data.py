"""
Normalise the extracted IDRiD "Disease Grading" subset into the layout dataset.py
expects. Moves the 413 train / 103 test JPGs into data/raw/images/{train,test}/
and copies the two label CSVs to data/raw/labels/{train,test}.csv.

The segmentation subset (separate 81-image set, different numbering) is left in
data/raw/_staging/ and NOT wired into the grading manifest — lesion flags for the
grading set come from grade priors (see dataset.py), which avoids the train/test
filename-collision that reusing IDRiD_0xx names across splits would cause.

Run:  python scripts/organize_data.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
STAGE = RAW / "_staging"


def _find_dir(base: Path, *keywords: str) -> Path | None:
    if not base.exists():
        return None
    for p in base.rglob("*"):
        if p.is_dir() and all(k.lower() in p.name.lower() for k in keywords):
            return p
    return None


def _find_file(base: Path, *keywords: str, ext: str = ".csv") -> Path | None:
    if not base.exists():
        return None
    for p in base.rglob(f"*{ext}"):
        if p.is_file() and all(k.lower() in p.name.lower() for k in keywords):
            return p
    return None


def _move_images(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            shutil.move(str(p), str(dst / p.name))
            n += 1
    return n


def main() -> int:
    dg = _find_dir(STAGE, "disease grading") or STAGE
    orig = _find_dir(dg, "original images") or _find_dir(STAGE, "original images", "grading")
    if orig is None:
        # fall back: locate the folder that has ~413 jpgs
        orig = _find_dir(STAGE, "disease grading", "original")
    train_src = _find_dir(dg, "original", "training") or _find_dir(dg, "training set")
    test_src = _find_dir(dg, "original", "testing") or _find_dir(dg, "testing set")
    # disambiguate: the grading image folders are the ones with hundreds of jpgs
    for cand in (train_src, test_src):
        if cand is None:
            print("[error] could not locate grading image folders in", STAGE)
            return 1

    nt = _move_images(train_src, RAW / "images" / "train")
    ns = _move_images(test_src, RAW / "images" / "test")

    gt = _find_dir(dg, "groundtruth") or dg
    train_csv = _find_file(gt, "training") or _find_file(dg, "training")
    test_csv = _find_file(gt, "testing") or _find_file(dg, "testing")
    (RAW / "labels").mkdir(parents=True, exist_ok=True)
    if train_csv:
        shutil.copy(str(train_csv), str(RAW / "labels" / "train.csv"))
    if test_csv:
        shutil.copy(str(test_csv), str(RAW / "labels" / "test.csv"))

    print(f"[ok] images: train={nt}  test={ns}")
    print(f"[ok] labels: train.csv={'yes' if train_csv else 'MISSING'}  "
          f"test.csv={'yes' if test_csv else 'MISSING'}")
    print(f"[note] segmentation subset left in {STAGE.relative_to(ROOT)} (optional).")
    ok = nt > 0 and ns > 0 and train_csv and test_csv
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
