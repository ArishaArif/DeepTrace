from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import SEED

DATASET_ROOT_PLACEHOLDER = "<PATH_TO_FRONTEND_FOLDERS>"
FALLBACK_ROOT = Path(r"C:\Users\bakht\data\real_vs_fake\real-vs-fake")
MANIFEST_PATH = ROOT / "train_manifest.csv"
CACHE_DIR = ROOT / "cache"
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}
CLASSES = (("real", 0), ("fake", 1))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_root():
    candidate = Path(DATASET_ROOT_PLACEHOLDER)
    if candidate.exists() and candidate.is_dir() and any(candidate.iterdir()):
        return candidate, "dataset_root"
    if not FALLBACK_ROOT.exists():
        raise SystemExit(f"neither DATASET_ROOT nor fallback root is available: {FALLBACK_ROOT}")
    return FALLBACK_ROOT, "fallback_140k"


def split_dirs(root):
    found = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.lower() in SPLIT_ALIASES:
            found[SPLIT_ALIASES[child.name.lower()]] = child
    return found


def class_dirs(split_dir):
    found = {}
    for name, label in CLASSES:
        for child in sorted(split_dir.iterdir()):
            if child.is_dir() and child.name.lower() == name:
                found[name] = (child, label)
    return found


def list_images(split_dir, split_name):
    items = []
    for cls_name, (folder, label) in class_dirs(split_dir).items():
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                items.append(
                    {
                        "image_id": f"{split_name}_{cls_name}_{path.stem}",
                        "path": str(path),
                        "label": label,
                        "split": split_name,
                        "class": cls_name,
                        "generator": path.parent.name if path.parent != folder else "",
                    }
                )
    return items


def subsample(items, per_class, seed=SEED):
    rng = random.Random(seed)
    out = []
    for cls_name, _ in CLASSES:
        subset = [it for it in items if it["class"] == cls_name]
        if len(subset) > per_class:
            subset = rng.sample(subset, per_class)
        out.extend(subset)
    return sorted(out, key=lambda it: it["image_id"])


def ensure_train_manifest(per_class=10000):
    if MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH)
        if (df["label"] == 0).sum() > 0 and (df["label"] == 1).sum() > 0:
            return df
    root, source = resolve_root()
    dirs = split_dirs(root)
    if "train" not in dirs:
        raise SystemExit(f"no train split under {root}")
    items = subsample(list_images(dirs["train"], "train"), per_class)
    df = pd.DataFrame(items)
    df.to_csv(MANIFEST_PATH, index=False)
    return df


def val_items(per_class=2000):
    root, source = resolve_root()
    dirs = split_dirs(root)
    if "val" not in dirs:
        raise SystemExit(f"no val split under {root}")
    return subsample(list_images(dirs["val"], "val"), per_class)


def test_items(per_class=None):
    root, source = resolve_root()
    dirs = split_dirs(root)
    if "test" not in dirs:
        return []
    items = list_images(dirs["test"], "test")
    if per_class is not None:
        items = subsample(items, per_class)
    return items


def ensure_fallback_valid():
    import zipfile

    val_dir = FALLBACK_ROOT / "valid"
    if val_dir.exists() and any(val_dir.rglob("*.jpg")):
        return False
    zip_path = Path(r"C:\Users\bakht\140k-real-and-fake-faces.zip")
    if not zip_path.exists():
        raise SystemExit(f"validation split missing and zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m
            for m in zf.infolist()
            if "real-vs-fake/valid/" in m.filename and m.filename.lower().endswith(".jpg")
        ]
        print(f"extracting {len(members)} valid jpgs from zip")
        zf.extractall(FALLBACK_ROOT.parent.parent, members=members)
    return True

