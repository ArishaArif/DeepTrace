from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import METHODS, SEED
from fractal.fractal_features import extract_patches, features_for_patches, load_gray

SAMPLE_DIR = PROJECT / "sample"
FEATURE_DIR = ROOT
COLUMNS = ["image_id", "label", "row", "col", "D", "lac4", "lac8", "lac16"]


def list_images():
    items = []
    for label, cls in ((0, "real"), (1, "fake")):
        for path in sorted((SAMPLE_DIR / cls).glob("*.jpg")):
            items.append((str(path), label, f"{cls}_{path.stem}"))
    return items


def work_one(item):
    path, label, image_id = item
    arr = load_gray(path)
    patches, rows, cols = extract_patches(arr)
    out = {}
    for method in METHODS:
        feats = features_for_patches(patches, method=method)
        out[method] = feats.astype(np.float32)
    return image_id, label, rows, cols, out


def build_frame(items, results):
    rr = np.concatenate([np.tile(np.arange(r[2]), r[3]) for r in results])
    cc = np.concatenate([np.repeat(np.arange(r[3]), r[2]) for r in results])
    image_ids = np.concatenate([np.array([r[0]] * (r[2] * r[3])) for r in results])
    labels = np.concatenate([np.array([r[1]] * (r[2] * r[3])) for r in results])
    frames = {}
    for method in METHODS:
        feats = np.concatenate([r[4][method] for r in results], axis=0)
        frames[method] = pd.DataFrame(
            {
                "image_id": image_ids,
                "label": labels.astype(np.int8),
                "row": rr.astype(np.int16),
                "col": cc.astype(np.int16),
                "D": feats[:, 0],
                "lac4": feats[:, 1],
                "lac8": feats[:, 2],
                "lac16": feats[:, 3],
            }
        )
    return frames


def main():
    from tqdm import tqdm

    np.random.seed(SEED)
    items = list_images()
    if not items:
        raise SystemExit(f"no sample images found under {SAMPLE_DIR}")
    workers = max(1, min(os.cpu_count() or 1, 8))
    print(f"extracting {len(items)} images with {workers} workers")
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in tqdm(pool.map(work_one, items, chunksize=4), total=len(items), unit="img"):
            results.append(res)
    frames = build_frame(items, results)
    for method, frame in frames.items():
        path = FEATURE_DIR / f"features_sample_{method}.parquet"
        frame.to_parquet(path, index=False)
        print(f"saved {path.name}: {len(frame)} rows")
    elapsed = time.perf_counter() - start
    print(f"runtime: {elapsed:.1f}s")
    return frames


if __name__ == "__main__":
    main()
