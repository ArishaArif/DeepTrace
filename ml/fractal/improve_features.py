"""Day 3: Extract candidate feature sets for ablation comparison.

Usage:
    python -m fractal.improve_features [--force] [--candidates A,B,C,D,E] [--benchmark-only]

Caches features per candidate per split to fractal/cache/features_<candidate>_<split>.parquet.
Resumable: existing cache files are skipped unless --force.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import PATCH, SEED, STRIDE
from fractal.fractal_features import extract_patches, load_gray

CACHE_DIR = ROOT / "cache"
MANIFEST = ROOT / "train_manifest.csv"
CANDIDATES = ("A", "B", "C", "D", "E", "F")
BENCHMARK_N = 200
MAX_FULL_RUNTIME_S = 40 * 60  # 40 minutes


def _read_manifest() -> list:
    import csv

    with MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _work_one(args_tuple):
    """Worker: load image, extract patches, compute features for one candidate."""
    path, candidate = args_tuple
    from fractal.fractal_features import extract_patches, load_gray
    from fractal.extra_features import compute_candidate

    try:
        arr = load_gray(path)
        patches, rows, cols = extract_patches(arr)
        feats = compute_candidate(candidate, patches)
        return (path, candidate, feats.astype(np.float32), rows, cols, True, "")
    except Exception as exc:
        return (path, candidate, None, 0, 0, False, f"{type(exc).__name__}: {exc}")


def _feature_names(candidate: str) -> list:
    from fractal.extra_features import CANDIDATE_FEATURES
    if CANDIDATE_FEATURES[candidate] is not None:
        return CANDIDATE_FEATURES[candidate]
    return [f"f{i}" for i in range(20)]


def extract_split(split: str, items: list, candidate: str, workers: int, force: bool) -> Path | None:
    """Extract features for one candidate on one split. Returns cache path or None."""
    out_path = CACHE_DIR / f"features_{candidate}_{split}.parquet"
    if out_path.exists() and not force:
        print(f"  [{candidate}/{split}] cache exists, skipping")
        return out_path
    if not items:
        print(f"  [{candidate}/{split}] no images")
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(row["path"], candidate) for row in items]

    results = []
    skipped = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_work_one, tasks, chunksize=8):
            path, cand, feats, rows, cols, ok, reason = res
            if ok:
                results.append((path, feats, rows, cols))
            else:
                skipped.append(reason)

    if not results:
        print(f"  [{candidate}/{split}] all failed!")
        return None

    # Build parquet
    n_features = results[0][1].shape[1]
    feat_names = _feature_names(candidate)
    # Pad names if needed
    while len(feat_names) < n_features:
        feat_names.append(f"extra_{len(feat_names)}")
    feat_names = feat_names[:n_features]

    # Map paths to image_ids
    path_to_id = {row["path"]: row["image_id"] for row in items}
    path_to_label = {row["path"]: int(row["label"]) for row in items}

    all_rows = []
    for path, feats, rows, cols in results:
        image_id = path_to_id.get(path, Path(path).stem)
        label = path_to_label.get(path, -1)
        for r in range(rows):
            for c in range(cols):
                all_rows.append((image_id, label, split, r, c, *feats[r * cols + c].tolist()))

    import pandas as pd
    columns = ["image_id", "label", "split", "row", "col"] + feat_names
    df = pd.DataFrame(all_rows, columns=columns)
    df["row"] = df["row"].astype(np.int16)
    df["col"] = df["col"].astype(np.int16)
    df["label"] = df["label"].astype(np.int8)
    df.to_parquet(out_path, index=False)
    if skipped:
        print(f"  [{candidate}/{split}] saved {len(df)} rows, skipped {len(skipped)}")
    else:
        print(f"  [{candidate}/{split}] saved {len(df)} rows")
    return out_path


def benchmark(items: list, candidates: list, workers: int) -> dict:
    """Time each candidate on a subset of images, return estimated full runtime."""
    sample = items[:BENCHMARK_N]
    total_images = len(items)
    timings = {}
    for cand in candidates:
        tasks = [(row["path"], cand) for row in sample]
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_work_one, tasks, chunksize=8))
        elapsed = time.perf_counter() - start
        rate = elapsed / max(1, len(sample))
        est_total = rate * total_images
        timings[cand] = {"rate_ms": rate * 1000, "est_total_s": est_total}
        status = "OK" if est_total <= MAX_FULL_RUNTIME_S else "SKIP (>40 min)"
        print(f"  candidate {cand}: {rate * 1000:.1f} ms/image, estimated total {est_total:.0f}s [{status}]")
    return timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--candidates", type=str, default="A,B,C,D,E")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    workers = args.workers or max(1, min(os.cpu_count() or 1, 8))
    cands = [c.strip() for c in args.candidates.split(",")]

    rows = _read_manifest()
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)

    train_items = by_split.get("train", [])
    val_items = by_split.get("val", [])
    print(f"manifest: train={len(train_items)}, val={len(val_items)}, workers={workers}")

    # Benchmark
    print("\n=== BENCHMARK ===")
    all_items = train_items  # benchmark on train (larger)
    timings = benchmark(all_items, cands, workers)

    if args.benchmark_only:
        return 0

    # Extract
    print("\n=== EXTRACTION ===")
    for cand in cands:
        if timings[cand]["est_total_s"] > MAX_FULL_RUNTIME_S:
            print(f"  [{cand}] SKIPPED: estimated {(timings[cand]['est_total_s']/60):.0f} min > 40 min limit")
            continue
        # For train and val
        extract_split("train", train_items, cand, workers, args.force)
        extract_split("val", val_items, cand, workers, args.force)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
