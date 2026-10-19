"""Unit tests for extra_features.py — synthetic patch validation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import PATCH, SEED
from fractal.extra_features import (
    box_counting_r2,
    compute_candidate,
    extra_lacunarity,
    local_slopes,
    multi_rep_features,
    multifractal_features,
)
from fractal.fractal_features import binarize


def _make_patches() -> dict:
    """Generate synthetic test patches [1, PATCH, PATCH] float32 in [0,1]."""
    rng = np.random.default_rng(SEED)
    patches = {}
    # All-ones
    patches["all_ones"] = np.ones((1, PATCH, PATCH), dtype=np.float32)
    # Single horizontal line
    line = np.zeros((1, PATCH, PATCH), dtype=np.float32)
    line[0, PATCH // 2, :] = 1.0
    patches["line"] = line
    # Random noise
    patches["noise"] = rng.random((1, PATCH, PATCH)).astype(np.float32)
    # Smooth gradient (horizontal ramp, same for every row)
    grad = np.tile(np.linspace(0, 1, PATCH, dtype=np.float32), (PATCH, 1))
    patches["gradient"] = grad[None]  # (1, 64, 64)
    return patches


def _assert_finite(arr: np.ndarray, name: str):
    assert np.all(np.isfinite(arr)), f"{name}: found non-finite values {arr[~np.isfinite(arr)]}"


RESULTS = []


def check(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn
    return deco


@check("extra_lacunarity: no NaN, positive values")
def test_extra_lac():
    patches = _make_patches()
    for key, p in patches.items():
        b = binarize(p, method="highpass")
        lac = extra_lacunarity(b[None] if b.ndim == 2 else b)
        _assert_finite(lac, f"lacunarity({key})")
        assert np.all(lac >= 1.0), f"lacunarity({key}) below 1: {lac}"


@check("box_counting_r2: values in [0, 1] or 0 for degenerate")
def test_r2():
    patches = _make_patches()
    for key, p in patches.items():
        b = binarize(p, method="highpass")
        b_batch = b[None] if b.ndim == 2 else b
        r2 = box_counting_r2(b_batch)
        _assert_finite(r2, f"r2({key})")
        assert np.all(r2 >= 0.0) and np.all(r2 <= 1.0), f"r2({key}) out of [0,1]: {r2}"


@check("local_slopes: finite, slope_diff = slope_small - slope_large")
def test_slopes():
    patches = _make_patches()
    for key, p in patches.items():
        b = binarize(p, method="highpass")
        b_batch = b[None] if b.ndim == 2 else b
        sl = local_slopes(b_batch)
        _assert_finite(sl, f"slopes({key})")
        assert sl.shape == (1, 3), f"slopes({key}) shape {sl.shape}"
        # Verify diff column
        assert abs(sl[0, 2] - (sl[0, 0] - sl[0, 1])) < 1e-10, f"slope_diff inconsistent for {key}"


@check("multifractal: no NaN/inf, D_0 near 2 for noise patch")
def test_multifractal():
    patches = _make_patches()
    for key, p in patches.items():
        mf = multifractal_features(p.astype(np.float64))
        _assert_finite(mf, f"multifractal({key})")
        assert mf.shape == (1, 8), f"multifractal({key}) shape {mf.shape}"
        if key == "noise":
            d0_grad = mf[0, 0]
            # D_0 for noise should be high (close to 2 for 2D uniform coverage)
            assert d0_grad > 1.5, f"D_0 for noise should be > 1.5, got {d0_grad}"
        # Width should be non-negative (D_0 >= D_2 typically)
        # Allow small negative due to numerical issues
        assert mf[0, 3] > -0.5, f"width_grad very negative for {key}: {mf[0, 3]}"


@check("multi_rep: double the baseline columns")
def test_multi_rep():
    patches = _make_patches()
    for key, p in patches.items():
        mr = multi_rep_features(p)
        assert mr.shape[1] == 8, f"multi_rep({key}) expected 8 cols, got {mr.shape[1]}"
        _assert_finite(mr, f"multi_rep({key})")


@check("compute_candidate A-E: correct shapes, all finite")
def test_compute_candidates():
    patches = _make_patches()
    p = patches["noise"]  # [1, 64, 64]
    expected_cols = {"A": 4, "B": 7, "C": 7, "D": 12, "E": 8}
    for cand, ncols in expected_cols.items():
        feats = compute_candidate(cand, p)
        assert feats.shape == (1, ncols), f"candidate {cand} shape {feats.shape}, expected (1, {ncols})"
        _assert_finite(feats, f"candidate_{cand}")


@check("batch of 49 patches processes correctly")
def test_batch():
    rng = np.random.default_rng(SEED)
    p = rng.random((49, PATCH, PATCH)).astype(np.float32)
    for cand in ("A", "B", "C", "D", "E"):
        feats = compute_candidate(cand, p)
        assert feats.shape[0] == 49, f"batch {cand} rows {feats.shape[0]}"
        _assert_finite(feats, f"batch_{cand}")


def main():
    passed = 0
    for name, fn in RESULTS:
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(RESULTS)} extra feature tests passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
