from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fractal.config import PATCH
from fractal.fractal_features import (
    binarize,
    extract_patches,
    fractal_dimension,
    lacunarity,
    load_gray,
    patch_features,
)


def _brute_lacunarity(b, r):
    m = np.asarray(b, dtype=np.float64)
    h, w = m.shape
    masses = []
    for y in range(h - r + 1):
        for x in range(w - r + 1):
            masses.append(m[y : y + r, x : x + r].sum())
    masses = np.asarray(masses, dtype=np.float64)
    mean_mass = masses.mean()
    if mean_mass <= 0:
        return 1.0
    return float((masses * masses).mean() / (mean_mass * mean_mass + 1e-12))


def _run(name, fn):
    try:
        fn()
        print(f"PASS {name}")
        return True
    except AssertionError as exc:
        print(f"FAIL {name}: {exc}")
        return False


def test_patch_grid_256():
    arr = np.random.default_rng(0).random((256, 256)).astype(np.float32)
    patches, rows, cols = extract_patches(arr)
    assert (rows, cols) == (7, 7), f"expected 7x7 got {rows}x{cols}"
    assert patches.shape == (49, PATCH, PATCH), f"unexpected shape {patches.shape}"


def test_patch_grid_128():
    arr = np.random.default_rng(0).random((128, 128)).astype(np.float32)
    patches, rows, cols = extract_patches(arr)
    assert (rows, cols) == (3, 3), f"expected 3x3 got {rows}x{cols}"
    assert patches.shape == (9, PATCH, PATCH), f"unexpected shape {patches.shape}"


def test_upscale_small_image():
    img = Image.new("L", (150, 100), color=128)
    arr = load_gray(img)
    assert min(arr.shape) == 128, f"expected shorter side 128, got {arr.shape}"
    assert min(arr.shape) == min(128, 128)


def test_fd_all_ones():
    b = np.ones((PATCH, PATCH), dtype=bool)
    d = fractal_dimension(b)
    assert abs(d - 2.0) <= 0.1, f"expected ~2, got {d}"


def test_fd_single_row():
    b = np.zeros((PATCH, PATCH), dtype=bool)
    b[PATCH // 2, :] = True
    d = fractal_dimension(b)
    assert abs(d - 1.0) <= 0.15, f"expected ~1, got {d}"


def test_fd_random_noise():
    rng = np.random.default_rng(42)
    b = rng.random((PATCH, PATCH)) < 0.5
    d = fractal_dimension(b)
    assert abs(d - 2.0) <= 0.15, f"expected ~2, got {d}"


def test_lacunarity_all_ones():
    b = np.ones((PATCH, PATCH), dtype=bool)
    for r in (4, 8, 16):
        lac = lacunarity(b, r)
        assert abs(lac - 1.0) <= 1e-6, f"r={r} expected 1, got {lac}"


def test_lacunarity_clumped_vs_spread():
    clumped = np.zeros((PATCH, PATCH), dtype=bool)
    clumped[:16, :16] = True
    rng = np.random.default_rng(42)
    idx = rng.choice(PATCH * PATCH, size=256, replace=False)
    spread = np.zeros(PATCH * PATCH, dtype=bool)
    spread[idx] = True
    spread = spread.reshape(PATCH, PATCH)
    lac_c = lacunarity(clumped, 8)
    lac_s = lacunarity(spread, 8)
    assert lac_c > lac_s * 1.05, f"clumped {lac_c} not clearly above spread {lac_s}"


def test_lacunarity_matches_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(5):
        b = rng.random((PATCH, PATCH)) < rng.uniform(0.2, 0.8)
        for r in (4, 8, 16):
            fast = lacunarity(b, r)
            ref = _brute_lacunarity(b, r)
            assert abs(fast - ref) <= 1e-6, f"r={r} fast {fast} ref {ref}"


def test_patch_features_shape():
    rng = np.random.default_rng(3)
    patch = rng.random((PATCH, PATCH)).astype(np.float32)
    feats = patch_features(patch, "mean")
    assert feats.shape == (4,), f"unexpected shape {feats.shape}"
    assert np.all(np.isfinite(feats)), f"non finite features {feats}"


def test_binarize_methods():
    rng = np.random.default_rng(5)
    patch = rng.random((PATCH, PATCH)).astype(np.float32)
    for method in ("mean", "otsu", "highpass"):
        b = binarize(patch, method)
        assert b.dtype == bool, f"{method} dtype {b.dtype}"
        assert b.shape == (PATCH, PATCH), f"{method} shape {b.shape}"
        assert 0 < b.mean() < 1, f"{method} degenerate mask"


TESTS = [
    ("patch grid 256x256 is 7x7 / 49", test_patch_grid_256),
    ("patch grid 128x128 is 3x3", test_patch_grid_128),
    ("100px shorter side upscales to 128", test_upscale_small_image),
    ("fractal dimension of all ones is 2", test_fd_all_ones),
    ("fractal dimension of single row is 1", test_fd_single_row),
    ("fractal dimension of 50% noise is 2", test_fd_random_noise),
    ("lacunarity of all ones is 1", test_lacunarity_all_ones),
    ("clumped lacunarity above spread", test_lacunarity_clumped_vs_spread),
    ("lacunarity matches brute force", test_lacunarity_matches_brute_force),
    ("patch features shape is (4,)", test_patch_features_shape),
    ("binarize methods produce masks", test_binarize_methods),
]


def main():
    np.random.seed(42)
    results = [_run(name, fn) for name, fn in TESTS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
