from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

import importlib

from fractal import predict_fractal as pf_module  # the re-exported function
pf = importlib.import_module("fractal.predict_fractal")  # the module, for internals
from fractal.config import MIN_SIZE, PATCH, SEED, STRIDE
from fractal.fractal_features import extract_patches, features_for_patches, load_gray

CACHE_DIR = ROOT / "cache"
SPEED_TARGET = 1.0
SAMPLE_DIR = PROJECT / "sample"

RESULTS: list = []


def check(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn

    return deco


def _run(name, fn):
    try:
        fn()
        print(f"PASS {name}")
        return True
    except AssertionError as exc:
        print(f"FAIL {name}: {exc}")
        return False


def _sample_image(cls: str = "real", index: int = 0) -> Path:
    files = sorted((SAMPLE_DIR / cls).glob("*.jpg"))
    assert files, f"no sample images in {SAMPLE_DIR / cls}"
    return files[index]


def _grid(h: int, w: int) -> tuple:
    return (h - PATCH) // STRIDE + 1, (w - PATCH) // STRIDE + 1


@check("256x256 gives 7x7 scores in [0, 1] with mean equal to fractal_score")
def test_shape_256():
    img = Image.open(_sample_image()).convert("RGB")
    assert img.size == (256, 256), img.size
    out = pf.predict_fractal(img)
    scores = out["patch_scores"]
    assert isinstance(scores, np.ndarray), type(scores)
    assert scores.shape == (7, 7), scores.shape
    assert scores.dtype == np.float32, scores.dtype
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0), (scores.min(), scores.max())
    assert abs(out["fractal_score"] - float(scores.mean())) <= 1e-6, (out["fractal_score"], scores.mean())


@check("128x128 gives 3x3 and a 100px side is upscaled")
def test_small_inputs():
    base = Image.open(_sample_image()).convert("RGB")
    small = base.resize((128, 128), Image.BICUBIC)
    out = pf.predict_fractal(small)
    assert out["patch_scores"].shape == (3, 3), out["patch_scores"].shape
    tiny = base.resize((100, 100), Image.BICUBIC)
    arr = pf._normalize_input(tiny)
    assert min(arr.shape) == MIN_SIZE, arr.shape
    out2 = pf.predict_fractal(tiny)
    assert out2["patch_scores"].shape == (3, 3), out2["patch_scores"].shape


@check("path, PIL RGB, PIL gray and numpy inputs give identical results")
def test_input_parity():
    path = _sample_image()
    with Image.open(path) as opened:
        rgb = opened.convert("RGB").copy()
    gray = rgb.convert("L")
    numpy_rgb = np.asarray(rgb, dtype=np.uint8)
    numpy_gray = np.asarray(gray, dtype=np.uint8)
    results = {
        "path": pf.predict_fractal(path),
        "pil_rgb": pf.predict_fractal(rgb),
        "pil_gray": pf.predict_fractal(gray),
        "np_rgb": pf.predict_fractal(numpy_rgb),
        "np_gray": pf.predict_fractal(numpy_gray),
    }
    ref = results["path"]["patch_scores"]
    for name, out in results.items():
        diff = float(np.max(np.abs(out["patch_scores"] - ref)))
        assert diff <= 1e-6, f"{name} differs by {diff}"


@check("two calls on the same image are identical")
def test_determinism():
    path = _sample_image()
    first = pf.predict_fractal(path)
    second = pf.predict_fractal(path)
    assert np.array_equal(first["patch_scores"], second["patch_scores"]), "first and second call differ"
    assert first["fractal_score"] == second["fractal_score"], "score differs between calls"
    pf.reset_model_cache()
    third = pf.predict_fractal(path)
    assert np.array_equal(first["patch_scores"], third["patch_scores"]), "reloaded model differs"


@check("non square 256x384 gives the expected grid shape")
def test_non_square():
    base = Image.open(_sample_image()).convert("RGB")
    tall = base.resize((256, 384), Image.BICUBIC)
    out = pf.predict_fractal(tall)
    expected = _grid(384, 256)
    assert out["patch_scores"].shape == expected, (out["patch_scores"].shape, expected)
    wide = base.resize((384, 256), Image.BICUBIC)
    out2 = pf.predict_fractal(wide)
    expected2 = _grid(256, 384)
    assert out2["patch_scores"].shape == expected2, (out2["patch_scores"].shape, expected2)


@check("predict_fractal_json is JSON serializable")
def test_json():
    out = pf.predict_fractal_json(_sample_image())
    text = json.dumps(out)
    back = json.loads(text)
    assert isinstance(back["patch_scores"], list), type(back["patch_scores"])
    assert len(back["patch_scores"]) == 7 and len(back["patch_scores"][0]) == 7
    assert isinstance(back["fractal_score"], float), type(back["fractal_score"])
    assert abs(back["fractal_score"] - sum(sum(row) for row in back["patch_scores"]) / 49) <= 1e-6
    assert len(text) < 20000, f"payload unexpectedly large: {len(text)}"

@check("speed under the one second target per 256x256 image")
def test_speed():
    path = _sample_image()
    with Image.open(path) as opened:
        img = opened.convert("RGB").copy()
    pf.reset_model_cache()
    t0 = time.perf_counter()
    pf.predict_fractal(img)
    cold = time.perf_counter() - t0
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        pf.predict_fractal(img)
        times.append(time.perf_counter() - t0)
    warm = float(np.median(times))
    print(f"     speed: cold {cold * 1000:.0f} ms (model load {pf.LAST_LOAD_SECONDS[0] * 1000:.0f} ms), warm {warm * 1000:.0f} ms per 256x256 image")
    assert warm < SPEED_TARGET, f"warm inference {warm:.3f}s exceeds target {SPEED_TARGET}s"


@check("predict_fractal agrees with the cached feature pipeline on 20 val images")
def test_cache_parity():
    import pandas as pd
    from fractal.extra_features import CANDIDATE_FEATURES, compute_candidate

    model = pf._load_model()
    method = pf.MODEL["method"]
    candidate = pf.MODEL.get("candidate")
    manifest = {row["image_id"]: row["path"] for row in _manifest_rows()}

    if candidate:
        # v2: compare against the cached candidate feature set
        feat_cols = CANDIDATE_FEATURES[candidate]
        frame = pd.read_parquet(CACHE_DIR / f"features_{candidate}_val.parquet",
                                columns=["image_id", "label", "row", "col"] + feat_cols)
    else:
        # v1: core features only
        frame = pd.read_parquet(CACHE_DIR / "features_val.parquet",
                                columns=["image_id", "label", "row", "col", "D", "lac4", "lac8", "lac16"])

    ids = sorted(frame["image_id"].unique())
    rng = np.random.default_rng(SEED)
    picked = [ids[i] for i in rng.choice(len(ids), size=min(20, len(ids)), replace=False)]
    worst = 0.0
    for image_id in picked:
        path = manifest[image_id]
        arr = load_gray(path)
        patches, n_rows, n_cols = extract_patches(arr)
        if candidate:
            feats = compute_candidate(candidate, patches, method=method)
        else:
            feats = features_for_patches(patches, method=method)
        reference = model.predict_proba(feats)[:, 1].reshape(n_rows, n_cols)
        live = pf.predict_fractal(path)["patch_scores"]
        assert live.shape == reference.shape, (image_id, live.shape, reference.shape)
        worst = max(worst, float(np.max(np.abs(live - reference))))
    print(f"     parity ({candidate or 'v1'}): max absolute difference over {len(picked)} images = {worst:.2e}")
    assert worst <= 1e-5, f"inference and cache disagree by {worst}"


def _manifest_rows():
    import csv

    with (ROOT / "train_manifest.csv").open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


TESTS = RESULTS


def main():
    np.random.seed(SEED)
    results = [_run(name, fn) for name, fn in TESTS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())