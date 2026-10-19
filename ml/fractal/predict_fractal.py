from __future__ import annotations

import atexit
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

try:
    from .config import FEATURES, MIN_SIZE, PATCH, STRIDE
    from .fractal_features import extract_patches, features_for_patches, load_gray
except ImportError:
    from config import FEATURES, MIN_SIZE, PATCH, STRIDE
    from fractal_features import extract_patches, features_for_patches, load_gray

# v2 model (candidate F feature set) is the frozen Day 3 final model; v1 is kept
# intact in model/v1/. The bundle records which candidate to compute, so the same
# loader also works if pointed back at a v1 bundle (plain core features).
MODEL_PATH = Path(__file__).resolve().parent / "model" / "v2" / "fractal_clf.joblib"
MODEL: dict = {}
LAST_LOAD_SECONDS: list = [0.0]


def _normalize_input(image):
    if isinstance(image, (str, Path)):
        return load_gray(image)
    if isinstance(image, Image.Image):
        return load_gray(image)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return load_gray(Image.fromarray(arr.astype(np.uint8), mode="L"))
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            return load_gray(Image.fromarray(arr.astype(np.uint8), mode="RGBA"))
        if arr.shape[2] == 3:
            return load_gray(Image.fromarray(arr.astype(np.uint8), mode="RGB"))
        if arr.shape[2] == 1:
            return load_gray(Image.fromarray(arr[:, :, 0].astype(np.uint8), mode="L"))
    raise ValueError(f"unsupported input shape {arr.shape}")


def _load_model():
    if "model" in MODEL:
        return MODEL["model"]
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"trained model not found at {MODEL_PATH}")
    start = time.perf_counter()
    bundle = joblib.load(MODEL_PATH)
    MODEL["model"] = bundle["pipeline"]
    MODEL["method"] = bundle["method"]
    MODEL["candidate"] = bundle.get("candidate")  # None for v1 (core features only)
    LAST_LOAD_SECONDS[0] = time.perf_counter() - start
    return MODEL["model"]


def _score_array(arr):
    model = _load_model()
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D grayscale array, got {arr.shape}")
    patches, rows, cols = extract_patches(arr)
    candidate = MODEL.get("candidate")
    if candidate:
        from .extra_features import compute_candidate
        feats = compute_candidate(candidate, patches, method=MODEL["method"])
    else:
        feats = features_for_patches(patches, method=MODEL["method"])
    proba = model.predict_proba(feats)[:, 1].astype(np.float32)
    return proba.reshape(rows, cols)


def predict_fractal(image):
    arr = _normalize_input(image)
    patch_scores = _score_array(arr)
    return {
        "patch_scores": patch_scores,
        "fractal_score": float(patch_scores.mean()),
    }


def predict_fractal_json(image):
    out = predict_fractal(image)
    return {
        "patch_scores": out["patch_scores"].astype(float).tolist(),
        "fractal_score": float(out["fractal_score"]),
    }


def reset_model_cache():
    MODEL.clear()
    LAST_LOAD_SECONDS[0] = 0.0


atexit.register(reset_model_cache)


def _self_test():
    sample = Path(__file__).resolve().parent.parent / "sample" / "real"
    candidates = sorted(sample.glob("*.jpg"))
    if not candidates:
        raise SystemExit("no sample image available for the predict_fractal test")
    out = predict_fractal(candidates[0])
    assert out["patch_scores"].shape == (7, 7), out["patch_scores"].shape
    assert 0.0 <= out["fractal_score"] <= 1.0, out["fractal_score"]
    with Image.open(candidates[0]) as img:
        out2 = predict_fractal(img)
    assert out2["patch_scores"].shape == (7, 7), out2["patch_scores"].shape
    print("PASS predict_fractal shape (7, 7) and score in [0, 1]")
    print(f"sample fractal_score {out['fractal_score']:.4f}")


if __name__ == "__main__":
    _self_test()
