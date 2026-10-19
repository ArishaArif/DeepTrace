"""Day 1: dummy predictions. Day 2: real ones, picked up automatically
once the deep branch's ml/predict.py exists."""
import hashlib
import os
import threading

import numpy as np
from PIL import Image

from app import config
from app.config import THRESHOLD

# Limits how many model calls run at once (CPU/RAM protection). Extra requests simply wait.
INFER_SLOT = threading.BoundedSemaphore(config.MAX_CONCURRENT_JOBS)

try:
    if os.getenv("USE_DUMMY") == "1":
        raise ImportError
    from ml.predict import predict_image as _real_predict   # owned by DEEP BRANCH
    DUMMY = False
except ImportError:
    DUMMY = True


def _dummy_predict(path):
    with open(path, "rb") as f:
        seed = int(hashlib.md5(f.read()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    with Image.open(path) as im:
        w, h = im.size
    w, h = max(w, 128), max(h, 128)              # contract: min size 128
    rows, cols = (h - 64) // 32 + 1, (w - 64) // 32 + 1   # patch 64, stride 32

    patch_scores = rng.random((rows, cols)).round(3)
    score = float(patch_scores.mean())
    return {"score": score, "patch_scores": patch_scores.tolist()}


def predict_image(path):
    """Returns {verdict, score, patch_scores}."""
    out = _dummy_predict(path) if DUMMY else _real_predict(path)
    # Real ML code may return numpy types; make them plain JSON-safe Python.
    out["score"] = float(np.clip(float(out["score"]), 0.0, 1.0))
    out["patch_scores"] = np.clip(np.asarray(out["patch_scores"], dtype=np.float32), 0, 1).tolist()
    # Backend owns the threshold (config.THRESHOLD); ML code does not set the verdict.
    out["verdict"] = "fake" if out["score"] >= THRESHOLD else "real"
    return out
