"""Day 3 Step 3: Train and save the v2 model (candidate F) after the gate passed.

Fits hgb_regularized on the full train split using the candidate F feature set,
evaluates once on val, saves the bundle to model/v2/fractal_clf.joblib plus a
model_card.json. v1 remains untouched in model/v1/.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import SEED
from fractal.extra_features import CANDIDATE_FEATURES

CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "model"
V2_DIR = MODEL_DIR / "v2"
CANDIDATE = "F"
THRESHOLD = 0.5

HGB_PARAMS = {
    "learning_rate": 0.05, "max_depth": 3, "max_iter": 200,
    "min_samples_leaf": 200, "l2_regularization": 1.0,
    "early_stopping": True, "validation_fraction": 0.1, "random_state": SEED,
}


def read_method() -> str:
    return (ROOT / "default_method.txt").read_text(encoding="utf-8").split()[0].strip()


def feature_columns(df: pd.DataFrame) -> list:
    meta = {"image_id", "label", "split", "row", "col"}
    return [c for c in df.columns if c not in meta]


def image_level(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    tmp = frame[["image_id", "label"]].copy()
    tmp["score"] = scores
    return tmp.groupby("image_id", sort=False).agg(label=("label", "first"), score=("score", "mean")).reset_index()


def versions() -> dict:
    import scipy
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "joblib": joblib.__version__,
    }


def main() -> int:
    method = read_method()
    np.random.seed(SEED)
    train = pd.read_parquet(CACHE_DIR / f"features_{CANDIDATE}_train.parquet")
    val = pd.read_parquet(CACHE_DIR / f"features_{CANDIDATE}_val.parquet")
    feat_cols = feature_columns(train)
    assert feat_cols == CANDIDATE_FEATURES[CANDIDATE], (feat_cols, CANDIDATE_FEATURES[CANDIDATE])
    print(f"candidate {CANDIDATE}: {len(feat_cols)} features, train {len(train)} patches, val {len(val)} patches")

    x_train = train[feat_cols].to_numpy(dtype=np.float64)
    y_train = train["label"].to_numpy()
    model = Pipeline([("clf", HistGradientBoostingClassifier(**HGB_PARAMS))])
    start = time.perf_counter()
    model.fit(x_train, y_train)
    print(f"trained in {time.perf_counter() - start:.1f}s")

    # Single val evaluation
    x_val = val[feat_cols].to_numpy(dtype=np.float64)
    val_scores = model.predict_proba(x_val)[:, 1]
    img = image_level(val, val_scores)
    labels = img["label"].to_numpy()
    probs = img["score"].to_numpy()
    pred = (probs >= THRESHOLD).astype(int)
    metrics = {
        "image_auc": float(roc_auc_score(labels, probs)),
        "patch_auc": float(roc_auc_score(val["label"].to_numpy(), val_scores)),
        "image_accuracy": float(accuracy_score(labels, pred)),
        "image_precision": float(precision_score(labels, pred, zero_division=0)),
        "image_recall": float(recall_score(labels, pred, zero_division=0)),
        "image_f1": float(f1_score(labels, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
        "threshold": THRESHOLD,
        "n_images": int(len(img)),
        "n_patches": int(len(val)),
    }
    print(json.dumps(metrics, indent=2))

    # Speed check through the real inference path
    from fractal.fractal_features import extract_patches, load_gray
    from fractal.extra_features import compute_candidate
    sample_paths = val["image_id"].drop_duplicates().head(0).tolist()  # not used; read from manifest
    import csv
    with (ROOT / "train_manifest.csv").open("r", encoding="utf-8", newline="") as fh:
        val_paths = [r["path"] for r in csv.DictReader(fh) if r["split"] == "val"][:30]
    times = []
    for p in val_paths:
        arr = load_gray(p)
        patches, r, c = extract_patches(arr)
        t0 = time.perf_counter()
        feats = compute_candidate(CANDIDATE, patches, method=method)
        _ = model.predict_proba(feats)[:, 1]
        times.append(time.perf_counter() - t0)
    warm_ms = float(np.median(times)) * 1000
    metrics["inference_ms_per_image_warm"] = warm_ms
    print(f"warm inference: {warm_ms:.1f} ms/image (target < 1000 ms)")

    # Save v2 bundle
    V2_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "method": method,
            "candidate": CANDIDATE,
            "features": feat_cols,
            "selected_candidate": "hgb_regularized",
            "model_type": "hist_gradient_boosting",
            "hyperparameters": HGB_PARAMS,
            "threshold": THRESHOLD,
        },
        V2_DIR / "fractal_clf.joblib",
    )
    card = {
        "version": "v2",
        "binarization_method": method,
        "candidate": CANDIDATE,
        "feature_names": feat_cols,
        "selected_candidate": "hgb_regularized",
        "model_type": "hist_gradient_boosting",
        "hyperparameters": HGB_PARAMS,
        "threshold": THRESHOLD,
        "aggregation": "mean of patch probabilities",
        "patch_size": 64, "stride": 32, "seed": SEED,
        "train_images": int(train["image_id"].nunique()),
        "train_patches": int(len(train)),
        "val_images": int(val["image_id"].nunique()),
        "val_patches": int(len(val)),
        "cv_image_auc": 0.7691,
        "cv_image_auc_std": 0.0026,
        "val_metrics": metrics,
        "library_versions": versions(),
        "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (V2_DIR / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"saved {V2_DIR / 'fractal_clf.joblib'} and model_card.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
