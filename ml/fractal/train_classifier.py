from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import FEATURES, SEED

CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "model"
CV_FOLDS = 5
THRESHOLD = 0.5
FEATURE_COLUMNS = list(FEATURES)


def read_method() -> str:
    return (ROOT / "default_method.txt").read_text(encoding="utf-8").split()[0].strip()


def load_split(name: str) -> pd.DataFrame:
    path = CACHE_DIR / f"features_{name}.parquet"
    if not path.is_file():
        raise SystemExit(f"missing cache {path}; run extract_features.py first")
    frame = pd.read_parquet(path, columns=["image_id", "label", "row", "col"] + FEATURE_COLUMNS)
    return frame


def image_level(frame: pd.DataFrame, scores: np.ndarray, agg: str = "mean") -> pd.DataFrame:
    tmp = frame[["image_id", "label"]].copy()
    tmp["score"] = scores
    grouped = tmp.groupby("image_id", sort=False)
    out = grouped.agg(label=("label", "first"), score=("score", agg))
    return out.reset_index()


def image_auc(frame: pd.DataFrame, scores: np.ndarray, agg: str = "mean") -> float:
    img = image_level(frame, scores, agg)
    if img["label"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(img["label"].to_numpy(), img["score"].to_numpy()))


def patch_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def candidates() -> list:
    return [
        (
            "logreg_C0.1",
            "logistic_regression",
            {"C": 0.1, "max_iter": 500, "class_weight": "balanced"},
            Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(**{"C": 0.1, "max_iter": 500, "class_weight": "balanced"}))]),
        ),
        (
            "logreg_C1.0",
            "logistic_regression",
            {"C": 1.0, "max_iter": 500, "class_weight": "balanced"},
            Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(**{"C": 1.0, "max_iter": 500, "class_weight": "balanced"}))]),
        ),
        (
            "hgb_regularized",
            "hist_gradient_boosting",
            {"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 200, "l2_regularization": 1.0, "early_stopping": True, "validation_fraction": 0.1, "random_state": SEED},
            Pipeline(
                [
                    ("clf", HistGradientBoostingClassifier(**{"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 200, "l2_regularization": 1.0, "early_stopping": True, "validation_fraction": 0.1, "random_state": SEED})),
                ]
            ),
        ),
        (
            "hgb_shallow",
            "hist_gradient_boosting",
            {"learning_rate": 0.05, "max_depth": 2, "max_iter": 300, "min_samples_leaf": 500, "l2_regularization": 1.0, "early_stopping": True, "validation_fraction": 0.1, "random_state": SEED},
            Pipeline(
                [
                    ("clf", HistGradientBoostingClassifier(**{"learning_rate": 0.05, "max_depth": 2, "max_iter": 300, "min_samples_leaf": 500, "l2_regularization": 1.0, "early_stopping": True, "validation_fraction": 0.1, "random_state": SEED})),
                ]
            ),
        ),
    ]


def fit_predict(model: Pipeline, x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    model.fit(x_fit, y_fit)
    return model.predict_proba(x_eval)[:, 1]


def cross_validate(train: pd.DataFrame) -> list:
    x = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = train["label"].to_numpy()
    groups = train["image_id"].to_numpy()
    splitter = GroupKFold(n_splits=CV_FOLDS)
    folds = list(splitter.split(x, y, groups))
    results = []
    for name, family, params, _ in candidates():
        start = time.perf_counter()
        fold_scores = []
        oof = np.zeros(len(train), dtype=np.float64)
        for fit_idx, eval_idx in folds:
            proto = next(c for c in candidates() if c[0] == name)[3]
            scores = fit_predict(proto, x[fit_idx], y[fit_idx], x[eval_idx])
            oof[eval_idx] = scores
            fold_scores.append(image_auc(train.iloc[eval_idx].reset_index(drop=True), scores))
        elapsed = time.perf_counter() - start
        row = {
            "candidate": name,
            "family": family,
            "params": params,
            "image_auc_folds": fold_scores,
            "cv_image_auc": float(np.mean(fold_scores)),
            "cv_image_auc_std": float(np.std(fold_scores)),
            "cv_patch_auc": patch_auc(y, oof),
            "cv_image_auc_mean_agg": image_auc(train, oof),
        }
        results.append(row)
        print(
            f"CV {name:16s} image AUC {row['cv_image_auc']:.4f} +/- {row['cv_image_auc_std']:.4f} "
            f"patch AUC {row['cv_patch_auc']:.4f} ({elapsed:.0f}s)"
        )
    return results


def evaluate(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    img = image_level(frame, scores, "mean")
    labels = img["label"].to_numpy()
    probs = img["score"].to_numpy()
    pred = (probs >= THRESHOLD).astype(int)
    out = {
        "image_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "patch_auc": patch_auc(frame["label"].to_numpy(), scores),
        "image_accuracy": float(accuracy_score(labels, pred)),
        "image_precision": float(precision_score(labels, pred, zero_division=0)),
        "image_recall": float(recall_score(labels, pred, zero_division=0)),
        "image_f1": float(f1_score(labels, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
        "threshold": THRESHOLD,
        "n_images": int(len(img)),
        "n_patches": int(len(frame)),
        "calibration": {
            "mean_patch_prob_real": float(np.mean(scores[frame["label"].to_numpy() == 0])),
            "mean_patch_prob_fake": float(np.mean(scores[frame["label"].to_numpy() == 1])),
            "mean_image_prob_real": float(np.mean(probs[labels == 0])),
            "mean_image_prob_fake": float(np.mean(probs[labels == 1])),
        },
        "aggregation_diagnostic": {
            "mean": float(roc_auc_score(labels, image_level(frame, scores, "mean")["score"].to_numpy())) if len(np.unique(labels)) > 1 else float("nan"),
            "max": float(roc_auc_score(labels, image_level(frame, scores, "max")["score"].to_numpy())) if len(np.unique(labels)) > 1 else float("nan"),
        },
    }
    top = image_topk_scores(frame, scores, 0.2)
    out["aggregation_diagnostic"]["top20pct_mean"] = (
        float(roc_auc_score(labels, top)) if len(np.unique(labels)) > 1 else float("nan")
    )
    out["aggregation_diagnostic"]["n_images"] = int(len(img))
    return out


def image_topk_scores(frame: pd.DataFrame, scores: np.ndarray, frac: float) -> np.ndarray:
    tmp = frame[["image_id", "label"]].copy()
    tmp["score"] = scores
    grouped = tmp.groupby("image_id", sort=False)
    k = frac

    def pick(g):
        n = max(1, int(np.ceil(k * len(g))))
        return float(np.mean(np.sort(g["score"].to_numpy())[-n:]))

    return grouped.apply(pick, include_groups=False).to_numpy()


def versions() -> dict:
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    method = read_method()
    np.random.seed(SEED)
    train = load_split("train")
    print(f"train rows {len(train)}, images {train['image_id'].nunique()}")
    if args.skip_cv and (MODEL_DIR / "model_card.json").is_file():
        card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
        name = card["selected_candidate"]
        cv_rows = card["cross_validation"]
    else:
        cv_rows = cross_validate(train)
        best = max(cv_rows, key=lambda r: r["cv_image_auc"])
        name = best["candidate"]
    chosen = [c for c in candidates() if c[0] == name][0]
    print(f"selected {name}: {chosen[2]}")
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    model = chosen[3]
    model.fit(x_train, train["label"].to_numpy())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "method": method,
            "features": FEATURE_COLUMNS,
            "selected_candidate": name,
            "model_type": chosen[1],
            "hyperparameters": chosen[2],
            "threshold": THRESHOLD,
        },
        MODEL_DIR / "fractal_clf.joblib",
    )
    del x_train
    val = load_split("val")
    x_val = val[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    start = time.perf_counter()
    val_scores = model.predict_proba(x_val)[:, 1]
    print(f"val inference on {len(val)} patches took {time.perf_counter() - start:.1f}s")
    del x_val
    metrics = evaluate(val, val_scores)
    print(json.dumps(metrics, indent=2))
    card = {
        "binarization_method": method,
        "feature_names": FEATURE_COLUMNS,
        "selected_candidate": name,
        "model_type": chosen[1],
        "hyperparameters": chosen[2],
        "threshold": THRESHOLD,
        "aggregation": "mean of patch probabilities",
        "patch_size": 64,
        "stride": 32,
        "seed": SEED,
        "cv_folds": CV_FOLDS,
        "train_images": int(train["image_id"].nunique()),
        "train_patches": int(len(train)),
        "val_images": int(val["image_id"].nunique()),
        "val_patches": int(len(val)),
        "cross_validation": cv_rows,
        "val_metrics": metrics,
        "library_versions": versions(),
        "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (MODEL_DIR / "cv_results.json").write_text(json.dumps(cv_rows, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "candidate": r["candidate"],
                "family": r["family"],
                "cv_image_auc": r["cv_image_auc"],
                "cv_image_auc_std": r["cv_image_auc_std"],
                "cv_patch_auc": r["cv_patch_auc"],
                "cv_image_auc_mean_agg": r["cv_image_auc_mean_agg"],
                "params": json.dumps(r["params"]),
            }
            for r in cv_rows
        ]
    ).to_csv(MODEL_DIR / "cv_results.csv", index=False)
    print(f"saved {MODEL_DIR / 'fractal_clf.joblib'} and model_card.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
