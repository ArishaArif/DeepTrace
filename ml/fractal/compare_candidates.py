"""Day 3 Step 2: Compare candidate feature sets by cross-validation on train.

Usage:
    python -m fractal.compare_candidates [--best-candidate C]

Runs GroupKFold(5, groups=image_id) on the train split for each candidate,
evaluates with logistic regression and HistGradientBoosting, reports image
and patch level AUC. Then evaluates the best on val (single confirmation).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import SEED

CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "model"
CV_FOLDS = 5
THRESHOLD = 0.5

# Baseline from Day 2 (hgb_regularized, features A)
BASELINE_CV_IMAGE_AUC = 0.6867
BASELINE_CV_IMAGE_AUC_STD = 0.00732
BASELINE_VAL_IMAGE_AUC = 0.6934


def load_candidate_cache(candidate: str, split: str) -> pd.DataFrame | None:
    """Load cached features for a candidate/split combo."""
    path = CACHE_DIR / f"features_{candidate}_{split}.parquet"
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names (everything after row/col)."""
    meta = {"image_id", "label", "split", "row", "col"}
    return [c for c in df.columns if c not in meta]


def image_auc(frame: pd.DataFrame, scores: np.ndarray) -> float:
    tmp = frame[["image_id", "label"]].copy()
    tmp["score"] = scores
    img = tmp.groupby("image_id", sort=False).agg(label=("label", "first"), score=("score", "mean"))
    if img["label"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(img["label"].to_numpy(), img["score"].to_numpy()))


def cross_validate_candidate(candidate: str, model_name: str, model_proto) -> dict | None:
    """5-fold GroupKFold CV on train split for one candidate + model combo."""
    train = load_candidate_cache(candidate, "train")
    if train is None:
        print(f"  [{candidate}] no train cache, skipping")
        return None
    feat_cols = get_feature_columns(train)
    x = train[feat_cols].to_numpy(dtype=np.float64)
    y = train["label"].to_numpy()
    groups = train["image_id"].to_numpy()
    splitter = GroupKFold(n_splits=CV_FOLDS)
    folds = list(splitter.split(x, y, groups))

    fold_aucs = []
    oof = np.zeros(len(train), dtype=np.float64)
    for fit_idx, eval_idx in folds:
        from sklearn.base import clone
        model = clone(model_proto)
        model.fit(x[fit_idx], y[fit_idx])
        scores = model.predict_proba(x[eval_idx])[:, 1]
        oof[eval_idx] = scores
        eval_frame = train.iloc[eval_idx].reset_index(drop=True)
        fold_aucs.append(image_auc(eval_frame, scores))

    cv_mean = float(np.mean(fold_aucs))
    cv_std = float(np.std(fold_aucs))
    cv_patch_auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan")
    cv_image_auc_pooled = image_auc(train, oof)

    return {
        "candidate": candidate,
        "model": model_name,
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "fold_aucs": fold_aucs,
        "cv_image_auc": cv_mean,
        "cv_image_auc_std": cv_std,
        "cv_patch_auc": cv_patch_auc,
        "cv_image_auc_pooled": cv_image_auc_pooled,
    }


def paired_fold_comparison(folds_a: list, folds_b: list) -> dict:
    """Judge if improvement of b over a exceeds fold noise."""
    diffs = [b - a for a, b in zip(folds_a, folds_b)]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    # t-like statistic
    se = std_diff / np.sqrt(len(diffs)) if std_diff > 0 else 1e-12
    t_stat = mean_diff / se
    return {"mean_diff": mean_diff, "std_diff": std_diff, "t_stat": float(t_stat), "improves": mean_diff > 0 and abs(mean_diff) > std_diff}


def model_protos() -> list:
    """Same model families as Day 2."""
    return [
        (
            "logreg_C1.0",
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=SEED)),
            ]),
        ),
        (
            "hgb_regularized",
            Pipeline([
                ("clf", HistGradientBoostingClassifier(
                    learning_rate=0.05, max_depth=3, max_iter=200,
                    min_samples_leaf=200, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.1, random_state=SEED,
                )),
            ]),
        ),
    ]


def evaluate_val(best_candidate: str, best_model_name: str) -> dict | None:
    """Fit on full train, evaluate once on val."""
    train = load_candidate_cache(best_candidate, "train")
    val = load_candidate_cache(best_candidate, "val")
    if train is None or val is None:
        return None
    feat_cols = get_feature_columns(train)
    x_train = train[feat_cols].to_numpy(dtype=np.float64)
    y_train = train["label"].to_numpy()
    x_val = val[feat_cols].to_numpy(dtype=np.float64)
    y_val = val["label"].to_numpy()

    # Find the right model
    chosen = None
    for name, proto in model_protos():
        if name == best_model_name:
            chosen = proto
            break
    if chosen is None:
        return None

    from sklearn.base import clone
    model = clone(chosen)
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    val_img_auc = image_auc(val, val_scores)
    val_patch_auc = float(roc_auc_score(y_val, val_scores)) if len(np.unique(y_val)) > 1 else float("nan")
    # Accuracy at threshold
    tmp = val[["image_id", "label"]].copy()
    tmp["score"] = val_scores
    img = tmp.groupby("image_id", sort=False).agg(label=("label", "first"), score=("score", "mean"))
    pred = (img["score"].to_numpy() >= THRESHOLD).astype(int)
    val_acc = float(accuracy_score(img["label"].to_numpy(), pred))

    return {
        "val_image_auc": val_img_auc,
        "val_patch_auc": val_patch_auc,
        "val_accuracy": val_acc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(ROOT / "comparison_results.json"))
    args = parser.parse_args()

    candidates = ["A", "B", "C", "D", "E", "F"]
    results = []

    print("=" * 70)
    print("CANDIDATE COMPARISON: GroupKFold(5) on TRAIN split only")
    print("=" * 70)

    for cand in candidates:
        train_path = CACHE_DIR / f"features_{cand}_train.parquet"
        if not train_path.is_file():
            print(f"\n[{cand}] NOT AVAILABLE (extraction pending or skipped)")
            continue
        print(f"\n[{cand}] loaded train cache")
        for model_name, model_proto in model_protos():
            start = time.perf_counter()
            res = cross_validate_candidate(cand, model_name, model_proto)
            elapsed = time.perf_counter() - start
            if res is None:
                continue
            res["time_s"] = elapsed
            results.append(res)
            print(
                f"  {model_name:18s} image AUC {res['cv_image_auc']:.4f} +/- {res['cv_image_auc_std']:.4f} "
                f"patch AUC {res['cv_patch_auc']:.4f} ({elapsed:.0f}s)"
            )

    # Paired comparison vs baseline (A + hgb_regularized)
    baseline_result = next((r for r in results if r["candidate"] == "A" and r["model"] == "hgb_regularized"), None)
    comparisons = []
    if baseline_result:
        print(f"\n{'=' * 70}")
        print("PAIRED COMPARISON vs baseline A (hgb_regularized):")
        print(f"  baseline CV image AUC: {baseline_result['cv_image_auc']:.4f} +/- {baseline_result['cv_image_auc_std']:.4f}")
        print(f"{'=' * 70}")
        for res in results:
            if res["candidate"] == "A" or res["model"] != "hgb_regularized":
                continue
            comp = paired_fold_comparison(baseline_result["fold_aucs"], res["fold_aucs"])
            comp["candidate"] = res["candidate"]
            comp["model"] = res["model"]
            comp["cv_image_auc"] = res["cv_image_auc"]
            comparisons.append(comp)
            verdict = "IMPROVES" if comp["improves"] else "no significant gain"
            print(f"  {res['candidate']:3s} {res['model']:18s} diff {comp['mean_diff']:+.4f} +/- {comp['std_diff']:.4f} [{verdict}]")

    # Find best candidate+model by CV image AUC
    if not results:
        print("No results!")
        return 1
    best = max(results, key=lambda r: r["cv_image_auc"])
    print(f"\n{'=' * 70}")
    print(f"BEST: candidate {best['candidate']} with {best['model']} → CV image AUC {best['cv_image_auc']:.4f}")
    print(f"{'=' * 70}")

    # Single val confirmation
    print("\nVAL CONFIRMATION (fit on full train, evaluate once on val):")
    val_res = evaluate_val(best["candidate"], best["model"])
    if val_res:
        print(f"  val image AUC: {val_res['val_image_auc']:.4f} (baseline val AUC: {BASELINE_VAL_IMAGE_AUC:.4f})")
        print(f"  val patch AUC: {val_res['val_patch_auc']:.4f}")
        print(f"  val accuracy:  {val_res['val_accuracy']:.4f}")

    # Save all results
    output = {
        "results": results,
        "comparisons": comparisons,
        "best": {"candidate": best["candidate"], "model": best["model"], "cv_image_auc": best["cv_image_auc"]},
        "val_confirmation": val_res,
        "baseline": {
            "cv_image_auc": BASELINE_CV_IMAGE_AUC,
            "cv_image_auc_std": BASELINE_CV_IMAGE_AUC_STD,
            "val_image_auc": BASELINE_VAL_IMAGE_AUC,
        },
        "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
