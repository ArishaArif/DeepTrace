from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from fractal.config import FEATURES, METHODS, SEED

RESULTS_PATH = ROOT / "sanity_results.csv"
PLOT_PATH = ROOT / "sanity_plot.png"


def patch_level(frames):
    rows = []
    for method in METHODS:
        df = frames[method]
        y = df["label"].to_numpy()
        for feat in FEATURES:
            auc = roc_auc_score(y, df[feat].to_numpy())
            direction = "fake" if auc >= 0.5 else "real"
            rows.append(
                {
                    "method": method,
                    "feature": feat,
                    "level": "patch",
                    "auc_signed": auc,
                    "auc_abs": max(auc, 1.0 - auc),
                    "higher_in": direction,
                }
            )
    return pd.DataFrame(rows)


def image_level(frames):
    out = {}
    for method in METHODS:
        out[method] = (
            frames[method].groupby("image_id", as_index=False)
            .agg({**{f: "mean" for f in FEATURES}, "label": "first"})
            .sort_values("image_id")
            .reset_index(drop=True)
        )
    return out


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def combined_cv(df, extra=()):
    cols = list(FEATURES) + list(extra)
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs, accs = [], []
    for train_idx, test_idx in skf.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], pred))
        accs.append(accuracy_score(y[test_idx], (pred >= 0.5).astype(int)))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs))


def plot_histograms(df, method):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    good = df[df["label"] == 0]
    bad = df[df["label"] == 1]
    for ax, feat in zip(axes, FEATURES):
        bins = np.linspace(
            float(min(df[feat].quantile(0.01), df[feat].quantile(0.01))),
            float(max(df[feat].quantile(0.99), df[feat].quantile(0.99))),
            50,
        )
        ax.hist(good[feat], bins=bins, density=True, alpha=0.55, label="real", color="#3b6ea5")
        ax.hist(bad[feat], bins=bins, density=True, alpha=0.55, label="fake", color="#b5483b")
        ax.set_title(f"{feat} ({method})")
        ax.set_xlabel(feat)
        ax.set_ylabel("density")
        ax.legend()
    fig.suptitle(f"Feature distributions by class, binarization = {method}")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def main():
    np.random.seed(SEED)
    frames = {m: pd.read_parquet(ROOT / f"features_sample_{m}.parquet") for m in METHODS}
    patch = patch_level(frames)
    img = image_level(frames)

    img_rows = []
    for method in METHODS:
        df = img[method]
        y = df["label"].to_numpy()
        real = df[df["label"] == 0]
        fake = df[df["label"] == 1]
        for feat in FEATURES:
            auc = roc_auc_score(y, df[feat].to_numpy())
            img_rows.append(
                {
                    "method": method,
                    "feature": feat,
                    "level": "image",
                    "auc_signed": auc,
                    "auc_abs": max(auc, 1.0 - auc),
                    "higher_in": "fake" if auc >= 0.5 else "real",
                    "cohens_d": cohens_d(fake[feat].to_numpy(), real[feat].to_numpy()),
                }
            )
    img_df = pd.DataFrame(img_rows)

    combo_rows = []
    best = None
    for method in METHODS:
        auc_m, auc_s, acc_m, acc_s = combined_cv(img[method])
        combo_rows.append(
            {
                "method": method,
                "feature": "combined_lr_5fold",
                "level": "image",
                "auc_signed": auc_m,
                "auc_abs": auc_m,
                "higher_in": "fake",
                "auc_std": auc_s,
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
            }
        )
        print(f"{method:9s} combined CV AUC {auc_m:.4f} +- {auc_s:.4f} | acc {acc_m:.4f}")
        if best is None or auc_m > best[1]:
            best = (method, auc_m)

    patch["cohens_d"] = np.nan
    results = pd.concat([patch, img_df, pd.DataFrame(combo_rows)], ignore_index=True)
    results = results[
        [
            "method",
            "feature",
            "level",
            "auc_signed",
            "auc_abs",
            "higher_in",
            "cohens_d",
            "auc_std",
            "accuracy_mean",
            "accuracy_std",
        ]
    ]
    results.to_csv(RESULTS_PATH, index=False)

    best_method = best[0]
    print(f"\nbest method: {best_method} (AUC {best[1]:.4f})")
    plot_histograms(img[best_method], best_method)

    src = ROOT / f"features_sample_{best_method}.parquet"
    dst = ROOT / "features_sample.parquet"
    dst.write_bytes(src.read_bytes())
    (ROOT / "default_method.txt").write_text(best_method + "\n", encoding="utf-8")
    print("saved", RESULTS_PATH.name, PLOT_PATH.name, dst.name)
    return best_method


if __name__ == "__main__":
    main()
