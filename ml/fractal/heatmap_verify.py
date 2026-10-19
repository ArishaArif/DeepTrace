"""Day 3 Step 4: Heatmap verification on 20 samples plus faithfulness tests.

Uses the final model through predict_fractal. Produces:
  - figures/heatmap_contact_sheet.png (4 rows x 5 cols)
  - figures/deletion_curve.png
  - quantitative checks on the full val set
  - faithfulness (deletion) test with a random-patch control
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import PATCH, SEED, STRIDE
from fractal.fractal_features import extract_patches, load_gray
from fractal.predict_fractal import predict_fractal

CACHE_DIR = ROOT / "cache"
FIG_DIR = ROOT / "figures"
MANIFEST = ROOT / "train_manifest.csv"
OUT_JSON = ROOT / "heatmap_verification.json"


def _manifest_val() -> list:
    import csv

    with MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == "val"]


def score_all_val(rows: list) -> pd.DataFrame:
    """Score every val image through predict_fractal, return per-image + patch matrix."""
    from tqdm import tqdm

    records = []
    for row in tqdm(rows, desc="score val", unit="img"):
        out = predict_fractal(row["path"])
        scores = out["patch_scores"]
        records.append({
            "image_id": row["image_id"],
            "label": int(row["label"]),
            "path": row["path"],
            "fractal_score": float(out["fractal_score"]),
            "patch_scores": scores,
        })
    return pd.DataFrame(records)


def select_samples(df: pd.DataFrame, n_per_class: int = 5) -> dict:
    """Pick 20 val images: 5 confident fakes, 5 confident reals, 5 FN, 5 FP."""
    rng = np.random.default_rng(SEED)
    fake = df[df["label"] == 1]
    real = df[df["label"] == 0]

    # Confidently correct fakes: highest scores
    confident_fake = fake.sort_values("fractal_score", ascending=False)
    # Confidently correct reals: lowest scores
    confident_real = real.sort_values("fractal_score", ascending=True)
    # False negatives: fakes scored low
    fn = fake.sort_values("fractal_score", ascending=True)
    # False positives: reals scored high
    fp = real.sort_values("fractal_score", ascending=False)

    def take(pool, k):
        if len(pool) < k:
            print(f"  note: only {len(pool)} images in a category, wanted {k}")
            return pool.head(len(pool))
        return pool.head(k)

    return {
        "confident_fake": take(confident_fake, n_per_class),
        "confident_real": take(confident_real, n_per_class),
        "false_negative": take(fn, n_per_class),
        "false_positive": take(fp, n_per_class),
    }


def save_contact_sheet(df: pd.DataFrame, samples: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ["confident_fake", "confident_real", "false_negative", "false_positive"]
    labels = {
        "confident_fake": "Correct fake",
        "confident_real": "Correct real",
        "false_negative": "Missed fake (FN)",
        "false_positive": "False alarm (FP)",
    }
    fig, axes = plt.subplots(4, 10, figsize=(20, 8), gridspec_kw={"wspace": 0.05, "hspace": 0.1})

    for row, cat in enumerate(categories):
        sub = samples[cat]
        for i in range(5):
            if i < len(sub):
                rec = sub.iloc[i]
                img = np.asarray(Image.open(rec["path"]).convert("RGB"))
                scores = rec["patch_scores"]
                # Image panel
                ax_img = axes[row, i * 2]
                ax_img.imshow(img)
                ax_img.set_xticks([]); ax_img.set_yticks([])
                if i == 0:
                    ax_img.set_ylabel(labels[cat], fontsize=9, rotation=0, ha="right", va="center")
                true = "FAKE" if rec["label"] == 1 else "REAL"
                ax_img.set_title(f"{true}\nscore {rec['fractal_score']:.2f}", fontsize=7)
                # Overlay panel
                ax_h = axes[row, i * 2 + 1]
                lo, hi = float(scores.min()), float(scores.max())
                norm = (scores - lo) / (hi - lo + 1e-9)
                ax_h.imshow(img)
                ax_h.imshow(norm, cmap="jet", alpha=0.6, extent=(0, img.shape[1], img.shape[0], 0), aspect="auto")
                ax_h.set_xticks([]); ax_h.set_yticks([])
            else:
                axes[row, i * 2].axis("off")
                axes[row, i * 2 + 1].axis("off")

    fig.suptitle("Heatmap contact sheet: image (left) and per-image normalized overlay (right)", fontsize=11)
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "heatmap_contact_sheet.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def quantitative_checks(df: pd.DataFrame) -> dict:
    """Full-val-set checks: heat concentration, center vs border, uniformity, proxy corr."""
    rows = []
    brightness_corr = []
    contrast_corr = []
    for rec in df.itertuples():
        scores = rec.patch_scores
        h, w = scores.shape
        # (a) heat concentration: fraction of total heat in top 20% of patches
        flat = scores.ravel()
        total = flat.sum() + 1e-9
        k = max(1, int(np.ceil(0.2 * len(flat))))
        top = np.sort(flat)[-k:]
        concentration = top.sum() / total
        # (b) center vs border mean
        mask_inner = np.zeros((h, w), dtype=bool)
        mask_inner[1:-1, 1:-1] = True
        center_mean = float(scores[mask_inner].mean())
        border_mean = float(scores[~mask_inner].mean())
        std = float(scores.std())
        rows.append({
            "label": rec.label,
            "concentration": concentration,
            "center_mean": center_mean,
            "border_mean": border_mean,
            "std": std,
        })
        # (d) patch score vs brightness/contrast, need raw patches
        arr = load_gray(rec.path)
        patches, r, c = extract_patches(arr)
        means = patches.mean(axis=(1, 2))
        stds = patches.std(axis=(1, 2))
        ps = scores.ravel()
        if ps.std() > 1e-9 and means.std() > 1e-9:
            brightness_corr.append(float(np.corrcoef(ps, means)[0, 1]))
        if ps.std() > 1e-9 and stds.std() > 1e-9:
            contrast_corr.append(float(np.corrcoef(ps, stds)[0, 1]))

    frame = pd.DataFrame(rows)
    fake = frame[frame["label"] == 1]
    real = frame[frame["label"] == 0]
    out = {
        "heat_concentration_top20pct": {
            "real_mean": float(real["concentration"].mean()),
            "fake_mean": float(fake["concentration"].mean()),
        },
        "center_vs_border": {
            "real_center": float(real["center_mean"].mean()),
            "real_border": float(real["border_mean"].mean()),
            "fake_center": float(fake["center_mean"].mean()),
            "fake_border": float(fake["border_mean"].mean()),
        },
        "flat_heatmaps_fraction_std_below_0.02": float((frame["std"] < 0.02).mean()),
        "brightness_correlation_mean": float(np.nanmean(brightness_corr)) if brightness_corr else float("nan"),
        "contrast_correlation_mean": float(np.nanmean(contrast_corr)) if contrast_corr else float("nan"),
        "n_images": len(frame),
    }
    return out


def deletion_test(df: pd.DataFrame, k_values=(2, 4, 8), n_fakes=200, n_random_draws=10) -> dict:
    """Blurring the k hottest patches vs random patches; measure fractal_score drop."""
    # 200 fake val images scored above 0.5
    pool = df[(df["label"] == 1) & (df["fractal_score"] > 0.5)].sort_values("fractal_score", ascending=False)
    use = pool.head(n_fakes)
    if len(use) < n_fakes:
        print(f"  note: only {len(use)} fake images above 0.5 (wanted {n_fakes})")

    rng = np.random.default_rng(SEED)
    results = {k: {"hottest": [], "random": []} for k in k_values}

    for rec in use.itertuples():
        arr = load_gray(rec.path)
        h, w = arr.shape
        scores = rec.patch_scores
        grid_h, grid_w = scores.shape
        base_score = predict_fractal(rec.path)["fractal_score"]

        flat_idx = np.argsort(scores.ravel())[::-1]  # hottest first

        for k in k_values:
            # Hottest patches
            drop_h = _blur_patches_and_rescore(arr, flat_idx[:k], grid_h, grid_w, h, w, base_score)
            results[k]["hottest"].append(drop_h)
            # Random control (mean over draws)
            drops_r = []
            for _ in range(n_random_draws):
                rand_idx = rng.choice(scores.size, size=k, replace=False)
                drops_r.append(_blur_patches_and_rescore(arr, rand_idx, grid_h, grid_w, h, w, base_score))
            results[k]["random"].append(float(np.mean(drops_r)))

    out = {}
    for k in k_values:
        hot = np.array(results[k]["hottest"])
        rnd = np.array(results[k]["random"])
        hot_ci = _bootstrap_ci(hot)
        rnd_ci = _bootstrap_ci(rnd)
        diff_ci = _bootstrap_ci(hot - rnd)
        out[k] = {
            "n_images": int(len(hot)),
            "hottest_mean_drop": float(np.mean(hot)),
            "hottest_ci95": hot_ci,
            "random_mean_drop": float(np.mean(rnd)),
            "random_ci95": rnd_ci,
            "mean_difference": float(np.mean(hot - rnd)),
            "difference_ci95": diff_ci,
            "faithful": bool(np.mean(hot - rnd) > 0 and diff_ci[0] > 0),
        }
    return {"per_k": out, "n_random_draws": n_random_draws, "blur_sigma": 4.0}


def _blur_patches_and_rescore(arr, patch_indices, grid_h, grid_w, h, w, base_score):
    """Replace the given patch regions with blurred versions, rescore, return drop."""
    blurred = gaussian_filter(arr, sigma=4.0, mode="reflect")
    modified = arr.copy()
    for idx in patch_indices:
        r = idx // grid_w
        c = idx % grid_w
        y0 = r * STRIDE
        x0 = c * STRIDE
        y1 = min(y0 + PATCH, h)
        x1 = min(x0 + PATCH, w)
        modified[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]
    new_score = predict_fractal(modified)["fractal_score"]
    return base_score - new_score


def _bootstrap_ci(values, n=1000, ci=95) -> list:
    rng = np.random.default_rng(SEED)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n)]
    lo = float(np.percentile(means, (100 - ci) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci) / 2))
    return [lo, hi]


def save_deletion_curve(del_res: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(int(k) for k in del_res["per_k"].keys())
    hot = [del_res["per_k"][k]["hottest_mean_drop"] for k in ks]
    rnd = [del_res["per_k"][k]["random_mean_drop"] for k in ks]
    hot_err = [
        [del_res["per_k"][k]["hottest_mean_drop"] - del_res["per_k"][k]["hottest_ci95"][0],
         del_res["per_k"][k]["hottest_ci95"][1] - del_res["per_k"][k]["hottest_mean_drop"]]
        for k in ks
    ]
    hot_err = np.array(hot_err).T
    rnd_err = np.array([
        [del_res["per_k"][k]["random_mean_drop"] - del_res["per_k"][k]["random_ci95"][0],
         del_res["per_k"][k]["random_ci95"][1] - del_res["per_k"][k]["random_mean_drop"]]
        for k in ks
    ]).T

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ks, hot, yerr=hot_err, marker="o", label="Hottest patches", capsize=3)
    ax.errorbar(ks, rnd, yerr=rnd_err, marker="s", label="Random patches (control)", capsize=3)
    ax.set_xlabel("Number of patches blurred (k)")
    ax.set_ylabel("Mean fractal_score drop")
    ax.set_title("Deletion faithfulness test")
    ax.legend(); ax.grid(True, alpha=0.3)
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "deletion_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    FIG_DIR.mkdir(exist_ok=True)
    rows = _manifest_val()
    print(f"scoring {len(rows)} val images through predict_fractal...")
    df = score_all_val(rows)

    print("\n=== QUANTITATIVE CHECKS (full val set) ===")
    quant = quantitative_checks(df)
    print(json.dumps(quant, indent=2))

    print("\n=== SELECT 20 SAMPLES + CONTACT SHEET ===")
    samples = select_samples(df)
    sheet = save_contact_sheet(df, samples)
    print(f"saved {sheet}")

    print("\n=== DELETION FAITHFULNESS TEST ===")
    del_res = deletion_test(df)
    curve = save_deletion_curve(del_res)
    print(f"saved {curve}")
    for k, v in del_res["per_k"].items():
        verdict = "FAITHFUL" if v["faithful"] else "NOT clearly faithful"
        print(f"  k={k}: hottest drop {v['hottest_mean_drop']:.4f}, random drop {v['random_mean_drop']:.4f}, "
              f"diff {v['mean_difference']:.4f} CI {v['difference_ci95']} [{verdict}]")

    output = {"quantitative": quant, "deletion": del_res, "date": str(pd.Timestamp.now())}
    OUT_JSON.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
