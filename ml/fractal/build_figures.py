"""Day 3 Step 6: Build report figures in fractal/figures/.

  1. box_counting_example.png   log-log box counting plot with fitted line and D
  2. lacunarity_example.png     lacunarity vs box size for a real and a fake patch
  3. feature_distributions.png  histograms of core features, real vs fake (val)
  4. pipeline_diagram.png       flow diagram drawn with matplotlib patches
  5. copies of heatmap, deletion and robustness figures
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import BOX_SIZES, LACUNARITY_SIZES, PATCH, SEED
from fractal.fractal_features import (
    _box_counts,
    _loglog_slope,
    binarize,
    extract_patches,
    lacunarity,
    load_gray,
)

FIG_DIR = ROOT / "figures"
CACHE_DIR = ROOT / "cache"
MANIFEST = ROOT / "train_manifest.csv"
METHOD = "highpass"


def _read_method() -> str:
    return (ROOT / "default_method.txt").read_text(encoding="utf-8").split()[0].strip()


def _val_rows() -> list:
    import csv

    with MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == "val"]


def _pick_patches():
    """Return one real and one fake representative val patch (64x64 float)."""
    rows = _val_rows()
    rng = np.random.default_rng(SEED)
    reals = [r for r in rows if int(r["label"]) == 0]
    fakes = [r for r in rows if int(r["label"]) == 1]
    real_row = reals[rng.integers(len(reals))]
    fake_row = fakes[rng.integers(len(fakes))]

    def center_patch(path):
        arr = load_gray(path)
        patches, r, c = extract_patches(arr)
        # take the center patch
        idx = (r // 2) * c + (c // 2)
        return patches[idx]

    return center_patch(real_row["path"]), center_patch(fake_row["path"])


def box_counting_example():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_p, fake_p = _pick_patches()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, patch, title in zip(axes, [real_p, fake_p], ["Real patch", "Fake patch"]):
        b = binarize(patch[None], method=_read_method())[0]
        counts, sizes = _box_counts(b[None].astype(bool), BOX_SIZES)
        counts = counts[0]
        d = _loglog_slope(counts[None], sizes)[0]
        log_inv = np.log(1.0 / sizes)
        log_c = np.log(np.maximum(counts, 1.0))
        ax.plot(log_inv, log_c, "o", label="box counts")
        xs = np.linspace(log_inv.min(), log_inv.max(), 50)
        ax.plot(xs, d * xs + np.mean(log_c) - d * np.mean(log_inv), "-", label=f"fit D={d:.2f}")
        ax.set_xlabel("log(1/box size)")
        ax.set_ylabel("log(box count)")
        ax.set_title(f"{title}, D = {d:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Box counting on the log-log scale")
    fig.savefig(FIG_DIR / "box_counting_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  box_counting_example.png")


def lacunarity_example():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_p, fake_p = _pick_patches()
    sizes = list(range(2, 33, 2))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, patch, title in zip(axes[:2], [real_p, fake_p], ["Real", "Fake"]):
        b = binarize(patch[None], method=_read_method())[0].astype(bool)
        lacs = [lacunarity(b[None], r)[0] for r in sizes]
        ax.plot(sizes, lacs, "o-")
        ax.set_xlabel("box size r")
        ax.set_ylabel("lacunarity Λ(r)")
        ax.set_title(f"{title} patch")
        ax.grid(True, alpha=0.3)
    # gliding box illustration
    ax = axes[2]
    ax.set_title("Gliding box illustration")
    grid = np.random.default_rng(SEED).integers(0, 2, (8, 8))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor="red", lw=2))
    ax.add_patch(Rectangle((2.5, 2.5), 3, 3, fill=False, edgecolor="blue", lw=2))
    ax.set_xlabel("window slides across the patch")
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Lacunarity vs box size and the gliding box")
    fig.savefig(FIG_DIR / "lacunarity_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  lacunarity_example.png")


def feature_distributions():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_parquet(CACHE_DIR / "features_val.parquet")
    feats = ["D", "lac4", "lac8", "lac16"]
    real = frame[frame["label"] == 0]
    fake = frame[frame["label"] == 1]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, f in zip(axes, feats):
        ax.hist(real[f], bins=50, alpha=0.6, label="real", density=True)
        ax.hist(fake[f], bins=50, alpha=0.6, label="fake", density=True)
        ax.set_title(f)
        ax.set_xlabel(f)
        ax.legend()
    fig.suptitle("Core feature distributions, real vs fake (val patches)")
    fig.savefig(FIG_DIR / "feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  feature_distributions.png")


def pipeline_diagram():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    steps = [
        "Input image", "Grayscale", "64x64 patches\n(stride 32)", "Per-patch\nfeatures\n(D, lac)",
        "Patch\nclassifier", "Patch score\nmatrix", "Heatmap +\nfractal_score",
    ]
    widths = [1.4, 1.3, 1.7, 1.7, 1.4, 1.4, 1.7]
    x = 0.3
    for label, w in zip(steps, widths):
        box = FancyBboxPatch((x, 1.5), w, 1.2, boxstyle="round,pad=0.05",
                             fc="#e8f0fe", ec="#34558b", lw=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, 2.1, label, ha="center", va="center", fontsize=8)
        if x + w < 11.5:
            arrow = FancyArrowPatch((x + w + 0.05, 2.1), (x + w + 0.4, 2.1),
                                    arrowstyle="-|>", mutation_scale=12, color="#34558b")
            ax.add_patch(arrow)
        x += w + 0.35
    ax.set_title("Fractal branch pipeline")
    fig.savefig(FIG_DIR / "pipeline_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  pipeline_diagram.png")


def copy_derived_figures():
    for name in ["heatmap_contact_sheet.png", "deletion_curve.png", "robustness.png",
                 "heatmap_preview.png", "position_bias.png"]:
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, FIG_DIR / name)
            print(f"  copied {name}")
        elif (FIG_DIR / name).is_file():
            print(f"  already in figures: {name}")


def main() -> int:
    FIG_DIR.mkdir(exist_ok=True)
    print("building figures:")
    box_counting_example()
    lacunarity_example()
    feature_distributions()
    pipeline_diagram()
    copy_derived_figures()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
