from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import SEED
from fractal.fractal_features import extract_patches, load_gray
from fractal.predict_fractal import predict_fractal

OUT_PREVIEW = ROOT / "heatmap_preview.png"
OUT_BIAS = ROOT / "position_bias.png"
OUT_JSON = ROOT / "heatmap_diagnostics.json"


def manifest_rows() -> list:
    with (ROOT / "train_manifest.csv").open("r", encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == "val"]


def score_all(rows: list) -> tuple:
    records = []
    grids = []
    for row in rows:
        out = predict_fractal(row["path"])
        records.append({"image_id": row["image_id"], "label": int(row["label"]), "fractal_score": float(out["fractal_score"])})
        grids.append(out["patch_scores"])
    shapes = {g.shape for g in grids}
    return records, grids, shapes


def preview(records: list, grids: list, rows: list) -> None:
    rng = np.random.default_rng(SEED)
    real_idx = [i for i, r in enumerate(records) if r["label"] == 0]
    fake_idx = [i for i, r in enumerate(records) if r["label"] == 1]
    picks = list(rng.choice(real_idx, size=6, replace=False)) + list(rng.choice(fake_idx, size=6, replace=False))
    fig, axes = plt.subplots(6, 4, figsize=(11, 16))
    for slot, idx in enumerate(picks):
        r = slot % 6
        col = 0 if slot < 6 else 2
        img = np.asarray(Image.open(rows[idx]["path"]).convert("RGB"))
        grid = grids[idx]
        up = np.asarray(
            Image.fromarray(np.clip(grid, 0.0, 1.0).astype(np.float32)).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
        )
        axes[r, col].imshow(img)
        axes[r, col].set_title(f"{'fake' if records[idx]['label'] else 'real'}  {records[idx]['fractal_score']:.3f}", fontsize=9)
        axes[r, col].axis("off")
        im = axes[r, col + 1].imshow(up, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[r, col + 1].set_title("patch score", fontsize=9)
        axes[r, col + 1].axis("off")
    fig.colorbar(im, ax=axes[:, 1], fraction=0.03, pad=0.02, label="fake probability")
    fig.suptitle("Fractal patch scores on validation images (left real, right fake)", fontsize=11)
    fig.savefig(OUT_PREVIEW, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PREVIEW.name}: 6 real and 6 fake")


def position_bias(records: list, grids: list, shapes: set) -> dict:
    by_label = {0: [], 1: []}
    for rec, grid in zip(records, grids):
        by_label[rec["label"]].append(grid)
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, label in zip(axes, (0, 1)):
        stack = np.stack(by_label[label], axis=0)
        mean_map = stack.mean(axis=0)
        im = ax.imshow(mean_map, cmap="inferno")
        ax.set_title(f"{'fake' if label else 'real'} mean patch score per position (n={stack.shape[0]})", fontsize=9)
        rows, cols = mean_map.shape
        border = np.concatenate([mean_map[0], mean_map[-1], mean_map[:, 0], mean_map[:, -1]])
        if rows > 2 and cols > 2:
            center = mean_map[1:-1, 1:-1].ravel()
        else:
            center = border
        out["fake" if label else "real"] = {
            "border_mean": float(border.mean()),
            "center_mean": float(center.mean()),
            "border_minus_center": float(border.mean() - center.mean()),
            "hotspot_position": [int(np.unravel_index(np.argmax(mean_map), mean_map.shape)[0]), int(np.unravel_index(np.argmax(mean_map), mean_map.shape)[1])],
            "hotspot_value": float(mean_map.max()),
            "mean_at_hotspot_overall": float(mean_map.mean()),
        }
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Position bias: mean patch score by grid position", fontsize=11)
    fig.savefig(OUT_BIAS, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_BIAS.name}")
    out["grid_shapes"] = sorted(str(s) for s in shapes)
    return out


def proxy_check(rows: list, records: list, grids: list) -> dict:
    scores, brightness, contrast = [], [], []
    for row, grid in zip(rows, grids):
        arr = load_gray(row["path"])
        patches, n_rows, n_cols = extract_patches(arr)
        scores.append(grid.ravel())
        brightness.append(patches.mean(axis=(1, 2)))
        contrast.append(patches.std(axis=(1, 2)))
    scores = np.concatenate(scores)
    brightness = np.concatenate(brightness)
    contrast = np.concatenate(contrast)
    r_b, p_b = pearsonr(scores, brightness)
    r_c, p_c = pearsonr(scores, contrast)
    out = {
        "pearson_r_brightness": float(r_b),
        "pearson_p_brightness": float(p_b),
        "pearson_r_contrast": float(r_c),
        "pearson_p_contrast": float(p_c),
        "n_patches": int(len(scores)),
        "warning": [],
    }
    if abs(r_b) >= 0.3:
        out["warning"].append("patch score tracks patch brightness")
    if abs(r_c) >= 0.3:
        out["warning"].append("patch score tracks patch contrast")
    return out


def border_hotspot(records: list, grids: list) -> dict:
    fake_grids = [g for rec, g in zip(records, grids) if rec["label"] == 1]
    on_border = 0
    for grid in fake_grids:
        r, c = np.unravel_index(np.argmax(grid), grid.shape)
        h, w = grid.shape
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            on_border += 1
    frac = on_border / max(1, len(fake_grids))
    return {
        "fake_images": len(fake_grids),
        "hotspot_on_border": on_border,
        "fraction_on_border": float(frac),
        "border_patch_fraction_of_grid": float(1.0 - ((fake_grids[0].shape[0] - 2) * (fake_grids[0].shape[1] - 2)) / fake_grids[0].size) if fake_grids else float("nan"),
    }


def main() -> int:
    rows = manifest_rows()
    if not rows:
        raise SystemExit("no val rows in manifest")
    print(f"scoring {len(rows)} val images")
    records, grids, shapes = score_all(rows)
    preview(records, grids, rows)
    report = {"position": position_bias(records, grids, shapes)}
    report["proxy"] = proxy_check(rows, records, grids)
    report["border_hotspot"] = border_hotspot(records, grids)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
