"""Day 3 Step 5: Robustness of the fractal branch alone under common corruptions.

On 500 val images per class (seed 42), apply transformations and recompute the
image-level AUC and accuracy at 0.5 through predict_fractal.

Transformations: JPEG quality 90/75/50, half-size downscale+upscale (bicubic),
Gaussian blur sigma 1, Gaussian noise sigma 5 (on the 0-255 scale).
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import SEED
from fractal.predict_fractal import predict_fractal

MANIFEST = ROOT / "train_manifest.csv"
FIG_DIR = ROOT / "figures"
OUT_CSV = ROOT / "robustness.csv"
PER_CLASS = 500


def _val_rows() -> list:
    import csv

    with MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "val"]
    return rows


def select_balanced(rows: list, per_class: int) -> list:
    real = [r for r in rows if int(r["label"]) == 0]
    fake = [r for r in rows if int(r["label"]) == 1]
    rng = np.random.default_rng(SEED)
    real = list(rng.choice(real, size=min(per_class, len(real)), replace=False))
    fake = list(rng.choice(fake, size=min(per_class, len(fake)), replace=False))
    return real + fake


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _apply_resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    small = img.resize((w // 2, h // 2), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def _apply_blur(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    blurred = gaussian_filter(arr, sigma=(sigma, sigma, 0.0), mode="reflect")
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), mode="RGB")


def _apply_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    noisy = arr + rng_global.normal(0, sigma, size=arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


rng_global = np.random.default_rng(SEED)

TRANSFORMS = {
    "clean": lambda img: img,
    "jpeg_q90": lambda img: _apply_jpeg(img, 90),
    "jpeg_q75": lambda img: _apply_jpeg(img, 75),
    "jpeg_q50": lambda img: _apply_jpeg(img, 50),
    "resize_half": _apply_resize,
    "blur_sigma1": lambda img: _apply_blur(img, 1.0),
    "noise_sigma5": lambda img: _apply_noise(img, 5.0),
}


def evaluate_transform(name: str, fn, selected: list) -> dict:
    from tqdm import tqdm

    rng = np.random.default_rng(SEED)
    globals()["rng_global"] = rng
    labels = []
    scores = []
    for row in tqdm(selected, desc=name, unit="img", leave=False):
        img = _load_rgb(row["path"])
        transformed = fn(img)
        out = predict_fractal(transformed)
        labels.append(int(row["label"]))
        scores.append(out["fractal_score"])
    labels = np.array(labels)
    scores = np.array(scores)
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    acc = float(accuracy_score(labels, (scores >= 0.5).astype(int)))
    return {"transform": name, "n_images": len(labels), "image_auc": auc, "accuracy_at_0.5": acc}


def save_figure(results: list) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["transform"] for r in results]
    aucs = [r["image_auc"] for r in results]
    accs = [r["accuracy_at_0.5"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, aucs, width=0.4, label="Image AUC")
    ax.bar(x + 0.2, accs, width=0.4, label="Accuracy @ 0.5")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Fractal branch robustness under corruptions (val)")
    ax.legend()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "robustness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    rows = _val_rows()
    selected = select_balanced(rows, PER_CLASS)
    print(f"selected {len(selected)} val images ({PER_CLASS}/class)")

    results = []
    for name, fn in TRANSFORMS.items():
        res = evaluate_transform(name, fn, selected)
        results.append(res)
        print(f"  {name:14s} AUC {res['image_auc']:.4f}  acc {res['accuracy_at_0.5']:.4f}")

    frame = pd.DataFrame(results)
    frame.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV.name}")

    fig = save_figure(results)
    print(f"saved {fig}")

    # Honest report: which hurts most
    clean = next(r for r in results if r["transform"] == "clean")["image_auc"]
    degraded = sorted(results, key=lambda r: r["image_auc"])
    print("\nDegradation vs clean:")
    for r in degraded:
        print(f"  {r['transform']:14s} AUC {r['image_auc']:.4f} (delta {r['image_auc'] - clean:+.4f})")
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
