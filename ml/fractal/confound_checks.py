from __future__ import annotations

import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import FEATURES, SEED
from fractal.extract_sample import list_images
from fractal.fractal_features import extract_patches, features_for_patches, load_gray
from fractal.sanity_test import combined_cv, cohens_d

TMP_DIR = ROOT / "tmp_recompressed"
REPORT_PATH = ROOT / "confound_results.csv"
JPEG_QUALITY = 75


def image_metadata():
    rows = []
    for path, label, image_id in list_images():
        path = Path(path)
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
        arr = load_gray(path)
        rows.append(
            {
                "image_id": image_id,
                "label": label,
                "width": w,
                "height": h,
                "mode": mode,
                "file_bytes": path.stat().st_size,
                "brightness_mean": float(arr.mean()),
                "brightness_std": float(arr.std()),
                "path": path,
            }
        )
    return pd.DataFrame(rows)


def size_mode_report(meta):
    print("\n[1] image size / mode distribution per class")
    out = []
    for label, name in ((0, "real"), (1, "fake")):
        subset = meta[meta["label"] == label]
        combos = Counter(zip(subset["width"], subset["height"], subset["mode"]))
        print(f"  {name}: {len(subset)} images")
        for (w, h, mode), n in combos.most_common(5):
            print(f"    {w}x{h} {mode}: {n}")
            out.append({"check": "size_mode", "label": name, "value": f"{w}x{h} {mode}", "n": n})
    real_keys = set(zip(meta[meta["label"] == 0]["width"], meta[meta["label"] == 0]["height"], meta[meta["label"] == 0]["mode"]))
    fake_keys = set(zip(meta[meta["label"] == 1]["width"], meta[meta["label"] == 1]["height"], meta[meta["label"] == 1]["mode"]))
    if real_keys != fake_keys:
        print("  FLAG: classes differ in size or mode")
        out.append({"check": "size_mode", "label": "flag", "value": "classes differ", "n": 0})
    else:
        print("  classes match exactly on size and mode")
    return pd.DataFrame(out)


def auc_block(meta, cols, title):
    print(f"\n[{title}]")
    y = meta["label"].to_numpy()
    rows = []
    for col in cols:
        vals = meta[col].to_numpy(dtype=np.float64)
        auc = roc_auc_score(y, vals)
        real = meta.loc[meta["label"] == 0, col].to_numpy(dtype=np.float64)
        fake = meta.loc[meta["label"] == 1, col].to_numpy(dtype=np.float64)
        d = cohens_d(fake, real)
        direction = "fake" if auc >= 0.5 else "real"
        print(
            f"  {col:16s} AUC {auc:.4f} (abs {max(auc, 1 - auc):.4f}, higher in {direction}) "
            f"| real mean {real.mean():.4f} std {real.std():.4f} | fake mean {fake.mean():.4f} std {fake.std():.4f}"
            f" | d {d:+.3f}"
        )
        rows.append(
            {
                "check": title,
                "feature": col,
                "auc_signed": auc,
                "auc_abs": max(auc, 1 - auc),
                "higher_in": direction,
                "real_mean": real.mean(),
                "fake_mean": fake.mean(),
                "cohens_d": d,
            }
        )
    return rows


def recompress(paths):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            dst = TMP_DIR / Path(path).name
            img.save(dst, format="JPEG", quality=JPEG_QUALITY)
        out.append(str(dst))
    return out


def recompute_features(paths):
    feats = []
    for path in paths:
        arr = load_gray(path)
        patches, _, _ = extract_patches(arr)
        feats.append(features_for_patches(patches, method="highpass").mean(axis=0))
    return np.asarray(feats, dtype=np.float64)


def main():
    np.random.seed(SEED)
    start = time.perf_counter()
    meta = image_metadata()
    rows = []
    rows.extend(size_mode_report(meta).to_dict("records"))

    print("\n[2] JPEG file size")
    size_rows = auc_block(meta, ["file_bytes"], "file_size")
    if size_rows[0]["auc_abs"] > 0.65:
        print("  WARNING: file size separates the classes, compression may leak the label")
    rows.extend(size_rows)

    best_method = (ROOT / "default_method.txt").read_text(encoding="utf-8").strip()
    features_df = pd.read_parquet(ROOT / f"features_sample_{best_method}.parquet")
    image_orig = (
        features_df.groupby("image_id", as_index=False)
        .agg({**{f: "mean" for f in FEATURES}, "label": "first"})
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    auc_orig, std_orig, acc_orig, _ = combined_cv(image_orig)
    print(f"\n[3] re-compression test at quality {JPEG_QUALITY} (method {best_method})")
    print(f"  original combined AUC {auc_orig:.4f} +- {std_orig:.4f}")

    paths = meta["path"].tolist()
    labels = meta["label"].to_numpy()
    image_ids = meta["image_id"].tolist()
    try:
        t0 = time.perf_counter()
        tmp_paths = recompress(paths)
        recomp = recompute_features(tmp_paths)
        print(f"  recompressed + recomputed in {time.perf_counter() - t0:.1f}s")
        recomp_df = pd.DataFrame(recomp, columns=list(FEATURES))
        recomp_df["image_id"] = image_ids
        recomp_df["label"] = labels
        recomp_df = recomp_df.sort_values("image_id").reset_index(drop=True)
        auc_re, std_re, acc_re, _ = combined_cv(recomp_df)
        print(f"  recompressed combined AUC {auc_re:.4f} +- {std_re:.4f}")
        print(f"  drop {auc_orig - auc_re:+.4f}")
        rows.append(
            {
                "check": "recompression",
                "feature": "combined_lr_5fold",
                "auc_signed": auc_re,
                "auc_abs": auc_re,
                "auc_std": std_re,
                "accuracy_mean": acc_re,
                "original_auc": auc_orig,
                "drop": auc_orig - auc_re,
            }
        )
    finally:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        print("  removed", TMP_DIR.name)

    print("\n[4] brightness and contrast")
    bright_rows = auc_block(meta, ["brightness_mean", "brightness_std"], "brightness_contrast")
    rows.extend(bright_rows)
    strong = [r for r in bright_rows if r["auc_abs"] > 0.6]
    if strong:
        print("  WARNING: brightness or contrast alone separates the classes")

    control = image_orig.merge(
        meta[["image_id", "brightness_mean", "brightness_std", "file_bytes"]],
        on="image_id",
        how="left",
    )
    auc_ctl, std_ctl, acc_ctl, _ = combined_cv(control, extra=["brightness_mean", "brightness_std"])
    print(f"  fractal + brightness controls combined AUC {auc_ctl:.4f} +- {std_ctl:.4f}")
    only = control[["image_id", "label", "brightness_mean", "brightness_std"]].copy()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    X = only[["brightness_mean", "brightness_std"]].to_numpy()
    y = only["label"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    print(f"  brightness controls alone combined AUC {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")
    rows.append(
        {
            "check": "controls",
            "feature": "fractal_plus_brightness",
            "auc_signed": auc_ctl,
            "auc_abs": auc_ctl,
            "auc_std": std_ctl,
            "accuracy_mean": acc_ctl,
            "original_auc": auc_orig,
            "drop": auc_orig - auc_ctl,
        }
    )
    rows.append(
        {
            "check": "controls",
            "feature": "brightness_only",
            "auc_signed": float(np.mean(aucs)),
            "auc_abs": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
        }
    )

    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False)
    print(f"\nsaved {REPORT_PATH.name} in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
