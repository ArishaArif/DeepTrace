from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.predict_fractal import predict_fractal

MANIFEST = ROOT / "train_manifest.csv"


def manifest_rows() -> list:
    with MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def score_split(rows: list, out_path: Path) -> pd.DataFrame:
    from tqdm import tqdm

    records = []
    failures = []
    start = time.perf_counter()
    for row in tqdm(rows, desc=out_path.stem, unit="img"):
        try:
            out = predict_fractal(row["path"])
            records.append(
                {
                    "image_id": row["image_id"],
                    "label": int(row["label"]),
                    "fractal_score": float(out["fractal_score"]),
                    "generator": row.get("generator", ""),
                }
            )
        except Exception as exc:
            failures.append({"image_id": row["image_id"], "path": row["path"], "reason": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(records)
    elapsed = time.perf_counter() - start
    if failures:
        fail_path = ROOT / "cache" / f"score_failures_{out_path.stem}.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"skipped {len(failures)} unreadable images, see {fail_path.name}")
    if out_path.name.endswith("val_scores.csv"):
        frame = frame[["image_id", "label", "fractal_score"]]
    else:
        frame = frame[["image_id", "label", "fractal_score", "generator"]]
    frame.to_csv(out_path, index=False)
    auc = float(roc_auc_score(frame["label"], frame["fractal_score"])) if frame["label"].nunique() > 1 else float("nan")
    acc = float(((frame["fractal_score"] >= 0.5).astype(int) == frame["label"]).mean())
    print(f"{out_path.name}: {len(frame)} rows, AUC {auc:.4f}, accuracy@0.5 {acc:.4f}, {elapsed / max(1, len(frame)) * 1000:.1f} ms/image")
    return frame


def main() -> int:
    rows = manifest_rows()
    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
    if "val" not in by_split:
        raise SystemExit("no val split in the manifest")
    targets = [("val", ROOT / "val_scores.csv"), ("test", ROOT / "test_scores.csv")]
    targets += [(s, ROOT / f"{s}_scores.csv") for s in sorted(by_split) if s not in ("train", "val", "test")]
    summary = []
    for split, out_path in targets:
        split_rows = by_split.get(split, [])
        if not split_rows:
            print(f"{split}: no images, file not written")
            continue
        frame = score_split(split_rows, out_path)
        summary.append({"split": split, "rows": len(frame), "file": out_path.name})
    train_rows = by_split.get("train", [])
    print(f"train images available in manifest: {len(train_rows)} (never scored here; val_scores.csv holds val only)")
    print("wrote:", ", ".join(item["file"] for item in summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
