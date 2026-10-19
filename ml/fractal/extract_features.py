from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from fractal.config import FEATURES, PATCH, SEED, STRIDE
from fractal.fractal_features import extract_patches, features_for_patches, load_gray

CACHE_DIR = ROOT / "cache"
MANIFEST = ROOT / "train_manifest.csv"
SKIPPED = CACHE_DIR / "skipped.csv"
DATASET_ROOT = Path(os.environ.get("NYLP_DATASET_ROOT", r"C:\Users\bakht\nonexistent_dataset_root"))
FALLBACK_ROOT = Path(r"C:\Users\bakht\data\real_vs_fake\real-vs-fake")
FALLBACK_ZIP = Path(r"C:\Users\bakht\140k-real-and-fake-faces.zip")
TRAIN_PER_CLASS = 10000
VAL_PER_CLASS = 2000
EVAL_LIMIT = 10000
ALIASES = {
    "train": ("train",),
    "val": ("valid", "val", "validation"),
    "test": ("test",),
}
HELDOUT_NAMES = ("heldout", "held_out", "held-out", "unseen")


def read_method() -> str:
    return (ROOT / "default_method.txt").read_text(encoding="utf-8").split()[0].strip()


def _has_classes(folder: Path) -> bool:
    return (folder / "real").is_dir() and (folder / "fake").is_dir()


def _split_dirs(base: Path) -> dict:
    found = {}
    for split, names in ALIASES.items():
        for name in names:
            candidate = base / name
            if _has_classes(candidate):
                found[split] = candidate
                break
    return found


def locate_splits() -> dict:
    if DATASET_ROOT.is_dir():
        found = _split_dirs(DATASET_ROOT)
        for child in sorted(DATASET_ROOT.iterdir()):
            if child.is_dir() and any(tok in child.name.lower() for tok in HELDOUT_NAMES):
                if _has_classes(child):
                    found[child.name] = child
        if "train" in found and "val" in found:
            return found
    if _has_classes(FALLBACK_ROOT / "train") and _has_classes(FALLBACK_ROOT / "valid"):
        found = {"train": FALLBACK_ROOT / "train", "val": FALLBACK_ROOT / "valid"}
        if _has_classes(FALLBACK_ROOT / "test"):
            found["test"] = FALLBACK_ROOT / "test"
        return found
    raise SystemExit("no dataset found: DATASET_ROOT is absent and the fallback is not extracted")


def _files(folder: Path) -> list:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def _pick(generator: Path, split: str, per_class: int) -> dict:
    out = {}
    for cls in ("real", "fake"):
        files = _files(generator / cls)
        if not files:
            raise SystemExit(f"no images in {generator / cls}")
        random.seed(SEED)
        chosen = files if len(files) <= per_class else random.sample(files, per_class)
        chosen = sorted(chosen)
        out[cls] = [{"path": str(p), "image_id": f"{split}_{cls}_{p.stem}", "label": 1 if cls == "fake" else 0} for p in chosen]
    return out


def build_manifest(splits: dict) -> tuple:
    random.seed(SEED)
    rows = []
    skipped = []
    counts = {}
    for split, folder in splits.items():
        subs = [d for d in sorted(folder.iterdir()) if d.is_dir() and d.name.lower() not in ("real", "fake") and _has_classes(d)]
        generators = [(folder.name, folder)] + [(d.name, d) for d in subs]
        per_class = TRAIN_PER_CLASS if split == "train" else VAL_PER_CLASS
        split_counts = {}
        for gen_name, gen_path in generators:
            picked = _pick(gen_path, split, per_class)
            for cls in ("real", "fake"):
                for item in picked[cls]:
                    item["split"] = split
                    item["generator"] = gen_name
                    rows.append(item)
                split_counts[cls] = split_counts.get(cls, 0) + len(picked[cls])
        counts[split] = split_counts
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["image_id"])
    if len(train) > TRAIN_PER_CLASS * 2:
        keep = []
        for cls in ("real", "fake"):
            pool = [r for r in train if r["label"] == (1 if cls == "fake" else 0)]
            random.seed(SEED)
            keep.extend(random.sample(pool, min(len(pool), TRAIN_PER_CLASS)))
        rows = [r for r in rows if r["split"] != "train"] + keep
    counts = {}
    for row in rows:
        per_class = counts.setdefault(row["split"], {"real": 0, "fake": 0})
        per_class["real" if row["label"] == 0 else "fake"] += 1
    rows.sort(key=lambda r: (r["split"], r["image_id"]))
    for r in rows:
        if not Path(r["path"]).is_file():
            skipped.append({"path": r["path"], "split": r["split"], "reason": "missing before extraction"})
    return rows, counts, skipped


def write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(fields) + "\n")
        for row in rows:
            values = []
            for field in fields:
                text = str(row.get(field, ""))
                values.append('"' + text.replace('"', '""') + '"' if "," in text or '"' in text else text)
            fh.write(",".join(values) + "\n")


def read_manifest() -> list:
    import csv

    rows = []
    with MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["label"] = int(row["label"])
            row.setdefault("generator", "")
            rows.append(row)
    return rows


def _work(item: dict) -> dict:
    method = read_method()
    row = {
        "path": item["path"],
        "image_id": item["image_id"],
        "label": int(item["label"]),
        "split": item["split"],
        "generator": item.get("generator", ""),
    }
    try:
        arr = load_gray(item["path"])
        patches, rows, cols = extract_patches(arr)
        feats = features_for_patches(patches, method=method).astype(np.float32)
        if feats.shape != (rows * cols, len(FEATURES)):
            raise ValueError(f"unexpected feature shape {feats.shape}")
        row.update({"ok": True, "rows": rows, "cols": cols, "feats": feats})
    except Exception as exc:
        row.update({"ok": False, "reason": f"{type(exc).__name__}: {exc}"})
    return row


def run_split(name: str, items: list, method: str, workers: int, force: bool) -> Path:
    from tqdm import tqdm

    out_path = CACHE_DIR / f"features_{name}.parquet"
    if out_path.exists() and not force:
        print(f"[{name}] cache exists, skipping ({out_path.name})")
        return out_path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not items:
        print(f"[{name}] no images, writing empty cache")
        table = pa.table(
            {
                "image_id": pa.array([], type=pa.string()),
                "label": pa.array([], type=pa.int8()),
                "split": pa.array([], type=pa.string()),
                "row": pa.array([], type=pa.int16()),
                "col": pa.array([], type=pa.int16()),
                "D": pa.array([], type=pa.float32()),
                "lac4": pa.array([], type=pa.float32()),
                "lac8": pa.array([], type=pa.float32()),
                "lac16": pa.array([], type=pa.float32()),
            }
        )
        writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        writer.close()
        return out_path
    print(f"[{name}] extracting {len(items)} images with {workers} workers")
    start = time.perf_counter()
    results = []
    skipped = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in tqdm(pool.map(_work, items, chunksize=8), total=len(items), unit="img"):
            if res["ok"]:
                results.append(res)
            else:
                skipped.append({"path": res["path"], "split": name, "reason": res["reason"]})
                results.append(res)
    elapsed = time.perf_counter() - start
    results = [r for r in results if r["ok"]]
    counts = np.array([r["rows"] * r["cols"] for r in results], dtype=np.int64)
    feats = np.concatenate([r["feats"] for r in results], axis=0)
    row_idx = np.concatenate([np.repeat(np.arange(r["rows"], dtype=np.int16), r["cols"]) for r in results])
    col_idx = np.concatenate([np.tile(np.arange(r["cols"], dtype=np.int16), r["rows"]) for r in results])
    labels = np.concatenate([np.full(r["rows"] * r["cols"], r["label"], dtype=np.int8) for r in results])
    image_ids = np.concatenate([np.full(r["rows"] * r["cols"], r["image_id"], dtype=object) for r in results])
    table = pa.table(
        {
            "image_id": pa.array(image_ids.tolist(), type=pa.string()),
            "label": pa.array(labels, type=pa.int8()),
            "split": pa.array([name] * len(labels), type=pa.string()),
            "row": pa.array(row_idx, type=pa.int16()),
            "col": pa.array(col_idx, type=pa.int16()),
            FEATURES[0]: pa.array(feats[:, 0], type=pa.float32()),
            FEATURES[1]: pa.array(feats[:, 1], type=pa.float32()),
            FEATURES[2]: pa.array(feats[:, 2], type=pa.float32()),
            FEATURES[3]: pa.array(feats[:, 3], type=pa.float32()),
        }
    )
    writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    writer.close()

    skipped_path = CACHE_DIR / f"skipped_{name}.csv"
    write_csv(skipped_path, skipped, ["path", "split", "reason"])
    all_skip = []
    if SKIPPED.exists():
        import csv

        with SKIPPED.open("r", encoding="utf-8", newline="") as fh:
            all_skip = [r for r in csv.DictReader(fh) if r.get("split") != name]
    all_skip.extend(skipped)
    write_csv(SKIPPED, all_skip, ["path", "split", "reason"])
    print(f"[{name}] saved {len(table)} rows in {elapsed:.1f}s, skipped {len(skipped)}")
    return out_path


def benchmark(items: list, method: str, workers: int, n: int, total: int) -> float:
    sample = items[:n]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_work, sample, chunksize=8))
    elapsed = time.perf_counter() - start
    rate = elapsed / max(1, len(sample))
    print(f"benchmark: {len(sample)} images in {elapsed:.2f}s, {rate * 1000:.1f} ms/image, estimated total for {total} images: {rate * total:.0f}s")
    return rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    method = read_method()
    workers = args.workers or max(1, min(os.cpu_count() or 1, 8))
    if MANIFEST.exists() and not args.force:
        print(f"reusing manifest {MANIFEST.name}")
    else:
        random.seed(SEED)
        splits = locate_splits()
        rows, counts, problems = build_manifest(splits)
        write_csv(MANIFEST, rows, ["path", "image_id", "label", "split", "generator"])
        print("splits:", {k: str(v) for k, v in splits.items()})
        for split, per_class in sorted(counts.items()):
            print(f"{split}: real={per_class.get('real', 0)} fake={per_class.get('fake', 0)}")
        for split in sorted({r["split"] for r in rows}):
            gens = sorted({r["generator"] for r in rows if r["split"] == split})
            print(f"{split} generator folders: {gens}")
        if problems:
            write_csv(SKIPPED, problems, ["path", "split", "reason"])
    rows = read_manifest()
    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
    ordered = ["train", "val"] + [s for s in sorted(by_split) if s not in ("train", "val")]
    pool_items = [r for s in ordered for r in by_split.get(s, [])]
    if args.limit:
        train_all = [r for r in rows if r["split"] == "train"]
        benchmark(train_all[: args.limit], method, workers, min(args.benchmark, len(train_all)), len(train_all))
        return 0
    if args.benchmark:
        train_all = [r for r in rows if r["split"] == "train"]
        benchmark(train_all[: args.benchmark], method, workers, min(args.benchmark, len(train_all)), len(pool_items))
    for split in ordered:
        run_split(split, by_split.get(split, []), method, workers, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
