from __future__ import annotations

import random
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = Path(r"C:\Users\bakht\140k-real-and-fake-faces.zip")
DATA_ROOT = Path(r"C:\Users\bakht\data")
TRAIN_DIR = DATA_ROOT / "real_vs_fake" / "real-vs-fake" / "train"
SAMPLE_DIR = ROOT / "sample"
TRAIN_MARKER = "real-vs-fake/train/"
SEED = 42
SAMPLE_PER_CLASS = 500


def show_layout():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
    print("zip entries:", len(names))
    for name in names[:10]:
        print("  ", name)


def _count_jpgs(folder):
    return len(list(folder.glob("*.jpg")))


def train_ready():
    real = TRAIN_DIR / "real"
    fake = TRAIN_DIR / "fake"
    return (
        real.is_dir()
        and fake.is_dir()
        and _count_jpgs(real) > 0
        and _count_jpgs(fake) > 0
    )


def extract_train():
    if train_ready():
        print("train split already extracted, reusing", TRAIN_DIR)
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [m for m in zf.infolist() if TRAIN_MARKER in m.filename and m.filename.endswith(".jpg")]
        print("train jpg entries:", len(members))
        zf.extractall(DATA_ROOT, members=members)
    print("extracted train split to", DATA_ROOT)


def verify_counts():
    real = TRAIN_DIR / "real"
    fake = TRAIN_DIR / "fake"
    n_real = _count_jpgs(real)
    n_fake = _count_jpgs(fake)
    print("train/real:", n_real)
    print("train/fake:", n_fake)
    if n_real == 0 or n_fake == 0:
        show_layout()
        raise SystemExit("extraction produced an empty class folder")
    return n_real, n_fake


def build_sample():
    real_dst = SAMPLE_DIR / "real"
    fake_dst = SAMPLE_DIR / "fake"
    if (
        real_dst.is_dir()
        and fake_dst.is_dir()
        and _count_jpgs(real_dst) == SAMPLE_PER_CLASS
        and _count_jpgs(fake_dst) == SAMPLE_PER_CLASS
    ):
        print("sample already built, reusing", SAMPLE_DIR)
        return
    random.seed(SEED)
    for cls, dst in (("real", real_dst), ("fake", fake_dst)):
        src = TRAIN_DIR / cls
        files = sorted(src.glob("*.jpg"))
        if len(files) < SAMPLE_PER_CLASS:
            raise SystemExit(f"not enough {cls} images in {src}: {len(files)}")
        picked = random.sample(files, SAMPLE_PER_CLASS)
        dst.mkdir(parents=True, exist_ok=True)
        for f in picked:
            shutil.copy2(f, dst / f.name)
    print("sample real:", _count_jpgs(real_dst))
    print("sample fake:", _count_jpgs(fake_dst))


def update_gitignore():
    path = ROOT / ".gitignore"
    wanted = ["sample/", "data/", "*.zip", "fractal/tmp_recompressed/"]
    existing = []
    if path.exists():
        existing = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    present = {line.strip() for line in existing}
    to_add = [w for w in wanted if w not in present]
    if not to_add:
        print(".gitignore already up to date")
        return
    lines = existing + ([] if existing and existing[-1] == "" else [""])
    lines.extend([""] if not existing or existing[-1] != "" else [])
    lines.extend(to_add)
    text = "\n".join(line for line in lines if line is not None).rstrip("\n") + "\n"
    path.write_text(text, encoding="utf-8")
    print(".gitignore updated with:", ", ".join(to_add))


def main():
    show_layout()
    extract_train()
    verify_counts()
    build_sample()
    update_gitignore()
    print("data setup complete")


if __name__ == "__main__":
    sys.exit(main())
