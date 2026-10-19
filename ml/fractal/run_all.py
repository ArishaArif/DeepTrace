from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("data setup", "setup_data.py"),
    ("unit tests", "test_features.py"),
    ("feature extraction", "extract_sample.py"),
    ("sanity test", "sanity_test.py"),
    ("confound checks", "confound_checks.py"),
    ("predict stub test", "predict_fractal.py"),
]


def main():
    for name, script in STEPS:
        print(f"\n=== {name}: {script} ===")
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT.parent))
        if result.returncode != 0:
            print(f"step failed: {script} (exit {result.returncode})")
            return result.returncode
    print("\nall steps completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
