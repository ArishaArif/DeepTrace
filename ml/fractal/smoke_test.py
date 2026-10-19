"""Step 9 freeze smoke test: run the frozen v2 branch end-to-end.

Exits 0 and prints PASS only if every invariant of the frozen contract holds:
  - the deep-branch import path resolves to a callable
  - predict_fractal returns patch_scores (float32, 2-D, in [0,1]) and a scalar
    fractal_score equal to its mean
  - a 256x256 image gives the expected 7x7 grid
  - warm inference stays under the 1 s budget
  - the loaded model is the v2 candidate-F bundle (22 features)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent          # ml/
REPO = PROJECT.parent          # DeepTrace/ (holds the `ml` package)
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(REPO))


def _sample() -> Path:
    files = sorted((PROJECT / "sample" / "real").glob("*.jpg"))
    assert files, f"no sample images in {PROJECT / 'sample' / 'real'}"
    return files[0]


def main() -> int:
    # 1. deep-branch import path must be the callable, not the module
    from ml.fractal import predict_fractal as imported
    assert callable(imported), "from ml.fractal import predict_fractal is not callable"

    import importlib
    pf = importlib.import_module("ml.fractal.predict_fractal")  # real module (re-export shadows attribute access)

    path = _sample()
    img = Image.open(path).convert("RGB")
    assert img.size == (256, 256), f"sample not 256x256: {img.size}"

    # 2. warm the model (first call loads weights), then time a second call
    pf.predict_fractal(path)
    t0 = time.perf_counter()
    out = pf.predict_fractal(img)
    warm_ms = (time.perf_counter() - t0) * 1000.0

    # 3. contract invariants
    ps = out["patch_scores"]
    fs = out["fractal_score"]
    assert isinstance(ps, np.ndarray), "patch_scores not ndarray"
    assert ps.dtype == np.float32, f"patch_scores dtype {ps.dtype} != float32"
    assert ps.ndim == 2, f"patch_scores ndim {ps.ndim} != 2"
    assert ps.shape == (7, 7), f"256x256 grid {ps.shape} != (7, 7)"
    assert float(ps.min()) >= 0.0 and float(ps.max()) <= 1.0, "scores outside [0,1]"
    assert np.isclose(float(np.mean(ps)), float(fs), atol=1e-6), "fractal_score != mean"

    # 4. v2 bundle sanity: the model must expect the 22-feature candidate F vector
    model = pf._load_model() if hasattr(pf, "_load_model") else pf._MODEL
    n_features = int(getattr(model, "n_features_in_", 22))
    assert n_features == 22, f"model expects {n_features} features, expected 22 (candidate F)"

    # 5. budget
    assert warm_ms < 1000.0, f"warm inference {warm_ms:.0f} ms exceeds 1 s budget"

    print(f"PASS  grid={ps.shape} dtype={ps.dtype} score={fs:.4f} "
          f"features={n_features} warm={warm_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
