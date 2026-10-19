# Fractal Branch — FREEZE

**Status: FROZEN at v2 (Day 3, 2026-10-19).** No further feature, model or
interface changes without a version bump and a new gate run.

This file is the single reference for "what is the final, shipped state of the
fractal branch". Everything here is verified against the code, not asserted.

---

## 1. Frozen artifact

| Item | Value |
|---|---|
| Model | `model/v2/fractal_clf.joblib` |
| SHA256 | `142744248CFFBA7833E1914DD4AC5E5C4838972E6AB7B3C7925DC08346B6C8BB` |
| Model card | `model/v2/model_card.json` |
| Classifier | `HistGradientBoostingClassifier` (regularized), seed 42 |
| Feature set | candidate **F** — 22 features |
| Binarization | `highpass` (core block also repeats `mean`) |
| Patch / stride | 64 / 32 |
| Aggregation / threshold | mean of patch probabilities / 0.5 |
| v1 (superseded, kept) | `model/v1/fractal_clf.joblib`, SHA256 `AA5776…9BE95` |

## 2. Frozen public interface (do not change)

```python
from ml.fractal import predict_fractal      # callable, re-exported in __init__.py
out = predict_fractal(image)                # image: path | PIL | np.ndarray
out["patch_scores"]   # np.float32, shape (rows, cols), values in [0, 1]
out["fractal_score"]  # float == patch_scores.mean(), in [0, 1]
predict_fractal_json(image)                 # same values, JSON-serializable lists
```

A 256×256 input yields a **7×7** grid. The Backend reads `patch_scores`; the Deep
branch reads `fractal_score`. Both contracts are byte-for-byte identical to v1 —
only the numeric values changed.

## 3. Final metrics (val split, 4,000 images / 196,000 patches)

| Metric | v1 | v2 (frozen) |
|---|---|---|
| Image AUC | 0.6934 | **0.7802** |
| Patch AUC | 0.5573 | 0.6020 |
| Accuracy @ 0.5 | 0.6405 | **0.7050** |
| Precision / Recall / F1 | — | 0.671 / 0.803 / 0.731 |
| Train CV image AUC | 0.6867 | 0.7691 (±0.0090) |
| Warm inference (256×256, CPU) | 62 ms | ~90–143 ms |

Adoption gate: CV beat baseline by +0.0824 (> 9 fold-std, ≥ 0.01 ✔) and inference
< 1 s ✔ → **v2 adopted**.

## 4. Verification performed on the frozen model

- **Tests:** 26/26 green (`python -m pytest ml/fractal`), incl. 8 predict tests,
  7 extra-feature tests, cache parity against the candidate-F pipeline.
- **Smoke test:** `python fractal/smoke_test.py` → `PASS` (7×7 float32, mean
  equality, 22 features, under-budget latency).
- **Heatmap diagnostics** (`heatmap_verification.json`): near-zero
  brightness/contrast correlation (−0.07 / +0.02), 0.15% flat maps.
- **Deletion test:** NOT causally faithful — documented limitation, not a claim.
- **Robustness** (`robustness.csv`): stable to JPEG q90; collapses under blur /
  half-res resize / q50 — documented limitation.

## 5. Environment

`requirements-fractal.txt`, Python 3.12.10, pinned:
numpy 2.5.4 · scipy 1.18.1 · scikit-learn 1.9.1 · pandas 3.0.6 · joblib 1.6.0 ·
pillow 13.0.0 · pyarrow 26.0.0 · matplotlib 3.11.2 · tqdm 4.70.1.

## 6. Deliverables in this branch

- `predict_fractal.py`, `fractal_features.py`, `extra_features.py`, `config.py`,
  `__init__.py` — the frozen pipeline + re-exported callable.
- `model/v2/` — shipped model + card; `model/v1/` — preserved baseline.
- `CHANGELOG.md` — v1→v2 diff + downstream action for the Deep owner.
- `EXPLAINABILITY_SECTION.md`, `VIVA_NOTES.md` — report + defense.
- `val_scores.csv` — re-exported through v2 for fusion retraining.
- `figures/` — 9 report figures.

## 7. Known limitations (stated, not hidden)

1. Heatmap is a diagnostic map, not a proven causal localization (deletion test).
2. Fragile to any loss of high-frequency detail (blur / downscale / strong JPEG).
3. Modest single-patch AUC (0.60); image reliability comes from aggregation.
4. Trained/validated on one dataset (real-vs-fake); cross-generator generalization
   not measured here.

## 8. Handoff actions

- **Deep owner:** retrain logistic fusion on the new `val_scores.csv` (v2 scores).
- **Backend owner:** no change required; `patch_scores` shape/dtype unchanged.
- Feed the branch **full-resolution, high-quality** crops to avoid the robustness
  cliff.
