# Fractal Branch CHANGELOG

## v2 (Day 3, 2026-10-19)

**The Deep branch owner must retrain the fusion on the new `val_scores.csv`.**
The fractal model changed, so every fractal score changed. The old fusion weights
were fit against v1 scores and are no longer valid.

### What changed

- Adopted a new patch classifier, **v2**, after the Day 3 adoption gate passed on
  train cross validation and a single val confirmation.
- Feature set expanded from the 4 core features `[D, lac4, lac8, lac16]` to the
  22-feature **candidate F** vector: the core four, two extra lacunarity scales
  (`lac2`, `lac32`), the box counting regression fit quality (`r2`), local
  box counting slopes (`slope_small`, `slope_large`, `slope_diff`), eight
  multifractal dimensions (`D0/D1/D2` and the `D0 minus D2` width for both the
  gradient and the intensity measure), and the mean binarization core
  (`D_mean`, `lac4_mean`, `lac8_mean`, `lac16_mean`).
- Classifier family unchanged: `HistGradientBoostingClassifier`, same
  regularized hyperparameters, seed 42, threshold 0.5, mean aggregation.
- Binarization default unchanged: `highpass`.

### Metrics, v1 to v2 (validation split, 4,000 images, 196,000 patches)

| metric | v1 | v2 |
|---|---|---|
| image AUC | 0.6934 | **0.7802** |
| patch AUC | 0.5573 | 0.6020 |
| image accuracy @ 0.5 | 0.6405 | 0.7050 |
| warm inference (256x256, CPU) | 62 ms | 143 ms |

Train cross validated image AUC (GroupKFold 5, train only): v1 0.6867, v2 0.7691.
The improvement over baseline is +0.0824, far beyond the fold standard deviation
of 0.0090, so the gate's first condition (beat baseline by at least 0.01 and by
more than one standard deviation) is met. The second condition (val not worse than
v1 by more than 0.002, and inference under one second) is also met.

### Interface, unchanged

`predict_fractal(image)` and `predict_fractal_json(image)` keep the exact frozen
signature and return format. `patch_scores` is still a float32 `(rows, cols)`
array in `[0, 1]` and `fractal_score` is still its mean. Only the values changed.

### Files

- v2 model: `model/v2/fractal_clf.joblib` (SHA256 `142744248CFFBA7833E1914DD4AC5E5C4838972E6AB7B3C7925DC08346B6C8BB`), card `model/v2/model_card.json`.
- v1 preserved unchanged: `model/v1/fractal_clf.joblib` (SHA256 `AA577699643406ABDA5E429F83105811D30CE472FE074C66589D4FC97F98BE95`), card `model/v1/model_card.json`.
- Re-exported through v2 `predict_fractal`: `val_scores.csv` (and `test_scores.csv` / held-out files when those folders exist).

### Action for the Deep branch owner

1. Retrain the logistic fusion on the new `fractal/val_scores.csv`.
2. The Backend needs no change: it reads `patch_scores` exactly as before.

## v1 (Day 2, 2026-10-19)

First trained patch classifier: `HistGradientBoostingClassifier` on the 4 core
fractal features with `highpass` binarization. Val image AUC 0.6934, accuracy
0.6405. Superseded by v2.
