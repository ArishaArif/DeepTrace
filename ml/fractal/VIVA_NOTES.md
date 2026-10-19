# Viva Notes — Fractal Branch

Defense-ready answers for the fractal (texture) branch. Numbers are all from the
frozen **v2** model on the **val** split unless stated otherwise. Keep this open
during questioning.

---

## 1. The branch in one breath

"A second, independent detector alongside the deep branch. It classifies an image
as real or generated from **fractal texture statistics only** — box-counting
dimension, lacunarity, local multiscale slopes and a multifractal spectrum —
computed on 64×64 patches, fed to a regularized gradient-boosted tree, averaged to
an image score in [0,1] with a patch grid for the heatmap. It is explainable by
construction: every input feature has a physical meaning."

---

## 2. Anticipated questions and answers

### Method

**Q: Why fractals for fake detection?**
Generated images reproduce low-level statistics imperfectly — GAN/diffusion
textures have subtly wrong multiscale structure. Box-counting and lacunarity are
exactly the statistics that capture that, and they are cheap and interpretable.

**Q: Walk me through one patch → one number.**
Grayscale → extract 64×64 patches (stride 32) → high-pass binarize the patch →
count boxes at sizes {2,4,8,16,32} → slope of log(count) vs log(1/size) is the
box-counting dimension `D`; lacunarity measures gappiness at each window size;
multifractal dimensions `D0/D1/D2` measure how the *mass distribution* scales. 22
such numbers per patch go to the classifier → patch probability → mean over the
grid = `fractal_score`.

**Q: Why 'highpass' binarization?**
Chosen on Day 1/2 by ablation over {mean, otsu, highpass}. High-pass keeps only
local texture detail and removes global brightness, which is why the heatmap is
uncorrelated with exposure. We also re-run the core features under 'mean'
binarization inside the feature vector so we are not hostage to one threshold.

**Q: What is the threshold and aggregation?**
0.5 on the probability; image score is the unweighted mean of patch
probabilities. Both are recorded in `model/v2/model_card.json`.

### Results

**Q: How good is it?**
Val image AUC **0.780**, accuracy **0.705** on 4,000 held-out images. Patch AUC
is only 0.60 — individual patches are weak, averaging 49 of them makes the image
decision reliable.

**Q: How much did Day 3 improve things, and was it worth it?**
v1 (4 features) → v2 (22 features): image AUC 0.693 → 0.780, accuracy 0.640 →
0.705. Gained +0.082 on train cross-validation, ~9× the fold std (0.009), so it
clears the adoption gate. Cost: inference 62 → 143 ms, still far under the 1 s
budget.

**Q: What is the adoption gate?**
Two conditions, both required: (1) candidate beats baseline CV AUC by ≥ 0.01 and
by > 1 fold-std; (2) val AUC not worse than v1 by > 0.002 and inference < 1 s.
Candidate F (the union of every group that helped individually) passed both.

**Q: How did you choose the feature set?**
Ablation. Built candidates B (extra lacunarity + R²), C (local slopes), D
(multifractal), E (multi-representation). Each beat baseline on CV, so per plan we
also built F = union, tested it, and F won (CV 0.769). No test-set peeking;
decisions were made on train CV with one val confirmation.

### Explainability & honesty (the likely hard questions)

**Q: Is the heatmap a valid explanation?**
Partly. It shows where the fractal statistic is high and we verified it is not a
brightness/contrast artifact (correlations −0.07 and +0.02). But our **deletion
test says it is NOT causally localized**: blurring the hottest patches drops the
score exactly like blurring random ones (diff ≈ 0, CI covers zero). Because the
image score is a *mean over the grid*, no small set of patches owns the decision.
We report this as a limitation, not a success.

**Q: Then what is the explanation?**
The **feature vector** is the explanation — it is interpretable by construction.
The heatmap is a diagnostic map of the statistic, not a "this pixel is fake"
attribution. We never claim pixel-level localization.

**Q: Biggest weakness?**
Robustness. Anything that removes fine detail hurts it: half-resolution rescaling
and blur drop accuracy to chance (~0.50); aggressive JPEG (q50) too. Mild JPEG q90
is fine (AUC 0.77). So the branch must receive a **full-resolution, high-quality
crop** — if the pipeline downscales faces first, this branch silently stops
working. Stated plainly in the report.

**Q: Does it just detect low resolution / JPEG artifacts / upscale traces?**
Confound checks argue no: the heatmap is near-uncorrelated with brightness and
contrast, and the features are scale-invariant statistics. It is genuinely reading
multiscale texture — which is also why destroying that texture (blur/resize)
kills it.

### Integration

**Q: What does the Deep/Backend branch need from you?**
`from ml.fractal import predict_fractal` → returns a dict with `patch_scores`
(float32 (rows,cols) for the heatmap) and `fractal_score` (float, mean). Fixed a
real bug: the package `__init__` was returning the *module*, not the function, so
the deep branch's call would have crashed with a TypeError. Re-exported the
callable; interface unchanged since v1.

**Q: You changed the model — what breaks downstream?**
The fusion. v2 re-scored every image, so the old fusion weights fit on v1
`val_scores.csv` are invalid. We re-exported `val_scores.csv` through v2 (AUC
0.780) and flagged in the CHANGELOG that the Deep owner must **retrain the
fusion**. Backend needs no change (reads `patch_scores` the same way).

**Q: Can they load just the .joblib?**
No. The pickle holds weights + a `candidate="F"` tag; the 22-feature computation
lives in Python (`extra_features.py`, `fractal_features.py`, `config.py`). Ship
the whole `fractal/` package or they get a 4-vs-22 feature crash.

---

## 3. Numbers cheat-sheet (v2, val 4000 imgs)

| Metric | Value |
|---|---|
| Image AUC | 0.780 |
| Patch AUC | 0.602 |
| Accuracy @0.5 | 0.705 |
| Precision / Recall / F1 | 0.671 / 0.803 / 0.731 |
| Confusion (real,fake) | [[1214,786],[394,1606]] |
| Train CV image AUC | 0.769 (±0.009 across folds) |
| Warm inference (256×256 CPU) | 143 ms |
| Features / model | 22 (candidate F) / HistGradientBoosting |

## 4. Files to point at if asked to "show me"

- `model/v2/model_card.json` — every hyperparameter and metric
- `CHANGELOG.md` — v1→v2 diff, hashes, downstream action
- `EXPLAINABILITY_SECTION.md` — features, verification, honest limitations
- `heatmap_verification.json` — quantitative checks + deletion result
- `robustness.csv` — corruption sweep
- `figures/` — box-counting, lacunarity, feature distributions, pipeline,
  heatmap contact sheet, deletion curve, robustness
- `test_predict.py`, `test_extra_features.py` — 8 + 7 tests, all green
