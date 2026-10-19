# Day 2 Report: Fractal Branch

## 1. What was trained

DATASET_ROOT was not available (placeholder path), so the fallback dataset was used: `140k Real and Fake Faces`, zip at `C:\Users\bakht\140k-real-and-fake-faces.zip`, extracted under `C:\Users\bakht\data`. Only the train and valid splits were extracted. The test split was not extracted and never touched, per the rules. There is no held-out generator folder in this dataset (single generator, StyleGAN).

Subsampled with `random.seed(42)` and recorded in `fractal/train_manifest.csv`: 10,000 train images per class (20,000 total, 980,000 patches) and 2,000 valid images per class (4,000 total, 196,000 patches). No image was skipped during extraction (`cache/skipped_train.csv` and `skipped_val.csv` contain only headers). All images are 256x256, so every grid is 7x7.

Binarization method: `highpass`, the frozen Day 1 default. Features per patch: `[D, lac4, lac8, lac16]`, extracted with the cached Day 1 library (`fractal_features.py`, unchanged). Patch labels are weak: every patch inherits its image's label. This is a real limitation and is reflected in the near-chance patch-level metrics.

Selected model: `hgb_regularized`, a `HistGradientBoostingClassifier` (learning_rate 0.05, max_depth 3, max_iter 200, min_samples_leaf 200, l2_regularization 1.0, early_stopping on with validation_fraction 0.1, random_state 42), saved as a full joblib bundle with the binarization method at `fractal/model/fractal_clf.joblib`. Model card: `fractal/model/model_card.json`. Decision threshold left at 0.5; no threshold tuning was done on this side.

## 2. Cross validation and the single val evaluation

Model selection used GroupKFold with 5 folds on the train split only, groups = image_id, so no patch of an image ever appears in both the fit and evaluation folds. Selection metric was image-level AUC (mean patch probability per image); all values from `fractal/model/cv_results.csv`:

| candidate | CV image AUC | std | CV patch AUC |
|---|---|---|---|
| logistic regression C=0.1 | 0.5382 | 0.0086 | 0.5119 |
| logistic regression C=1.0 | 0.5384 | 0.0089 | 0.5119 |
| hgb_regularized | 0.6867 | 0.0073 | 0.5571 |
| hgb_shallow | 0.6671 | 0.0084 | 0.5505 |

The gradient boosting models beat logistic regression by roughly 0.15 image AUC, which is far outside the fold standard deviations. The gain is nonlinear interactions between D and the three lacunarity scales; on Day 1 a linear combination at image level reached only 0.582 on a 1,000-image sample.

Evaluation once on the val split (4,000 images, 196,000 patches), threshold 0.5, aggregation = mean of patch probabilities:

- patch-level AUC: 0.5573
- image-level AUC: 0.6934
- image-level accuracy: 0.6405, precision 0.6187, recall 0.7325, F1 0.6708
- confusion matrix (rows = true real/fake): [[1097, 903], [535, 1465]]
- calibration: mean predicted probability 0.4950 for real images, 0.5046 for fake images. The probabilities are heavily compressed around 0.5; the ranking works (AUC 0.69) but the raw probabilities are not calibrated. The Deep branch's logistic fusion should absorb this, since it refits its own scaling.
- aggregation diagnostic (not a contract change): mean 0.6934, top-20% mean 0.6260, max 0.5933. The contract's mean is the best aggregator; hot-spot style aggregation is worse, consistent with the weak patch-level signal.

The contract (mean of patch scores) is unchanged and was not reselected based on the diagnostic.

## 3. Test and held-out generator results

None available. The fallback dataset's test split was deliberately never extracted, and the dataset has no held-out generator folder. Test and held-out numbers will only exist when the team's own DATASET_ROOT with a held-out generator becomes available; `export_scores.py` already writes `test_scores.csv` and `<generator>_scores.csv` automatically when those folders exist. This is a reporting gap, not a tuning decision: nothing outside the train split was used for any selection.

Stated plainly: Day 1's go/no-go was NO-GO on accuracy because the fractal signal mostly did not survive JPEG re-compression (image AUC 0.582 to 0.512). That caveat still stands for today's numbers. The val image AUC of 0.6934 was measured on images with the same compression pipeline as training; part of the boost over logistic regression may be the generator's compression fingerprint rather than fractal geometry. A re-compression check on this model is the first thing to run on Day 3.

## 4. Heatmap diagnostics

From `heatmap_check.py`, saved as `heatmap_preview.png`, `position_bias.png`, `heatmap_diagnostics.json` (all 4,000 val images scored through `predict_fractal`).

Position bias: mean patch score minus center patch score is -0.021 for real and -0.015 for fake, i.e. border patches score slightly lower than central ones for both classes. Both mean maps share the same hottest position at grid cell (2, 4), near the center of the face region. The heatmap is not dominated by position; some of the hot region is plausibly facial structure.

Proxy check: Pearson r between patch score and patch mean brightness is 0.093, and with patch contrast (std) 0.048, over 196,000 patches. Both weak; no brightness or contrast warning triggered.

Border hotspots: 41.9% of fake val images have their single hottest patch on the grid border, slightly below the 49.0% expected if hotspots were uniform over the 7x7 grid. Hotspots are not border-seeking.

Caveat for the Backend: the dynamic range is small (image fractal_scores span 0.330 to 0.535 over the val set with an interquartile range of 0.493 to 0.510; within an image patches typically span about 0.40 to 0.56), so a naive [0, 1] colormap will look almost uniform. Stretching the colormap to the per-image min/max (display only) would make the heatmap readable without touching the contract values.

## 5. Interface status

- Model: `fractal/model/fractal_clf.joblib` (pipeline + method + feature order). Card: `fractal/model/model_card.json`.
- Scores file for the Deep branch: `fractal/val_scores.csv`, 4,000 rows, columns `image_id,label,fractal_score`, val only, no train rows. AUC of the file itself: 0.6932, accuracy@0.5 0.6390.
- Import line for Deep and Backend:

```python
from fractal.predict_fractal import predict_fractal, predict_fractal_json
```

`predict_fractal(image)` accepts a path, PIL image (any mode) or numpy array (HxW or HxWx3 uint8) and returns `{"patch_scores": float32 (rows, cols) in [0, 1], "fractal_score": float = mean of patch_scores}`. `predict_fractal_json(image)` returns the same with `patch_scores` as a nested list; it is JSON serializable.
- Speed: 62 ms warm, 95 ms cold (model load 36 ms) per 256x256 image on CPU, well under the 1 s target; no vectorization of the frozen feature code was needed. Deterministic: identical inputs give identical outputs (verified).
- Tests: `python fractal/test_predict.py` passes 8/8, including the check that `predict_fractal` agrees with the cached training pipeline on 20 val images to 3e-8 (no train vs inference mismatch).

## 6. Known limitations

Weak patch labels dominate everything: a patch of background or plain skin carries the image's fake label. Patch-level AUC is 0.557, close to chance; the image-level signal is aggregation of many weakly informative patches, not localization. The heatmap should be read as "where the texture statistics deviate", not "the model's evidence".

Single generator (StyleGAN) and all images pre-resized to 256x256, which suppresses the micro-texture box counting and lacunarity would otherwise see. Day 1's confound findings carry over unchanged: the strongest one is that the sample-level signal dropped to chance after JPEG re-compression at quality 75; it has not yet been re-tested on today's boosted model. Predicted probabilities are compressed around 0.5 and poorly calibrated. No test or held-out-generator numbers exist yet. The 0.5 threshold was fixed and the accuracy figures depend on it being tuned by the Deep branch on validation data.

## 7. Recommended Day 3 improvements, ranked by expected value

1. Re-compression robustness: retrain with compressed-image augmentation (random JPEG quality) and re-run the quality-75 check on the new model. This addresses the one finding that made Day 1 a NO-GO and decides whether the branch may claim any accuracy at all.
2. Add more lacunarity scales and the multifractal spectrum (f(alpha)) to the patch vector. The jump from linear 0.538 to boosted 0.687 shows the 4 current features carry unconsumed nonlinear structure; richer texture descriptions give boosting more to use.
3. Position-aware de-biasing: either add (row, col) as training features and zero them at inference, or train on face-aligned crops only. Even though position bias is small today, background patches with fake labels add noise that this removes.
4. Run everything through the team's DATASET_ROOT with a held-out generator as soon as it exists; transfer to an unseen generator is the number that actually matters for the joint score.
5. Calibration (isotonic on train predictions) so fractal_score becomes usable as a probability and fuses more cleanly in the logistic layer.
