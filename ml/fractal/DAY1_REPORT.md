# Day 1 Report: Fractal Branch

## 1. What was run

Sample: 500 real and 500 fake images drawn with `random.seed(42)` from the train split of `140k Real and Fake Faces` only. The valid and test splits were never extracted or touched; only entries whose path contains `real-vs-fake/train/` were extracted from the zip (50,000 real and 50,000 fake images present in `C:\Users\bakht\data`). All images are 256x256 RGB JPEGs and give 7 x 7 = 49 patches each, so every method produced 49,000 patch rows over 1000 images.

Every run used `SEED = 42` for the sample draw, the noise in unit tests, and the stratified folds. Three binarization methods were compared: `mean`, `otsu`, and `highpass`. Feature extraction took 64.7 seconds on 8 workers for all three methods combined. The unit tests passed 11/11; no test was weakened. The full pipeline can be rerun with `python fractal/run_all.py`.

Reproducibility note: the `random.choice` sample of file names matched the sample on disk, and the extraction and confound runs finished without errors.

## 2. Results

Patch level AUCs (absolute, with the direction the feature leans), and image level AUCs with Cohen's d, per method:

| method | feature | patch AUC (abs, higher in) | image AUC (abs, higher in) | Cohen's d (fake - real) |
|---|---|---|---|---|
| mean | D | 0.503 (fake) | 0.521 (fake) | +0.09 |
| mean | lac4 | 0.508 (real) | 0.540 (real) | -0.12 |
| mean | lac8 | 0.505 (real) | 0.535 (real) | -0.11 |
| mean | lac16 | 0.503 (real) | 0.528 (real) | -0.09 |
| otsu | D | 0.511 (fake) | 0.541 (fake) | +0.16 |
| otsu | lac4 | 0.512 (real) | 0.556 (real) | -0.12 |
| otsu | lac8 | 0.510 (real) | 0.549 (real) | -0.11 |
| otsu | lac16 | 0.507 (real) | 0.541 (real) | -0.10 |
| highpass | D | 0.516 (fake) | 0.554 (fake) | +0.21 |
| highpass | lac4 | 0.508 (real) | 0.540 (real) | -0.15 |
| highpass | lac8 | 0.509 (real) | 0.544 (real) | -0.14 |
| highpass | lac16 | 0.508 (real) | 0.547 (real) | -0.13 |

Combined image level logistic regression (StandardScaler, 5 fold stratified CV, each image in exactly one fold):

| method | mean AUC | std | mean accuracy | std |
|---|---|---|---|---|
| mean | 0.551 | 0.034 | 0.550 | 0.026 |
| otsu | 0.545 | 0.012 | 0.538 | 0.014 |
| highpass | 0.582 | 0.041 | 0.545 | 0.047 |

Patch level AUCs are essentially 0.50 for every feature and every method. Patch level signal is absent, which is expected given that patch labels are weak (every patch inherits its image label). Image level aggregation lifts the best feature to 0.55, and the combined classifier reaches 0.58.

## 3. Confound findings

Image size and mode: identical across classes (all 256x256 RGB). No confound on that axis.

File size: real mean 28,570 bytes, fake mean 28,143 bytes. AUC 0.538 with the real class slightly larger, Cohen's d -0.11. This is a weak effect, below the 0.65 warning level, so compression is not a class marker by itself.

Re-compression at quality 75, with the highpass features recomputed: the combined image level AUC fell from 0.582 to 0.512, a drop of 0.070. This is the most important finding. The fractal signal mostly does not survive re-encoding, which suggests that a large part of what the highpass features measure is the generator's specific texture as preserved by the original JPEG pipeline rather than a robust structural property of generated faces. After re-compression the branch is statistically at chance.

Brightness and contrast: mean intensity AUC 0.503 (d +0.00), std intensity AUC 0.554 (higher in real, d -0.17). Neither is close to a strong predictor. A model using brightness mean and std alone reaches AUC 0.549. Adding them as controls to the fractal features gives 0.582, unchanged from the fractal-only result, so the fractal features are not merely proxies for brightness or contrast. Their weakness is not explained by a simple photometric confound.

## 4. Chosen default method

`highpass` is the default. It has the highest combined cross validated AUC (0.582 against 0.551 for `mean` and 0.545 for `otsu`), the largest single feature effect (D, d +0.21), and a clear directional story: D is higher in fake images, lacunarity lower. Binarizing the residual of a Gaussian blur keeps local texture and discards low frequency lighting and identity, which is the reason it behaves best here. The choice is recorded in `fractal/default_method.txt` and the features are copied to `fractal/features_sample.parquet`.

## 5. Go/no-go

Using the suggested rubric, the original combined AUC of 0.582 falls in the 0.55 to 0.65 band, which maps to CAUTIOUS GO. Taken at face value the fractal branch supports the deep branch but should not carry accuracy.

The re-compression result argues for a stricter reading. After re-compression the combined AUC is 0.512, which is at chance, and the rubric's own instruction is that the confound checks should come first. The honest conclusion is: **NO-GO on accuracy for now, pending a check on a cleaner setting.** Treat the heatmap as the primary deliverable (localization and explainability) and let the Deep branch carry the classification score; revisit the accuracy claim after testing on images that went through a single matched compression pipeline for both classes. This is a deliberate downgrade from what the raw numbers would have suggested, and it is the conservative and defensible call.

## 6. Known limitations

The dataset contains a single generator (StyleGAN). Fractal features learned here may not transfer to diffusion based or other generators.

All images were pre-resized to 256x256, which suppresses much of the high-frequency micro texture that box counting and lacunarity would otherwise see. This is likely a major reason the patch level AUCs are near chance.

Patch labels are weak: a patch inherits its image label even when the patch is background, hair, or plain skin. Patch level AUCs should not be read as localization ability.

The sample is small (1000 images). The 5 fold standard deviations are 0.03 to 0.05, so differences between methods of a few hundredths of an AUC point are not meaningful.

Only JPEG quality 75 was tested in the re-compression check. Other quality levels could give a different drop.

No threshold, method, or feature was tuned to improve the reported result. Every number above was computed by the scripts in this folder.
