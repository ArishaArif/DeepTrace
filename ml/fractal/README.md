# Fractal Branch

Explainability engine for the hybrid deepfake detector. It binarizes 64x64 grayscale patches, measures their fractal dimension and lacunarity, and turns those into a patch level fake probability map.

## Files

| file | purpose |
|---|---|
| `config.py` | Frozen shared contract: patch size, stride, minimum size, box sizes, seeds, labels |
| `fractal_features.py` | Core library: image loading, patch extraction, binarization, box counting dimension, lacunarity |
| `test_features.py` | Unit tests with PASS/FAIL output; runnable with `python` or `pytest` |
| `setup_data.py` | Extracts only the train split from the dataset zip, builds the 500+500 sample, updates `.gitignore` |
| `extract_sample.py` | Batch feature extraction for all three binarization methods into parquet files |
| `sanity_test.py` | Patch and image level AUCs, Cohen's d, combined cross validated classifier, picks the default method |
| `confound_checks.py` | Size/mode, file size, re-compression, brightness and contrast checks |
| `extract_features.py` | Day 2: locate splits (with 140k fallback), build `train_manifest.csv`, cache per-split patch features to `cache/features_<split>.parquet` (resumable, `--force` to redo) |
| `train_classifier.py` | Day 2: GroupKFold model selection on train, single val evaluation, writes `model/fractal_clf.joblib` and `model/model_card.json` |
| `predict_fractal.py` | `predict_fractal(image)` and `predict_fractal_json(image)`: patch score map plus fractal_score from the trained model |
| `test_predict.py` | Day 2 acceptance tests: shapes, input parity, determinism, JSON, speed, train-vs-inference cache agreement |
| `export_scores.py` | Day 2: writes `val_scores.csv` (and `test_scores.csv` / held-out files when present) for the Deep branch |
| `heatmap_check.py` | Day 2: `heatmap_preview.png`, `position_bias.png`, `heatmap_diagnostics.json` |
| `run_all.py` | Runs every Day 1 step in order |
| `DAY1_REPORT.md` | Findings, AUC table, confound results, go/no-go |
| `DAY2_REPORT.md` | Trained model, CV and val metrics, heatmap diagnostics, interface status |
| `sanity_plot.png` | Real vs fake histograms of the four features for the best method |
| `sanity_results.csv` | All sanity numbers |
| `confound_results.csv` | All confound numbers |
| `default_method.txt` | Name of the selected binarization method |

## Rerun everything, in order

Day 1 (sample pipeline, already done):

```
python fractal/setup_data.py
python fractal/test_features.py
python fractal/extract_sample.py
python fractal/sanity_test.py
python fractal/confound_checks.py
```

Day 2 (full data, cache through report):

```
python fractal/extract_features.py
python fractal/train_classifier.py
python fractal/test_predict.py
python fractal/predict_fractal.py
python fractal/export_scores.py
python fractal/heatmap_check.py
```

`extract_features.py` is resumable: existing `cache/features_<split>.parquet` files are skipped unless `--force` is passed. Expected runtimes on this machine: feature extraction a few minutes per split when warm-skipped, training (4 candidates x 5 folds) about 10 minutes, tests under a minute, score export about 7 minutes for 4,000 val images (97 ms/image), heatmap check about 15 minutes.

## Interface

```python
from fractal.predict_fractal import predict_fractal, predict_fractal_json

out = predict_fractal("path/to/image.jpg")   # also accepts PIL images and numpy arrays
out["patch_scores"]   # np.ndarray float32, shape (rows, cols), values in [0, 1]
out["fractal_score"]  # float, exactly the mean of patch_scores

js = predict_fractal_json("path/to/image.jpg")  # same content, patch_scores as nested lists, JSON serializable
```

Trained model at `fractal/model/fractal_clf.joblib` (HistGradientBoosting pipeline, highpass method, seed 42). Validation scores for the fusion model are in `fractal/val_scores.csv` (val images only). Decision threshold is 0.5 and is owned by the Deep branch. See `DAY2_REPORT.md` for metrics and heatmap caveats.
