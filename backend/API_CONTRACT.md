# API Contract (frozen after Day 1)

Base URL: `http://<backend-host>:8000` (local: `http://localhost:8000`)
CORS is enabled for all origins. Interactive docs: `/docs`.
Both analyze endpoints take `multipart/form-data` with one field named `file`.

## POST /analyze/image
Allowed: .jpg .jpeg .png .webp, max 10 MB

```json
{
  "verdict": "fake",
  "score": 0.73,
  "patch_scores": [[0.2, 0.8, 0.9], [0.1, 0.6, 0.7]],
  "heatmap_png": "<base64 PNG, no data: prefix>"
}
```

- `verdict`: "fake" if score >= server threshold (default 0.5, tuned on the validation set on Day 3), else "real". Lowercase.
- `score`: 0 to 1, higher = more likely fake (output of the fusion model)
- `patch_scores`: rows x cols matrix of floats 0..1 from the fractal branch. 64x64 patches, stride 32. Images smaller than 128x128 are resized up before patching.
  rows = (H - 64) // 32 + 1, cols = (W - 64) // 32 + 1, where H, W are the analysed image dimensions (after any upscaling to 128 minimum).
- `heatmap_png`: base64 PNG, transparent red overlay, same pixel size as the uploaded image. Low score = transparent, high = solid red.
  Frontend: `<img src={"data:image/png;base64," + heatmap_png}>` stacked over the original image; opacity slider = CSS opacity.
- `score` comes from the fusion of both branches while the heatmap comes only from the fractal branch, so the heatmap average will not equal `score`.
- UI wording: "regions contributing to the fake score" (not "manipulated regions").

## POST /analyze/video
Allowed: .mp4 .mov .avi .mkv, max 50 MB

```json
{
  "verdict": "real",
  "score": 0.31,
  "frames": [{"t": 0.0, "score": 0.25}, {"t": 1.0, "score": 0.37}]
}
```

- `t`: seconds (float) from the start of the video. `score`: mean of frame scores. `verdict`: same threshold rule as image.
- `frames` length varies (frames are sampled, only frames with a detected face are kept).

## Errors
`{"detail": "message"}` with status:
- 400: unsupported file type, or unreadable image/video
- 413: file too large
- 422: no face found in video
- 500: model / internal error

## GET /health
`{"status": "ok", "dummy_mode": true}`  (`dummy_mode` is true until ML code is plugged in)

## What I need from the ML roles (Day 2)
| File | Function | Owner | Returns |
|---|---|---|---|
| `ml/predict.py` | `predict_image(path)` | DEEP BRANCH | `{"score": float, "patch_scores": [[float]]}` (no verdict needed) |
| `ml/video_utils.py` | `sample_frames(path)` | DEEP BRANCH | list of `(t_seconds, PIL.Image)` face crops |

Backend applies the threshold and renders the heatmap, so ML code never touches either.
