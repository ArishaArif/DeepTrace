# DeepTrace Backend

FastAPI service for the explainable deepfake detection project (fractal texture analysis + CLIP).
It serves image and video analysis behind the frozen contract in [`API_CONTRACT.md`](API_CONTRACT.md).

## Run (Windows)

```bat
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- To let teammates on the same network reach it: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Endpoints

| Method | Path | Input | Returns |
|---|---|---|---|
| GET | `/health` | none | `{status, dummy_mode}` |
| POST | `/analyze/image` | `file` (jpg, png, webp; max 10 MB) | `{verdict, score, patch_scores, heatmap_png}` |
| POST | `/analyze/video` | `file` (mp4, mov, avi, mkv; max 50 MB) | `{verdict, score, frames: [{t, score}]}` |

Errors always use `{"detail": "message"}` with status 400, 413, 422 or 500. Full details are in `API_CONTRACT.md`.

## Dummy mode and real models

The backend returns random but contract-shaped data until the ML code exists.
It switches to the real models automatically when `ml/predict.py` (owned by the Deep Branch) can be imported.
`/health` reports which mode is active (`dummy_mode`).

- Force dummy mode: `set USE_DUMMY=1` before starting the server.
- Real mode from the `backend` folder: the repo's `ml/` package must be importable, so run `set PYTHONPATH=..` first.
- The decision threshold lives in `app/config.py` (`THRESHOLD`, default 0.5).

## Tests

```bat
pytest -v
```

The tests check the response shapes against the contract (keys, value ranges, patch grid size, heatmap size) and the error codes.

## Layout

```
backend/
  app/
    main.py        endpoints, upload checks, error handling
    inference.py   dummy or real image prediction, applies the threshold
    video.py       frame scoring and averaging
    heatmap.py     patch_scores -> transparent red PNG overlay
    schemas.py     response models
    config.py      limits, allowed types, threshold
  test_api.py
  requirements.txt
  API_CONTRACT.md
```

## Notes

This is a screening tool, not forensic evidence. The heatmap shows regions contributing to the fake score,
not proof of manipulation.
