# Deploy (Day 3)

## Settings (environment variables, no code edits)
| Variable | Default | Use |
|---|---|---|
| `THRESHOLD` | 0.5 | Set to the value DEEP BRANCH tuned on validation data |
| `ALLOWED_ORIGINS` | `*` | Set to the frontend URL once it is live, e.g. `https://deeptrace.vercel.app` |
| `MAX_VIDEO_FRAMES` | 30 | Frames scored per video (evenly spread). Lower it if videos are slow |
| `VIDEO_TIME_BUDGET_S` | 60 | After this, the video returns the frames scored so far |
| `MAX_CONCURRENT_JOBS` | 2 | Model calls at once; extra requests wait their turn |
| `USE_DUMMY` | unset | `1` forces dummy mode (fallback for the live demo) |

## Option A: Docker on Render / Railway / Fly / Hugging Face Spaces (Docker SDK)
From the repo root: `docker build -f backend/Dockerfile -t deeptrace .`
Point the platform at the Dockerfile with the repo root as build context. It sets `$PORT`.
Hugging Face Spaces expects port 7860: set `PORT=7860`.

## Option B: no Docker
Start command from the `backend` folder, with `PYTHONPATH` set to the repo root:
`uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## After deploying
1. `GET /health` shows `dummy_mode: false` (if true, `ml/` is not importable).
2. Send the public URL to Frontend; you set `ALLOWED_ORIGINS` to their URL.
3. Timing: every analyze response has an `X-Process-Time-ms` header. Test 10 images and 2 videos
   on the live link and record the times for the report. Requests slower than 8 s log a warning.
4. First request after a platform cold start is slow (free tiers sleep). Hit `/health` a minute before the demo.
