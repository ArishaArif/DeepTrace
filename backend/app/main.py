import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app import config
from app.heatmap import render_heatmap
from app.inference import DUMMY, INFER_SLOT, predict_image
from app.schemas import ImageResult, VideoResult
from app.video import analyze_video

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app):
    """Warm up the real model once at startup so the first user request is not slow."""
    if not DUMMY:
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                path = tmp.name
            Image.new("RGB", (256, 256), (128, 128, 128)).save(path)
            t0 = time.perf_counter()
            predict_image(path)
            logger.info("Model warm-up done in %.1fs", time.perf_counter() - t0)
        except Exception:
            logger.exception("Model warm-up failed (server keeps running)")
        finally:
            if path and os.path.exists(path):
                os.remove(path)
    yield


app = FastAPI(title="Deepfake Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-ms"],
)


@app.middleware("http")
async def timing(request: Request, call_next):
    """Time every request. Reported in a header so the JSON contract stays untouched."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - t0
    response.headers["X-Process-Time-ms"] = str(int(dt * 1000))
    if request.url.path.startswith("/analyze"):
        level = logging.WARNING if dt > config.SLOW_WARN_S else logging.INFO
        logger.log(level, "%s took %.2fs (status %s)", request.url.path, dt, response.status_code)
    return response


def save_upload(file, allowed, max_mb):
    """Check type and size, save to a temp file, return its path.
    Reads in chunks and stops as soon as the limit is passed, so a huge upload is never held in RAM."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type. Use one of: {sorted(allowed)}")
    limit = max_mb * 1024 * 1024
    written = 0
    too_big = False
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                too_big = True
                break
            tmp.write(chunk)
    if too_big or written == 0:
        os.remove(tmp.name)
        if too_big:
            raise HTTPException(413, f"File too large (max {max_mb} MB)")
        raise HTTPException(400, "Empty file")
    return tmp.name


@app.get("/health")
def health():
    return {"status": "ok", "dummy_mode": DUMMY}


@app.post("/analyze/image", response_model=ImageResult)
def analyze_image(file: UploadFile = File(...)):
    path = save_upload(file, config.IMAGE_TYPES, config.MAX_IMAGE_MB)
    try:
        try:
            with Image.open(path) as im:      # closes the file so Windows can delete it
                im.load()
                size = im.size
        except Exception:
            raise HTTPException(400, "Could not read image")
        try:
            with INFER_SLOT:
                result = predict_image(path)
            result["heatmap_png"] = render_heatmap(result["patch_scores"], size)
        except OSError:
            logger.exception("Unreadable image reached the model")
            raise HTTPException(400, "Could not read image")
        except Exception:
            logger.exception("Image analysis failed")
            raise HTTPException(500, "Model error while analysing the image")
        return result
    finally:
        os.remove(path)


@app.post("/analyze/video", response_model=VideoResult)
def analyze_video_endpoint(file: UploadFile = File(...)):
    path = save_upload(file, config.VIDEO_TYPES, config.MAX_VIDEO_MB)
    try:
        return analyze_video(path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except OSError:
        logger.exception("Unreadable video")
        raise HTTPException(400, "Could not read video")
    except Exception:
        logger.exception("Video analysis failed")
        raise HTTPException(500, "Model error while analysing the video")
    finally:
        os.remove(path)
