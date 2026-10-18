import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app import config
from app.heatmap import render_heatmap
from app.inference import DUMMY, predict_image
from app.schemas import ImageResult, VideoResult
from app.video import analyze_video

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Deepfake Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def save_upload(file, allowed, max_mb):
    """Check type and size, save to a temp file, return its path."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type. Use one of: {sorted(allowed)}")
    data = file.file.read()
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {max_mb} MB)")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
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
                size = im.size
        except Exception:
            raise HTTPException(400, "Could not read image")
        try:
            result = predict_image(path)
            result["heatmap_png"] = render_heatmap(result["patch_scores"], size)
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
    except Exception:
        logger.exception("Video analysis failed")
        raise HTTPException(500, "Model error while analysing the video")
    finally:
        os.remove(path)
