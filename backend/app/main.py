import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ImageAnalysisResponse, VideoAnalysisResponse
from app.inference import run_image_pipeline, run_video_pipeline
from app.heatmap import generate_heatmap_base64

app = FastAPI(
    title="Explainable Deepfake Detection API",
    description="Backend integration serving fused predictions from Fractal & Deep Branch models."
)

# Enable CORS for Frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Deepfake Detection API is running"}

@app.post("/analyze/image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run ML inference on image
        ml_result = run_image_pipeline(tmp_path)
        
        # Render blended heatmap base64 overlay
        heatmap_b64 = generate_heatmap_base64(tmp_path, ml_result["patch_scores"])

        return {
            "verdict": ml_result["verdict"],
            "score": ml_result["score"],
            "patch_scores": ml_result["patch_scores"],
            "heatmap_png": heatmap_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

@app.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File uploaded must be a video.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run ML inference on video frames
        video_result = run_video_pipeline(tmp_path)
        return video_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)