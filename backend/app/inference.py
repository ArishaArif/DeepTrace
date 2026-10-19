import sys
from pathlib import Path
from typing import Dict, Any

# Ensure root directory is on PATH so ml/ modules import cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import interfaces directly from ML modules[cite: 1, 2]
from ml.deep.predict import predict_image, predict_deep
from ml.fractal.predict import predict_fractal
from app.video import extract_frames, cleanup_frames

def run_image_pipeline(image_path: str) -> Dict[str, Any]:
    """
    Executes ML pipeline for single image via deep branch predict_image fusion wrapper[cite: 1].
    """
    result = predict_image(image_path)
    return {
        "verdict": result.get("verdict", "Real"),
        "score": float(result.get("score", 0.0)),
        "patch_scores": result.get("patch_scores", [])
    }

def run_video_pipeline(video_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Extracts frames from video, computes score per frame, and aggregates video prediction.
    """
    sampled_frames = extract_frames(video_path, fps_sample_rate=1.0)
    if not sampled_frames:
        raise ValueError("No frames could be extracted from the video.")

    frame_results = []
    total_score = 0.0

    try:
        for timestamp, frame_file in sampled_frames:
            # Predict frame score using single image pipeline
            frame_res = run_image_pipeline(frame_file)
            score = frame_res["score"]
            total_score += score
            
            frame_results.append({
                "t": round(timestamp, 2),
                "score": float(score)
            })

        avg_score = total_score / len(sampled_frames)
        verdict = "Fake" if avg_score >= threshold else "Real"

        return {
            "verdict": verdict,
            "score": float(avg_score),
            "frames": frame_results
        }
    finally:
        cleanup_frames(sampled_frames)