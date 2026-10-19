from pydantic import BaseModel
from typing import List

class FrameScore(BaseModel):
    t: float
    score: float

class ImageAnalysisResponse(BaseModel):
    verdict: str  # "Real" or "Fake"
    score: float  # Fused prediction probability (0.0 to 1.0)
    patch_scores: List[List[float]]  # Patch matrix from Fractal Branch
    heatmap_png: str  # Base64 encoded PNG data URI

class VideoAnalysisResponse(BaseModel):
    verdict: str  # "Real" or "Fake"
    score: float  # Aggregated score over all frames
    frames: List[FrameScore]  # Individual frame timings and scores