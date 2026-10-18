from pydantic import BaseModel


class ImageResult(BaseModel):
    verdict: str                    # "fake" or "real"
    score: float                    # 0 to 1, higher = more likely fake
    patch_scores: list[list[float]] # rows x cols, each 0 to 1
    heatmap_png: str                # base64 PNG, transparent overlay, same size as upload


class FrameScore(BaseModel):
    t: float                        # seconds
    score: float


class VideoResult(BaseModel):
    verdict: str
    score: float                    # mean of frame scores
    frames: list[FrameScore]
