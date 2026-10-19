import os
import tempfile
import time

import numpy as np

from app import config
from app.config import THRESHOLD
from app.inference import DUMMY, INFER_SLOT, predict_image

sample_frames = None
if not DUMMY:
    try:
        # DEEP BRANCH: sample_frames(path) -> list of (t_seconds, PIL.Image of the cropped face)
        from ml.video_utils import sample_frames
    except ImportError:
        # ml.predict can ship before ml.video_utils; keep the server (and /analyze/image) up.
        sample_frames = None


def _spread(items, n):
    """Pick at most n items, evenly spread over the whole video (not just the first n)."""
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def analyze_video(path):
    if DUMMY:
        rng = np.random.default_rng(len(path))
        scored = [(float(t), float(rng.random())) for t in range(8)]
    else:
        if sample_frames is None:
            raise RuntimeError("ml.video_utils.sample_frames is not available yet")
        faces = _spread(list(sample_frames(path)), config.MAX_VIDEO_FRAMES)
        scored = []
        start = time.perf_counter()
        for t, face in faces:
            if scored and time.perf_counter() - start > config.VIDEO_TIME_BUDGET_S:
                break                      # time budget hit: return the partial timeline
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                face.save(tmp.name)
            try:
                with INFER_SLOT:
                    scored.append((float(t), predict_image(tmp.name)["score"]))
            finally:
                os.remove(tmp.name)

    if not scored:
        raise ValueError("no faces found in video")

    score = sum(s for _, s in scored) / len(scored)
    return {
        "verdict": "fake" if score >= THRESHOLD else "real",
        "score": score,
        "frames": [{"t": t, "score": s} for t, s in scored],
    }
