import os
import tempfile

import numpy as np

from app.config import THRESHOLD
from app.inference import DUMMY, predict_image

sample_frames = None
if not DUMMY:
    try:
        # DEEP BRANCH: sample_frames(path) -> list of (t_seconds, PIL.Image of the cropped face)
        from ml.video_utils import sample_frames
    except ImportError:
        # ml.predict can ship before ml.video_utils; keep the server (and /analyze/image) up.
        sample_frames = None


def analyze_video(path):
    if DUMMY:
        rng = np.random.default_rng(len(path))
        scored = [(float(t), float(rng.random())) for t in range(8)]
    else:
        if sample_frames is None:
            raise RuntimeError("ml.video_utils.sample_frames is not available yet")
        scored = []
        for t, face in sample_frames(path):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                face.save(tmp.name)
            try:
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
