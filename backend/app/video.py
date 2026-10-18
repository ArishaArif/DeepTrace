import os
import tempfile

import numpy as np

from app.config import THRESHOLD
from app.inference import DUMMY, predict_image

if not DUMMY:
    # DEEP BRANCH: sample_frames(path) -> list of (t_seconds, PIL.Image of the cropped face)
    from ml.video_utils import sample_frames


def analyze_video(path):
    if DUMMY:
        rng = np.random.default_rng(len(path))
        scored = [(float(t), float(rng.random())) for t in range(8)]
    else:
        scored = []
        for t, face in sample_frames(path):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                face.save(tmp.name)
            try:
                scored.append((t, predict_image(tmp.name)["score"]))
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
