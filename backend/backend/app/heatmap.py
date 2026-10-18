import base64
import io

import numpy as np
from PIL import Image


def render_heatmap(patch_scores, size):
    """Turn the patch score grid into a transparent red overlay (base64 PNG).
    Low scores are see-through, high scores are solid red.
    size = (width, height) of the original image."""
    grid = np.clip(np.array(patch_scores, dtype=np.float32), 0, 1)
    small = Image.fromarray(grid, mode="F")
    big = np.clip(np.array(small.resize(size, Image.BICUBIC)), 0, 1)

    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 60
    rgba[..., 3] = (big * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
