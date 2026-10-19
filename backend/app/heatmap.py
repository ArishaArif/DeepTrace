import base64
import io

import numpy as np
from PIL import Image

PATCH, STRIDE = 64, 32


def _cell_grid(grid):
    """Patch (r, c) covers pixels [r*32, r*32+64). Split the canvas into 32 px cells; each cell
    takes the mean of the (up to 4) patches covering it. Result: (rows+1) x (cols+1), so the
    heat lines up with the pixels the patches actually looked at (no edge stretching)."""
    p = np.pad(grid, 1, constant_values=np.nan)
    stack = np.stack([p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]])
    return np.nanmean(stack, axis=0).astype(np.float32)


def render_heatmap(patch_scores, size):
    """Turn the patch score grid into a transparent red overlay (base64 PNG).
    Low scores are see-through, high scores are solid red.
    size = (width, height) of the original image."""
    grid = np.clip(np.array(patch_scores, dtype=np.float32), 0, 1)
    cells = Image.fromarray(_cell_grid(grid), mode="F")
    big = np.clip(np.array(cells.resize(size, Image.BICUBIC)), 0, 1)

    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 60
    rgba[..., 3] = (big * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
