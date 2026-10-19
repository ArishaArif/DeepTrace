"""Frozen shared contract for the Fractal Branch."""

PATCH = 64
STRIDE = 32
MIN_SIZE = 128
BOX_SIZES = (2, 4, 8, 16, 32)
LACUNARITY_SIZES = (4, 8, 16)
GAUSSIAN_SIGMA = 2.0
EPS = 1e-12
SEED = 42
LABEL_REAL = 0
LABEL_FAKE = 1
METHODS = ("mean", "otsu", "highpass")
FEATURES = ("D", "lac4", "lac8", "lac16")
DEFAULT_METHODS = {"mean", "otsu", "highpass"}
