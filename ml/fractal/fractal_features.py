from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    from .config import (
        BOX_SIZES,
        GAUSSIAN_SIGMA,
        LACUNARITY_SIZES,
        MIN_SIZE,
        PATCH,
        STRIDE,
    )
except ImportError:
    from config import (
        BOX_SIZES,
        GAUSSIAN_SIGMA,
        LACUNARITY_SIZES,
        MIN_SIZE,
        PATCH,
        STRIDE,
    )

EPS = 1e-12


def load_gray(source):
    """Load an image as a grayscale float32 array in [0, 1].

    source may be a path or an already opened PIL image. Images whose shorter
    side is below MIN_SIZE are upscaled proportionally with bicubic
    interpolation so that the shorter side becomes exactly MIN_SIZE.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    else:
        img = source
    if img.mode != "L":
        img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape
    short = min(h, w)
    if short < MIN_SIZE:
        scale = MIN_SIZE / float(short)
        nw = max(MIN_SIZE, int(round(w * scale)))
        nh = max(MIN_SIZE, int(round(h * scale)))
        arr = np.asarray(img.resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr, dtype=np.float32)


def extract_patches(arr):
    """Return patches [N, PATCH, PATCH], rows, cols in row major order."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    h, w = arr.shape
    if h < PATCH or w < PATCH:
        raise ValueError(f"image {h}x{w} is smaller than patch size {PATCH}")
    rows = (h - PATCH) // STRIDE + 1
    cols = (w - PATCH) // STRIDE + 1
    sy, sx = arr.strides
    view = np.lib.stride_tricks.as_strided(
        arr,
        shape=(rows, cols, PATCH, PATCH),
        strides=(sy * STRIDE, sx * STRIDE, sy, sx),
        writeable=False,
    )
    patches = view.reshape(rows * cols, PATCH, PATCH).copy()
    return patches, rows, cols


def _otsu_thresholds(patches):
    n = patches.shape[0]
    quant = np.clip((patches.reshape(n, -1) * 255.0).round(), 0.0, 255.0).astype(np.int64)
    offsets = (np.arange(n, dtype=np.int64) * 256).reshape(n, 1)
    hist = np.bincount((quant + offsets).ravel(), minlength=n * 256).reshape(n, 256)
    hist = hist.astype(np.float64)
    total = np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
    prob = hist / total
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(prob, axis=1)
    mu = np.cumsum(prob * levels, axis=1)
    mu_total = mu[:, -1:]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_total * omega - mu) ** 2 / denom
    between = np.where(denom > 0, between, -np.inf)
    return (np.argmax(between, axis=1).astype(np.float64) + 0.5) / 255.0


def binarize(patches, method="mean"):
    """Binarize one patch or a batch of patches with the given method."""
    p = np.asarray(patches, dtype=np.float32)
    single = p.ndim == 2
    if single:
        p = p[None]
    if method == "mean":
        b = p > p.mean(axis=(1, 2), keepdims=True)
    elif method == "otsu":
        b = p > _otsu_thresholds(p)[:, None, None]
    elif method == "highpass":
        blurred = gaussian_filter(p, sigma=(0.0, GAUSSIAN_SIGMA, GAUSSIAN_SIGMA), mode="reflect")
        residual = p - blurred
        b = residual > residual.mean(axis=(1, 2), keepdims=True)
    else:
        raise ValueError(f"unknown binarization method: {method}")
    return b[0] if single else b


def _box_counts(batch, sizes):
    counts = []
    used = []
    for s in sizes:
        if s > batch.shape[1] or s > batch.shape[2]:
            continue
        hb = (batch.shape[1] // s) * s
        wb = (batch.shape[2] // s) * s
        bb = batch[:, :hb, :wb]
        n_h = hb // s
        n_w = wb // s
        c = bb.reshape(bb.shape[0], n_h, s, n_w, s).any(axis=(2, 4)).sum(axis=(1, 2))
        counts.append(c.astype(np.float64))
        used.append(float(s))
    return np.stack(counts, axis=1), np.asarray(used, dtype=np.float64)


def _loglog_slope(counts, sizes):
    xs = np.log(1.0 / sizes)
    mask = counts > 0
    n_valid = mask.sum(axis=1)
    out = np.full(counts.shape[0], np.nan, dtype=np.float64)
    ok = n_valid >= 2
    if not np.any(ok):
        return out
    c = counts[ok]
    m = mask[ok].astype(np.float64)
    y = np.log(np.maximum(c, 1.0))
    xbar = (m * xs).sum(axis=1) / m.sum(axis=1)
    ybar = (m * y).sum(axis=1) / m.sum(axis=1)
    dx = m * (xs - xbar[:, None])
    num = (dx * (y - ybar[:, None])).sum(axis=1)
    den = (dx * (xs - xbar[:, None])).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = num / den
    slope = np.where(den > 0, slope, np.nan)
    out[ok] = slope
    return out


def fractal_dimension(b):
    """Box counting dimension of one binary patch or a batch of patches."""
    b = np.asarray(b, dtype=bool)
    single = b.ndim == 2
    if single:
        b = b[None]
    counts, sizes = _box_counts(b, BOX_SIZES)
    dims = _loglog_slope(counts, sizes)
    return float(dims[0]) if single else dims


def lacunarity(b, r):
    """Gliding box lacunarity at box size r over fully valid window positions."""
    m = np.asarray(b, dtype=np.float64)
    single = m.ndim == 2
    if single:
        m = m[None]
    r = int(r)
    h, w = m.shape[1], m.shape[2]
    if r <= 0 or r > min(h, w):
        out = np.full(m.shape[0], np.nan)
        return float(out[0]) if single else out
    integral = np.zeros((m.shape[0], h + 1, w + 1), dtype=np.float64)
    integral[:, 1:, 1:] = np.cumsum(np.cumsum(m, axis=1), axis=2)
    masses = (
        integral[:, r:, r:]
        - integral[:, :-r, r:]
        - integral[:, r:, :-r]
        + integral[:, :-r, :-r]
    )
    mean_mass = masses.mean(axis=(1, 2))
    mean_sq = (masses * masses).mean(axis=(1, 2))
    lac = mean_sq / (mean_mass * mean_mass + EPS)
    lac = np.where(mean_mass > 0, lac, 1.0)
    lac = np.where(np.isfinite(lac), lac, 1.0)
    return float(lac[0]) if single else lac


def features_for_patches(patches, method="mean"):
    """Feature matrix [N, 4] of (D, lac4, lac8, lac16) for a patch batch."""
    p = np.asarray(patches, dtype=np.float32)
    single = p.ndim == 2
    if single:
        p = p[None]
    b = binarize(p, method)
    dims = fractal_dimension(b)
    lacs = [lacunarity(b, r) for r in LACUNARITY_SIZES]
    feats = np.column_stack([dims] + lacs).astype(np.float64)
    feats[:, 0] = np.where(np.isfinite(feats[:, 0]), feats[:, 0], 0.0)
    for j in range(1, feats.shape[1]):
        feats[:, j] = np.where(np.isfinite(feats[:, j]), feats[:, j], 1.0)
    return feats[0] if single else feats


def patch_features(patch, method="mean"):
    """Return [D, lac4, lac8, lac16] for a single patch."""
    return features_for_patches(patch, method=method)
