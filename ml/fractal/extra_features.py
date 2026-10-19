"""Extra feature candidates for Day 3 ablation.

Never alter the frozen functions in fractal_features.py.
Each function operates on a batch of binarized patches [N, PATCH, PATCH] (bool)
or raw float patches [N, PATCH, PATCH] depending on the feature group.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from .config import BOX_SIZES, EPS, GAUSSIAN_SIGMA, LACUNARITY_SIZES, PATCH
    from .fractal_features import _box_counts, _loglog_slope, binarize, lacunarity
except ImportError:
    from config import BOX_SIZES, EPS, GAUSSIAN_SIGMA, LACUNARITY_SIZES, PATCH
    from fractal_features import _box_counts, _loglog_slope, binarize, lacunarity


# ---------------------------------------------------------------------------
# Group B: Extra lacunarity scales + R-squared of box-counting regression
# ---------------------------------------------------------------------------

def extra_lacunarity(b: np.ndarray) -> np.ndarray:
    """Return columns [lac2, lac32] for batch b of binarized patches."""
    b = np.asarray(b, dtype=bool)
    if b.ndim == 2:
        b = b[None]
    lac2 = lacunarity(b, 2)
    lac32 = lacunarity(b, 32)
    result = np.column_stack([lac2, lac32])
    # Guard: replace any remaining NaN/inf with 1.0 (neutral lacunarity)
    result = np.where(np.isfinite(result), result, 1.0)
    return result


def box_counting_r2(b: np.ndarray) -> np.ndarray:
    """R-squared of the log-log linear fit used for fractal dimension."""
    b = np.asarray(b, dtype=bool)
    if b.ndim == 2:
        b = b[None]
    n = b.shape[0]
    # Guard: if patch too small for any box size, return zeros
    h, w = b.shape[1], b.shape[2]
    valid_sizes = [s for s in BOX_SIZES if s <= h and s <= w]
    if len(valid_sizes) < 3:
        return np.zeros(n, dtype=np.float64)
    counts, sizes = _box_counts(b, valid_sizes)
    xs = np.log(1.0 / sizes)  # shape (K,)
    mask = counts > 0  # (N, K)
    n_valid = mask.sum(axis=1)
    out = np.zeros(n, dtype=np.float64)
    ok = n_valid >= 3  # need at least 3 points for meaningful R^2
    if not np.any(ok):
        return out
    c = counts[ok]
    m = mask[ok].astype(np.float64)
    y = np.log(np.maximum(c, 1.0))
    n_pts = m.sum(axis=1, keepdims=True)
    xbar = (m * xs).sum(axis=1, keepdims=True) / n_pts
    ybar = (m * y).sum(axis=1, keepdims=True) / n_pts
    dx = m * (xs - xbar)
    dy = m * (y - ybar)
    ss_xx = (dx * dx).sum(axis=1)
    ss_yy = (dy * dy).sum(axis=1)
    ss_xy = (dx * dy).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_xx > 0, (ss_xy * ss_xy) / (ss_xx * np.maximum(ss_yy, EPS)), np.nan)
    r2 = np.where(np.isfinite(r2), r2, 0.0)
    out[ok] = r2
    return out


# ---------------------------------------------------------------------------
# Group C: Local slopes on separate scale ranges
# ---------------------------------------------------------------------------

SMALL_SIZES = (2, 4, 8)
LARGE_SIZES = (8, 16, 32)


def local_slopes(b: np.ndarray) -> np.ndarray:
    """Return [slope_small, slope_large, slope_diff]."""
    b = np.asarray(b, dtype=bool)
    if b.ndim == 2:
        b = b[None]
    counts_s, sizes_s = _box_counts(b, SMALL_SIZES)
    counts_l, sizes_l = _box_counts(b, LARGE_SIZES)
    slope_s = _loglog_slope(counts_s, sizes_s)
    slope_l = _loglog_slope(counts_l, sizes_l)
    slope_s = np.where(np.isfinite(slope_s), slope_s, 0.0)
    slope_l = np.where(np.isfinite(slope_l), slope_l, 0.0)
    diff = slope_s - slope_l
    return np.column_stack([slope_s, slope_l, diff])


# ---------------------------------------------------------------------------
# Group D: Multifractal features
# ---------------------------------------------------------------------------

MULTIFRACTAL_BOXES = (4, 8, 16, 32)


def _partition_masses(patches: np.ndarray, box_size: int) -> np.ndarray:
    """For each patch, compute masses per box. Returns [N, n_boxes_total]."""
    n, h, w = patches.shape
    hb = (h // box_size) * box_size
    wb = (w // box_size) * box_size
    cropped = patches[:, :hb, :wb]
    n_h = hb // box_size
    n_w = wb // box_size
    # Reshape to [N, n_h, box, n_w, box] → sum over box dims → [N, n_h, n_w]
    reshaped = cropped.reshape(n, n_h, box_size, n_w, box_size)
    masses = reshaped.sum(axis=(2, 4))
    return masses.reshape(n, -1)  # [N, n_h * n_w]


def _multifractal_Dq(measure_batch: np.ndarray, q_values=(0, 1, 2)) -> np.ndarray:
    """Compute D_q for q in q_values for a batch of float patches [N, H, W].

    measure_batch should be non-negative and normalized per-patch to sum=1.
    Returns array [N, len(q_values)].
    """
    n = measure_batch.shape[0]
    results = np.full((n, len(q_values)), np.nan, dtype=np.float64)
    sizes = np.array(MULTIFRACTAL_BOXES, dtype=np.float64)
    log_inv_eps = np.log(sizes)  # log(epsilon) — slope vs log(1/eps)

    # Precompute partition sums for each scale
    chi = {q: [] for q in q_values}
    for s in MULTIFRACTAL_BOXES:
        boxes = _partition_masses(measure_batch, s)  # [N, n_boxes]
        # Normalize per-patch (some patches may have zero total)
        row_sums = boxes.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(row_sums > EPS, boxes / row_sums, 0.0)
        for qi, q in enumerate(q_values):
            if q == 0:
                # chi_0 = number of boxes with nonzero mass
                val = (p > EPS).sum(axis=1).astype(np.float64)
            elif q == 1:
                # chi_1 = exp(-sum p log p) — but we work with log(chi)
                # Actually for q=1 we use Shannon: S(eps) = -sum p log p
                # tau(1) from limit: D_1 = lim S(eps)/log(1/eps)
                # We store the entropy directly
                with np.errstate(divide="ignore", invalid="ignore"):
                    p_log_p = np.where(p > EPS, p * np.log(p + EPS), 0.0)
                val = -p_log_p.sum(axis=1)  # entropy at this scale
            else:
                # chi_q = sum p^q
                with np.errstate(divide="ignore", invalid="ignore"):
                    val = np.where(p > EPS, p ** q, 0.0).sum(axis=1)
            chi[q].append(val)

    # Now fit slope of log(chi_q) vs log(1/eps) for each q
    # For D_q: chi_q(eps) ~ eps^{tau(q)}, so log(chi_q) ~ tau(q) * log(eps)
    # D_q = tau(q)/(q-1)
    log_inv = np.log(1.0 / sizes)  # log(1/eps) — note: log(eps) = -log(1/eps)

    for qi, q in enumerate(q_values):
        vals = np.stack(chi[q], axis=1)  # [N, len(sizes)]
        if q == 0:
            # chi_0 = count of occupied boxes, log(chi_0) ~ D_0 * log(1/eps)
            log_chi = np.log(np.maximum(vals, 1.0))
            slope = _simple_slope(log_chi, log_inv, n)
            results[:, qi] = slope  # D_0
        elif q == 1:
            # entropy S(eps) ~ D_1 * log(1/eps)
            # vals already are entropies, fit slope directly
            slope = _simple_slope(vals, log_inv, n)
            results[:, qi] = slope  # D_1
        else:
            # chi_q = sum p^q, log(chi_q) ~ tau(q) * log(eps) = -tau(q) * log(1/eps)
            # tau(q) = (q-1)*D_q
            # so D_q = slope / (q-1) where slope is of log(chi_q) vs log(1/eps)
            log_chi = np.log(np.maximum(vals, EPS))
            slope = _simple_slope(log_chi, log_inv, n)
            # tau(q) = -slope (since log(chi) ~ -tau(q) * log(1/eps))
            tau_q = -slope
            results[:, qi] = tau_q / (q - 1.0)  # D_q

    return np.where(np.isfinite(results), results, 0.0)


def _simple_slope(y: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """OLS slope of y vs x for each row. y: [N, K], x: [K]."""
    K = len(x)
    x_mean = x.mean()
    y_mean = y.mean(axis=1, keepdims=True)
    x_c = x - x_mean  # [K]
    y_c = y - y_mean  # [N, K]
    ss_xy = (y_c * x_c).sum(axis=1)
    ss_xx = (x_c * x_c).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(ss_xx > 0, ss_xy / ss_xx, 0.0)
    return slope


def multifractal_features(raw_patches: np.ndarray) -> np.ndarray:
    """Compute multifractal features for raw float patches [N, H, W] in [0,1].

    Returns columns:
    [D0_grad, D1_grad, D2_grad, width_grad, D0_int, D1_int, D2_int, width_int]
    """
    p = np.asarray(raw_patches, dtype=np.float64)
    if p.ndim == 2:
        p = p[None]
    n = p.shape[0]

    # Intensity measure: normalized intensity
    intensity = np.maximum(p, 0.0)
    i_sums = intensity.reshape(n, -1).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        int_measure = np.where(i_sums > EPS, intensity.reshape(n, -1) / i_sums, 0.0).reshape(n, p.shape[1], p.shape[2])

    # Gradient magnitude measure
    gy = np.zeros_like(p)
    gx = np.zeros_like(p)
    gy[:, 1:, :] = p[:, 1:, :] - p[:, :-1, :]
    gx[:, :, 1:] = p[:, :, 1:] - p[:, :, :-1]
    grad = np.sqrt(gx ** 2 + gy ** 2)
    g_sums = grad.reshape(n, -1).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        grad_measure = np.where(g_sums > EPS, grad.reshape(n, -1) / g_sums, 0.0).reshape(n, p.shape[1], p.shape[2])

    # Compute D_q for each measure
    q_vals = (0, 1, 2)
    d_grad = _multifractal_Dq(grad_measure, q_vals)
    d_int = _multifractal_Dq(int_measure, q_vals)

    width_grad = d_grad[:, 0] - d_grad[:, 2]
    width_int = d_int[:, 0] - d_int[:, 2]

    return np.column_stack([d_grad, width_grad, d_int, width_int])


# ---------------------------------------------------------------------------
# Group E: Multi-representation (features on two binarizations)
# ---------------------------------------------------------------------------

def multi_rep_features(patches: np.ndarray, methods=("highpass", "mean")) -> np.ndarray:
    """Core features computed on multiple binarizations, concatenated."""
    try:
        from .fractal_features import features_for_patches
    except ImportError:
        from fractal_features import features_for_patches

    p = np.asarray(patches, dtype=np.float32)
    if p.ndim == 2:
        p = p[None]
    parts = []
    for m in methods:
        feats = features_for_patches(p, method=m)
        parts.append(feats)
    return np.column_stack(parts)


# ---------------------------------------------------------------------------
# Aggregation: compute all candidate feature groups for a batch of raw patches
# ---------------------------------------------------------------------------

CANDIDATE_FEATURES = {
    "A": ["D", "lac4", "lac8", "lac16"],
    "B": ["D", "lac4", "lac8", "lac16", "lac2", "lac32", "r2"],
    "C": ["D", "lac4", "lac8", "lac16", "slope_small", "slope_large", "slope_diff"],
    "D": ["D", "lac4", "lac8", "lac16", "D0_grad", "D1_grad", "D2_grad", "width_grad",
          "D0_int", "D1_int", "D2_int", "width_int"],
    "E": ["D_hp", "lac4_hp", "lac8_hp", "lac16_hp", "D_mean", "lac4_mean", "lac8_mean", "lac16_mean"],
    "F": ["D", "lac4", "lac8", "lac16", "lac2", "lac32", "r2",
          "slope_small", "slope_large", "slope_diff",
          "D0_grad", "D1_grad", "D2_grad", "width_grad", "D0_int", "D1_int", "D2_int", "width_int",
          "D_mean", "lac4_mean", "lac8_mean", "lac16_mean"],
}


def compute_candidate(candidate: str, patches: np.ndarray, method: str = "highpass") -> np.ndarray:
    """Compute features for a named candidate on raw patches [N, PATCH, PATCH]."""
    try:
        from .fractal_features import features_for_patches
    except ImportError:
        from fractal_features import features_for_patches

    p = np.asarray(patches, dtype=np.float32)
    if p.ndim == 2:
        p = p[None]
    b = binarize(p, method=method)

    if candidate == "A":
        return features_for_patches(p, method=method)

    elif candidate == "B":
        base = features_for_patches(p, method=method)
        extra_lac = extra_lacunarity(b)
        r2 = box_counting_r2(b)
        return np.column_stack([base, extra_lac, r2])

    elif candidate == "C":
        base = features_for_patches(p, method=method)
        slopes = local_slopes(b)
        return np.column_stack([base, slopes])

    elif candidate == "D":
        base = features_for_patches(p, method=method)
        mf = multifractal_features(p.astype(np.float64))
        return np.column_stack([base, mf])

    elif candidate == "E":
        return multi_rep_features(p, methods=("highpass", "mean"))

    elif candidate == "F":
        # Union of every group that improved on train CV (B + C + D + the mean arm of E)
        base = features_for_patches(p, method=method)                 # highpass core (4)
        extra_lac = extra_lacunarity(b)                               # lac2, lac32
        r2 = box_counting_r2(b)[:, None]                              # r2
        slopes = local_slopes(b)                                      # slope_small/large/diff
        mf = multifractal_features(p.astype(np.float64))             # 8 multifractal
        mean_core = features_for_patches(p, method="mean")           # mean-binarized core (4)
        return np.column_stack([base, extra_lac, r2, slopes, mf, mean_core])

    else:
        raise ValueError(f"unknown candidate: {candidate}")
