"""
Smooth approximations for non-differentiable objective terms.

Used by MultiObjectiveScore when use_smooth_objectives is True to restore
gradient continuity for SA and other scalar optimizers.
"""

import numpy as np

# Defaults from plan: p≈8, k≈10; k bounded [5, 20] to avoid overflow
DEFAULT_JAIN_SMOOTH_P = 8.0
DEFAULT_SOFTPLUS_K = 10.0
SOFTPLUS_K_MIN = 5.0
SOFTPLUS_K_MAX = 20.0
JAIN_EPS = 1e-6


def smooth_min_jain(j_r: float, j_i: float, p: float = DEFAULT_JAIN_SMOOTH_P) -> float:
    """
    Generalized-mean smooth minimum of two Jain indices (L_{-p} mean).

    J_smooth = ((J_R^{-p} + J_I^{-p}) / 2)^{-1/p}
    Clamps J_R, J_I to [eps, 1] to avoid blow-up. Lower is worse (bottleneck).
    """
    j_r = max(JAIN_EPS, min(1.0, float(j_r)))
    j_i = max(JAIN_EPS, min(1.0, float(j_i)))
    if p <= 0:
        p = DEFAULT_JAIN_SMOOTH_P
    half_inv_p = 0.5 * (j_r ** (-p) + j_i ** (-p))
    if half_inv_p <= 0:
        return min(j_r, j_i)
    return float(half_inv_p ** (-1.0 / p))


def softplus_hinge(x: float, k: float = DEFAULT_SOFTPLUS_K) -> float:
    """
    Softplus approximation to max(0, x). Smooth hinge for Moran's I term.

    softplus(x, k) = (1/k) * ln(1 + exp(k*x))
    Numerically stable: for large k*x use max(0,x) + ln(1+exp(-|k*x|))/k
    to avoid overflow in exp(k*x). k is clamped to [5, 20] to prevent overflow.
    """
    k = max(SOFTPLUS_K_MIN, min(SOFTPLUS_K_MAX, float(k)))
    kx = k * x
    if kx >= 0:
        # ln(1 + e^{kx}) = kx + ln(1 + e^{-kx}); stable when kx large
        return x + np.log1p(np.exp(-kx)) / k
    else:
        # ln(1 + e^{kx}); e^{kx} in (0,1], no overflow
        return np.log1p(np.exp(kx)) / k
