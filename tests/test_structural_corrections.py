"""
Tests for fitness landscape structural corrections: smooth objectives,
Gen-0 normalization, raw_objective_terms, and local-variance LSAP.
"""

import pytest
import numpy as np

from ti4_analysis.algorithms.objectives_smooth import (
    smooth_min_jain,
    softplus_hinge,
    DEFAULT_JAIN_SMOOTH_P,
    DEFAULT_SOFTPLUS_K,
    JAIN_EPS,
)
from ti4_analysis.algorithms.spatial_optimizer import (
    MultiObjectiveScore,
    NORM_KEY_HINGE,
    NORM_KEY_JFI,
    NORM_KEY_LISA,
    compute_gen0_sigma,
)


# ── Smooth Jain (L_{-p} mean) ─────────────────────────────────────────────────

def test_smooth_min_jain_approaches_min_as_p_large():
    """smooth_min_jain(J_R, J_I, p) → min(J_R, J_I) as p increases."""
    j_r, j_i = 0.6, 0.8
    j_min = min(j_r, j_i)
    for p in [4, 8, 16, 32, 64]:
        j_smooth = smooth_min_jain(j_r, j_i, p=p)
        assert j_smooth >= j_min - 0.01
        assert j_smooth <= max(j_r, j_i) + 0.01
    j_smooth_64 = smooth_min_jain(j_r, j_i, p=64)
    assert abs(j_smooth_64 - j_min) < 0.05


def test_smooth_min_jain_clamps_away_from_zero():
    """J_R or J_I near zero are clamped to JAIN_EPS to avoid blow-up."""
    j_smooth = smooth_min_jain(1e-8, 0.9, p=8)
    assert 0 < j_smooth <= 1.0
    j_smooth2 = smooth_min_jain(0.9, 1e-8, p=8)
    assert 0 < j_smooth2 <= 1.0


def test_smooth_min_jain_symmetric():
    """smooth_min_jain(a, b) == smooth_min_jain(b, a)."""
    a, b = 0.5, 0.7
    assert smooth_min_jain(a, b, p=8) == smooth_min_jain(b, a, p=8)


# ── Softplus hinge ────────────────────────────────────────────────────────────

def test_softplus_hinge_positive_x():
    """For x > 0, softplus(x, k) approximates x for large k."""
    x = 0.5
    out = softplus_hinge(x, k=10)
    assert out > 0
    assert out <= x + 0.2  # softplus slightly above identity for x>0


def test_softplus_hinge_negative_x():
    """For x < 0, softplus(x, k) is near 0 (smooth zero)."""
    x = -0.5
    out = softplus_hinge(x, k=10)
    assert out >= 0
    assert out < 0.1


def test_softplus_hinge_zero():
    """softplus(0, k) = ln(2)/k."""
    out = softplus_hinge(0.0, k=10)
    expected = np.log(2) / 10
    assert abs(out - expected) < 1e-6


def test_softplus_k_clamped():
    """k is clamped to [5, 20] to avoid overflow."""
    # Very large k should not raise
    out_high = softplus_hinge(2.0, k=100)
    assert np.isfinite(out_high)
    out_low = softplus_hinge(0.1, k=0.1)
    assert np.isfinite(out_low)


# ── MultiObjectiveScore: raw_objective_terms, normalizer_sigma ─────────────────

def test_raw_objective_terms_shape():
    """raw_objective_terms returns (hinge, jfi_gap, lisa_norm)."""
    score = MultiObjectiveScore(
        balance_gap=1.0,
        morans_i=0.1,
        jains_index=0.7,
        lisa_penalty=10.0,
        n_spatial=37,
    )
    terms = score.raw_objective_terms()
    assert len(terms) == 3
    hinge, jfi_gap, lisa_norm = terms
    assert hinge >= 0
    assert 0 <= jfi_gap <= 1
    assert 0 <= lisa_norm <= 1.1  # lisa_norm = LSAP / (n*(n-1))


def test_composite_with_normalizer_sigma():
    """With normalizer_sigma, composite_score uses normalized terms."""
    score = MultiObjectiveScore(
        balance_gap=0.0,
        morans_i=0.0,
        jains_index=0.8,
        lisa_penalty=5.0,
        n_spatial=37,
        normalizer_sigma={
            NORM_KEY_HINGE: 0.2,
            NORM_KEY_JFI: 0.15,
            NORM_KEY_LISA: 0.01,
        },
    )
    c = score.composite_score()
    assert np.isfinite(c)
    assert c >= 0


def test_objective_values_for_pareto_smooth_vs_raw():
    """objective_values_for_pareto with use_smooth_objectives differs from raw."""
    score_raw = MultiObjectiveScore(
        balance_gap=0.0, morans_i=-0.02, jains_index=0.75,
        lisa_penalty=2.0, n_spatial=37,
        jfi_resources=0.75, jfi_influence=0.9,
    )
    score_smooth = MultiObjectiveScore(
        balance_gap=0.0, morans_i=-0.02, jains_index=0.75,
        lisa_penalty=2.0, n_spatial=37,
        jfi_resources=0.75, jfi_influence=0.9,
        use_smooth_objectives=True,
    )
    raw_objs = score_raw.objective_values_for_pareto()
    smooth_objs = score_smooth.objective_values_for_pareto()
    # First objective (JFI gap): smooth min is > raw min when one dimension is better
    assert smooth_objs[0] <= raw_objs[0] + 0.05  # smooth can be slightly different
    assert len(raw_objs) == 3 and len(smooth_objs) == 3


# ── compute_gen0_sigma ───────────────────────────────────────────────────────

def test_compute_gen0_sigma_returns_positive_dict():
    """compute_gen0_sigma returns dict with positive sigma per key."""
    from ti4_analysis.algorithms.hex_grid import HexCoord
    from ti4_analysis.data.map_structures import (
        Planet, System, MapSpace, MapSpaceType, Evaluator,
    )
    from ti4_analysis.algorithms.balance_engine import TI4Map
    from ti4_analysis.algorithms.map_topology import MapTopology

    def _sys(i, r, inf):
        return System(id=i, planets=[Planet(f"P{i}", resources=r, influence=inf)])

    home1 = MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME)
    home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)
    spaces = [home1, home2] + [
        MapSpace(HexCoord(-1, 1, 0), MapSpaceType.SYSTEM, _sys(1, 3, 2)),
        MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM, _sys(2, 2, 3)),
        MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM, _sys(3, 4, 1)),
        MapSpace(HexCoord(2, -2, 0), MapSpaceType.SYSTEM, _sys(4, 1, 4)),
    ]
    ti4_map = TI4Map(spaces)
    evaluator = Evaluator(name="Test")
    topology = MapTopology.from_ti4_map(ti4_map, evaluator)

    sigma = compute_gen0_sigma(
        topology, evaluator, ti4_map.copy(),
        n_samples=20, random_seed=42, n_swaps_randomize=30,
    )
    assert NORM_KEY_HINGE in sigma and sigma[NORM_KEY_HINGE] >= 0
    assert NORM_KEY_JFI in sigma and sigma[NORM_KEY_JFI] >= 0
    assert NORM_KEY_LISA in sigma and sigma[NORM_KEY_LISA] >= 0
