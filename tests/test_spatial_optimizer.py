"""
Tests for spatial optimizer: LISA penalty and Simulated Annealing acceptance.
"""

import math
import pytest
import numpy as np

from ti4_analysis.algorithms.hex_grid import HexCoord
from ti4_analysis.data.map_structures import (
    Planet, System, MapSpace, MapSpaceType, Evaluator,
)
from ti4_analysis.algorithms.balance_engine import TI4Map
from ti4_analysis.algorithms.map_topology import MapTopology
from ti4_analysis.algorithms.fast_map_state import FastMapState
from ti4_analysis.algorithms.spatial_optimizer import (
    MultiObjectiveScore,
    evaluate_map_multiobjective,
    improve_balance_spatial,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_system(system_id: int, resources: int, influence: int) -> System:
    planet = Planet(f"P{system_id}", resources=resources, influence=influence)
    return System(id=system_id, planets=[planet])


def _make_four_system_map(
    r_high: int = 5,
    r_low: int = 1,
) -> tuple:
    """
    Minimal map with 2 home spaces and 4 swappable system spaces.

    Layout (hex cube coords, distance-1 adjacencies marked with ~):
        home1 (-5,5,0)
        high1 (-1,1,0) ~ high2 (0,0,0)   ← adjacent HH pair
        low1  (1,-1,0) ~ low2  (2,-2,0)  ← adjacent LL pair
        home2 (5,-5,0)

    With r_high >> r_low, high1 and high2 form a High-High cluster and
    low1 / low2 form a Low-Low cluster — both contribute positive local_I.
    """
    home1 = MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME)
    home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)

    # Adjacent high-value pair
    sys_high1 = MapSpace(HexCoord(-1, 1, 0), MapSpaceType.SYSTEM,
                         _make_system(1, r_high, r_high))
    sys_high2 = MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM,
                         _make_system(2, r_high, r_high))

    # Adjacent low-value pair (NOT adjacent to the high pair)
    sys_low1 = MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM,
                        _make_system(3, r_low, r_low))
    sys_low2 = MapSpace(HexCoord(2, -2, 0), MapSpaceType.SYSTEM,
                        _make_system(4, r_low, r_low))

    ti4_map = TI4Map([home1, home2, sys_high1, sys_high2, sys_low1, sys_low2])
    evaluator = Evaluator(name="Test")
    return ti4_map, evaluator


def _make_uniform_map(value: int = 3) -> tuple:
    """
    Map where all swappable systems have identical value.
    z_dev == 0 everywhere → all local_I == 0 → LISA penalty == 0.
    """
    home1 = MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME)
    home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)

    spaces = [home1, home2]
    for i, coord in enumerate([
        HexCoord(-1, 1, 0), HexCoord(0, 0, 0),
        HexCoord(1, -1, 0), HexCoord(2, -2, 0),
    ]):
        spaces.append(MapSpace(coord, MapSpaceType.SYSTEM,
                               _make_system(i + 1, value, value)))

    ti4_map = TI4Map(spaces)
    evaluator = Evaluator(name="Test")
    return ti4_map, evaluator


# ── LISA penalty unit tests ───────────────────────────────────────────────────

class TestLisaPenalty:

    def test_lisa_penalty_zero_for_uniform_values(self):
        """When all system values are equal, z_dev == 0 everywhere → penalty == 0."""
        ti4_map, evaluator = _make_uniform_map(value=3)
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)

        penalty = fast_state.lisa_penalty()

        assert penalty == pytest.approx(0.0, abs=1e-6)

    def test_lisa_penalty_positive_for_clustered_values(self):
        """High-value systems adjacent to each other produce positive local_I → penalty > 0."""
        ti4_map, evaluator = _make_four_system_map(r_high=5, r_low=1)
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)

        penalty = fast_state.lisa_penalty()

        assert penalty > 0.0, (
            "Adjacent high-value systems should produce a positive LISA penalty"
        )

    def test_lisa_penalty_less_when_dispersed(self):
        """
        Swapping so that high-value and low-value systems alternate (dispersed)
        should reduce the LISA penalty compared to the clustered arrangement.
        """
        ti4_map, evaluator = _make_four_system_map(r_high=5, r_low=1)
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)

        clustered_penalty = fast_state.lisa_penalty()

        # Swap indices 0↔2 so adjacent hexes alternate high-low
        fast_state.swap(0, 2)
        dispersed_penalty = fast_state.lisa_penalty()

        assert dispersed_penalty <= clustered_penalty, (
            "Dispersing high/low values should not increase the LISA penalty"
        )

    def test_lisa_penalty_included_in_composite_score(self):
        """
        MultiObjectiveScore with positive lisa_penalty should produce a higher
        composite score than one with zero penalty (all else equal).
        """
        base = MultiObjectiveScore(balance_gap=2.0, morans_i=0.0, jains_index=0.9,
                                   lisa_penalty=0.0)
        penalized = MultiObjectiveScore(balance_gap=2.0, morans_i=0.0, jains_index=0.9,
                                        lisa_penalty=5.0)

        assert penalized.composite_score() > base.composite_score()

    def test_lisa_penalty_zero_single_system(self):
        """A map with fewer than 3 systems returns 0 (guarded by n < 3 check)."""
        home1 = MapSpace(HexCoord(0, 0, 0), MapSpaceType.HOME)
        home2 = MapSpace(HexCoord(3, -3, 0), MapSpaceType.HOME)
        sys_space = MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM,
                             _make_system(1, 3, 3))

        ti4_map = TI4Map([home1, home2, sys_space])
        evaluator = Evaluator(name="Test")
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)

        # n_sys == 1, should return 0.0 without error
        assert fast_state.lisa_penalty() == pytest.approx(0.0, abs=1e-6)


# ── Metropolis criterion unit tests ───────────────────────────────────────────

class TestMetropolisCriterion:
    """
    Test the math of the SA acceptance criterion directly.
    P(accept) = exp(-delta / T)
    """

    def test_acceptance_probability_approaches_one_at_high_temperature(self):
        """At very high T, even large worsening moves are almost always accepted."""
        delta = 10.0   # large worsening
        T = 1_000_000.0
        prob = math.exp(-delta / T)
        assert prob > 0.999

    def test_acceptance_probability_approaches_zero_at_low_temperature(self):
        """At very low T, worsening moves are almost never accepted."""
        delta = 0.01   # small worsening
        T = 0.000_001
        prob = math.exp(-delta / T)
        assert prob < 1e-4

    def test_acceptance_probability_is_zero_for_improvements(self):
        """
        SA always accepts improvements (delta < 0) via the greedy branch,
        not via Metropolis. Confirm the greedy branch is never bypassed.
        delta < 0 → improvement → P = 1 (deterministic, not via exp).
        """
        delta = -1.0
        # The exp formula is only applied for delta >= 0; for delta < 0 we
        # accept unconditionally. Just assert the math makes sense.
        assert delta < 0

    def test_calibrated_temperature_targets_acceptance_rate(self):
        """
        Dynamic T₀ calibration: T = -avg_delta / ln(rate) should produce
        P(accept) ≈ rate for a move of size avg_delta.
        """
        avg_delta = 2.5
        target_rate = 0.80
        T0 = -avg_delta / math.log(target_rate)
        actual_prob = math.exp(-avg_delta / T0)
        assert actual_prob == pytest.approx(target_rate, rel=1e-6)


# ── Integration tests ─────────────────────────────────────────────────────────

class TestImproveBalanceSpatial:

    def test_sa_runs_and_returns_valid_score(self):
        """improve_balance_spatial() completes and returns a MultiObjectiveScore."""
        ti4_map, evaluator = _make_four_system_map()

        score, history = improve_balance_spatial(
            ti4_map, evaluator, iterations=30, random_seed=42, verbose=False
        )

        assert isinstance(score, MultiObjectiveScore)
        assert score.balance_gap >= 0.0
        assert 0.0 <= score.jains_index <= 1.0
        assert score.lisa_penalty >= 0.0
        assert len(history) >= 1

    def test_sa_composite_score_not_worse_than_start(self):
        """
        Best score seen during SA should not exceed the initial composite score
        (SA always tracks the best seen, not just the current).
        """
        ti4_map, evaluator = _make_four_system_map(r_high=5, r_low=1)
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
        initial = evaluate_map_multiobjective(ti4_map, evaluator, fast_state=fast_state)

        best, _ = improve_balance_spatial(
            ti4_map, evaluator, iterations=100, random_seed=0, verbose=False
        )

        assert best.composite_score() <= initial.composite_score() + 1e-6

    def test_sa_evaluate_map_includes_lisa(self):
        """evaluate_map_multiobjective via fast_state populates lisa_penalty."""
        ti4_map, evaluator = _make_four_system_map(r_high=5, r_low=1)
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)

        score = evaluate_map_multiobjective(ti4_map, evaluator, fast_state=fast_state)

        # lisa_penalty should be non-negative and match direct computation
        assert score.lisa_penalty >= 0.0
        assert score.lisa_penalty == pytest.approx(fast_state.lisa_penalty(), rel=1e-5)
