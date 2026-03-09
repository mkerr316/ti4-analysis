"""
Tests for NSGA-II optimizer: BFS blob, OX1 crossover, sorting, and integration.
"""

import random
import pytest
import numpy as np

from ti4_analysis.algorithms.hex_grid import HexCoord
from ti4_analysis.data.map_structures import (
    Planet, System, MapSpace, MapSpaceType, Evaluator,
)
from ti4_analysis.algorithms.balance_engine import TI4Map
from ti4_analysis.algorithms.map_topology import MapTopology
from ti4_analysis.algorithms.fast_map_state import FastMapState
from ti4_analysis.algorithms.spatial_optimizer import evaluate_map_multiobjective
from ti4_analysis.algorithms.nsga2_optimizer import (
    Individual,
    _build_lookups,
    _bfs_blob,
    _ox1_crossover,
    _build_offspring,
    _nsga2_dominates,
    _fast_nondominated_sort,
    _crowding_distance,
    nsga2_optimize,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_system(system_id: int, resources: int, influence: int) -> System:
    return System(id=system_id, planets=[Planet(f"P{system_id}", resources, influence)])


def _make_six_system_map() -> tuple:
    """
    Map with 2 HOME spaces and 6 swappable SYSTEM spaces arranged in a line.

    Layout (cube coords):
        home1 (-5,5,0)
        sys1  (-2,2,0) ~ sys2(-1,1,0) ~ sys3(0,0,0) ~ sys4(1,-1,0) ~ sys5(2,-2,0)
        sys6  (3,-3,0)
        home2 (5,-5,0)

    6 distinct system IDs guarantee the OX1 crossover permutation tests work
    without tile collision.
    """
    home1 = MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME)
    home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)
    systems = [
        MapSpace(HexCoord(-2,  2, 0), MapSpaceType.SYSTEM, _make_system(1, 3, 1)),
        MapSpace(HexCoord(-1,  1, 0), MapSpaceType.SYSTEM, _make_system(2, 1, 3)),
        MapSpace(HexCoord( 0,  0, 0), MapSpaceType.SYSTEM, _make_system(3, 4, 2)),
        MapSpace(HexCoord( 1, -1, 0), MapSpaceType.SYSTEM, _make_system(4, 2, 4)),
        MapSpace(HexCoord( 2, -2, 0), MapSpaceType.SYSTEM, _make_system(5, 5, 1)),
        MapSpace(HexCoord( 3, -3, 0), MapSpaceType.SYSTEM, _make_system(6, 1, 5)),
    ]
    ti4_map = TI4Map([home1, home2] + systems)
    evaluator = Evaluator(name="Test")
    return ti4_map, evaluator


# ── BFS blob tests ────────────────────────────────────────────────────────────

class TestBfsBlob:

    def _setup(self):
        ti4_map, evaluator = _make_six_system_map()
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        _, spatial_row_to_s_pos, s_pos_to_spatial_row = _build_lookups(topology)
        rng = random.Random(0)
        return topology, spatial_row_to_s_pos, s_pos_to_spatial_row, rng

    def test_blob_size_approximately_half(self):
        """BFS blob should select roughly blob_fraction of swappable positions."""
        topology, r2s, s2r, rng = self._setup()
        S = len(topology.swappable_indices)
        blob = _bfs_blob(topology, s2r, r2s, rng, blob_fraction=0.5)
        # Allow ±1 for edge effects / connectivity limits
        assert abs(len(blob) - S // 2) <= 2

    def test_blob_positions_in_valid_range(self):
        """All blob positions must be valid swappable-position indices."""
        topology, r2s, s2r, rng = self._setup()
        S = len(topology.swappable_indices)
        for _ in range(10):
            blob = _bfs_blob(topology, s2r, r2s, rng, blob_fraction=0.5)
            assert all(0 <= s < S for s in blob)

    def test_blob_is_spatially_contiguous(self):
        """
        Every position in the blob must be reachable from every other via
        adjacency through blob positions only (i.e., the blob is connected).
        """
        topology, r2s, s2r, rng = self._setup()
        blob = _bfs_blob(topology, s2r, r2s, rng, blob_fraction=0.6)

        if len(blob) <= 1:
            return  # trivially connected

        # Build adjacency among blob positions only
        blob_list = list(blob)
        adj = {s: set() for s in blob_list}
        for s in blob_list:
            if s not in s2r:
                continue
            row = s2r[s]
            _, cols = topology.spatial_W[row].nonzero()
            for col in cols:
                if col in r2s and r2s[col] in blob:
                    adj[s].add(r2s[col])

        # BFS from first blob member — should reach all others
        start = blob_list[0]
        visited = {start}
        queue = [start]
        while queue:
            cur = queue.pop()
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        assert visited == set(blob_list), (
            "BFS blob is not spatially contiguous — disconnected positions found"
        )

    def test_blob_no_infinite_loop(self):
        """BFS terminates in bounded time (visited set prevents cycling)."""
        topology, r2s, s2r, rng = self._setup()
        # If visited set is broken, this would hang; pytest timeout catches it
        for _ in range(50):
            blob = _bfs_blob(topology, s2r, r2s, rng, blob_fraction=0.5)
            assert len(blob) > 0


# ── OX1 crossover tests ───────────────────────────────────────────────────────

class TestOx1Crossover:

    def _get_systems(self):
        ti4_map, evaluator = _make_six_system_map()
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        systems = [ti4_map.spaces[i].system for i in topology.swappable_indices]
        return systems, topology

    def test_offspring_is_valid_permutation_of_parent_a(self):
        """Offspring tile IDs must be identical multiset as parent A's tiles."""
        systems, topology = self._get_systems()
        parent_a = systems
        parent_b = list(reversed(systems))
        blob = frozenset(range(len(systems) // 2))

        offspring = _ox1_crossover(parent_a, parent_b, blob)

        assert sorted(s.id for s in offspring) == sorted(s.id for s in parent_a)

    def test_offspring_contains_no_duplicates(self):
        """No tile may appear twice in the offspring."""
        systems, topology = self._get_systems()
        parent_a = systems
        parent_b = list(reversed(systems))
        blob = frozenset([0, 2, 4])

        offspring = _ox1_crossover(parent_a, parent_b, blob)

        ids = [s.id for s in offspring]
        assert len(ids) == len(set(ids)), f"Duplicate tiles found: {ids}"

    def test_blob_positions_taken_from_parent_a(self):
        """Positions inside the blob must have parent A's system IDs."""
        systems, topology = self._get_systems()
        parent_a = systems
        parent_b = list(reversed(systems))
        blob = frozenset([0, 1, 2])

        offspring = _ox1_crossover(parent_a, parent_b, blob)

        for s in blob:
            assert offspring[s].id == parent_a[s].id, (
                f"Position {s}: expected parent A's tile {parent_a[s].id}, "
                f"got {offspring[s].id}"
            )

    def test_crossover_full_blob_equals_parent_a(self):
        """blob_positions = all positions → offspring identical to parent A."""
        systems, _ = self._get_systems()
        S = len(systems)
        blob = frozenset(range(S))
        offspring = _ox1_crossover(systems, list(reversed(systems)), blob)
        assert [s.id for s in offspring] == [s.id for s in systems]

    def test_crossover_empty_blob_equals_parent_b(self):
        """blob_positions = {} → offspring identical to parent B."""
        systems, _ = self._get_systems()
        parent_b = list(reversed(systems))
        blob = frozenset()
        offspring = _ox1_crossover(systems, parent_b, blob)
        assert [s.id for s in offspring] == [s.id for s in parent_b]


# ── NSGA-II sorting tests ─────────────────────────────────────────────────────

def _make_individual(gap, morans, lisa):
    """Construct a minimal Individual with the given objective values."""
    score = object.__new__(type('MockScore', (), {
        'balance_gap': gap,
        'morans_i': morans,
        'lisa_penalty': lisa,
        'jains_index': 0.9,
    }))
    score.balance_gap = gap
    score.morans_i = morans
    score.lisa_penalty = lisa
    score.jains_index = 0.9
    ind = object.__new__(Individual)
    ind.score = score
    ind.map = None
    ind.fast_state = None
    ind.rank = 0
    ind.crowding_distance = 0.0
    return ind


class TestNsga2Dominance:

    def test_clearly_better_dominates(self):
        """Individual better on all 3 objectives dominates the other."""
        a = _make_individual(gap=1.0, morans=0.1, lisa=0.5)
        b = _make_individual(gap=2.0, morans=0.3, lisa=1.0)
        assert _nsga2_dominates(a, b)
        assert not _nsga2_dominates(b, a)

    def test_no_domination_when_tradeoff(self):
        """Neither dominates when one is better on one objective, worse on another."""
        a = _make_individual(gap=1.0, morans=0.5, lisa=1.0)
        b = _make_individual(gap=2.0, morans=0.1, lisa=0.5)
        assert not _nsga2_dominates(a, b)
        assert not _nsga2_dominates(b, a)

    def test_identical_does_not_dominate(self):
        """An individual does not dominate an equal individual."""
        a = _make_individual(gap=1.0, morans=0.2, lisa=0.5)
        b = _make_individual(gap=1.0, morans=0.2, lisa=0.5)
        assert not _nsga2_dominates(a, b)


class TestNonDominatedSort:

    def test_rank0_individuals_not_dominated(self):
        """Every rank-0 individual must be genuinely non-dominated."""
        individuals = [
            _make_individual(1.0, 0.5, 0.5),  # Pareto: good gap
            _make_individual(2.0, 0.1, 0.3),  # Pareto: good morans
            _make_individual(3.0, 0.6, 0.1),  # Pareto: good lisa
            _make_individual(4.0, 0.8, 0.9),  # Dominated by all above
        ]
        fronts = _fast_nondominated_sort(individuals)
        rank0 = fronts[0]
        for ind in rank0:
            for other in individuals:
                if other is ind:
                    continue
                assert not _nsga2_dominates(other, ind), (
                    f"rank-0 individual with gap={ind.score.balance_gap} "
                    f"is dominated by gap={other.score.balance_gap}"
                )

    def test_dominated_individual_not_in_rank0(self):
        """The clearly dominated individual must be in a later front."""
        dominated = _make_individual(gap=4.0, morans=0.8, lisa=0.9)
        others = [
            _make_individual(1.0, 0.1, 0.1),
        ]
        fronts = _fast_nondominated_sort(others + [dominated])
        rank0_ids = {id(ind) for ind in fronts[0]}
        assert id(dominated) not in rank0_ids

    def test_all_non_dominated_all_rank0(self):
        """When no individual dominates any other, all must be rank-0."""
        individuals = [
            _make_individual(1.0, 0.5, 0.9),
            _make_individual(2.0, 0.1, 0.7),
            _make_individual(3.0, 0.4, 0.1),
        ]
        fronts = _fast_nondominated_sort(individuals)
        assert len(fronts[0]) == 3


class TestCrowdingDistance:

    def test_boundary_individuals_get_infinity(self):
        """Individuals at the extremes of each objective get crowding distance ∞."""
        front = [
            _make_individual(1.0, 0.5, 0.5),
            _make_individual(2.0, 0.3, 0.7),
            _make_individual(3.0, 0.1, 0.9),
        ]
        _crowding_distance(front)
        # The individuals with min/max on any objective must have inf distance
        # (gap: 1.0 is min, 3.0 is max → both get inf)
        gaps = [ind.score.balance_gap for ind in front]
        min_gap_ind = front[gaps.index(min(gaps))]
        max_gap_ind = front[gaps.index(max(gaps))]
        assert min_gap_ind.crowding_distance == float('inf')
        assert max_gap_ind.crowding_distance == float('inf')

    def test_two_or_fewer_all_infinity(self):
        """Front of size ≤ 2 — all individuals get ∞ crowding distance."""
        front = [_make_individual(1.0, 0.2, 0.5), _make_individual(2.0, 0.1, 0.8)]
        _crowding_distance(front)
        for ind in front:
            assert ind.crowding_distance == float('inf')

    def test_middle_individual_finite_distance(self):
        """The middle individual in a 3-member front gets a finite distance."""
        front = [
            _make_individual(1.0, 0.9, 0.9),
            _make_individual(2.0, 0.5, 0.5),
            _make_individual(3.0, 0.1, 0.1),
        ]
        _crowding_distance(front)
        distances = [ind.crowding_distance for ind in front]
        middle = [d for d in distances if d != float('inf')]
        assert len(middle) >= 1
        assert all(np.isfinite(d) for d in middle)


# ── Integration test ──────────────────────────────────────────────────────────

class TestNsga2Optimize:

    def test_nsga2_returns_pareto_front(self):
        """nsga2_optimize completes and returns nonempty list of (TI4Map, score) tuples."""
        ti4_map, evaluator = _make_six_system_map()
        result = nsga2_optimize(
            ti4_map, evaluator,
            generations=3,
            population_size=10,
            random_seed=42,
            verbose=False,
        )
        assert len(result) > 0
        for map_obj, score in result:
            assert isinstance(map_obj, TI4Map)
            assert score.balance_gap >= 0.0
            assert score.lisa_penalty >= 0.0

    def test_pareto_front_members_are_non_dominated(self):
        """No returned solution should be dominated by any other returned solution."""
        ti4_map, evaluator = _make_six_system_map()
        result = nsga2_optimize(
            ti4_map, evaluator,
            generations=5,
            population_size=12,
            random_seed=7,
            verbose=False,
        )
        # Reconstruct Individuals for dominance checking
        individuals = []
        for m, s in result:
            ind = _make_individual(s.balance_gap, s.morans_i, s.lisa_penalty)
            individuals.append(ind)

        for i, a in enumerate(individuals):
            for j, b in enumerate(individuals):
                if i != j:
                    assert not _nsga2_dominates(b, a), (
                        f"Solution {i} (gap={a.score.balance_gap:.2f}) "
                        f"is dominated by solution {j} (gap={b.score.balance_gap:.2f})"
                    )

    def test_warm_seeds_not_worse_than_cold_on_average(self):
        """
        Warm-seeded individuals (HC-optimised) should have lower average balance_gap
        than cold-seeded ones (random permutations) — validate inoculation value.
        """
        from ti4_analysis.algorithms.nsga2_optimizer import _seed_population, _build_lookups

        ti4_map, evaluator = _make_six_system_map()
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        rng = random.Random(0)

        pop = _seed_population(
            ti4_map, topology, evaluator,
            pop_size=20, warm_fraction=0.5,
            rng=rng, verbose=False,
        )

        # First 10 are cold starts, last 10 are warm starts
        cold = pop[:10]
        warm = pop[10:]

        avg_cold_gap = sum(ind.score.balance_gap for ind in cold) / len(cold)
        avg_warm_gap = sum(ind.score.balance_gap for ind in warm) / len(warm)

        assert avg_warm_gap <= avg_cold_gap, (
            f"Warm seeds (avg gap={avg_warm_gap:.2f}) should be ≤ "
            f"cold seeds (avg gap={avg_cold_gap:.2f})"
        )
