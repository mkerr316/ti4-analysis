"""
NSGA-II optimizer with BFS-PMX crossover for TI4 map balance.

Generates a diverse Pareto front of maps representing different trade-offs
across three objectives:

    1. balance_gap    (minimize) — scalar resource equality
    2. abs(morans_i)  (minimize) — global spatial clustering
    3. lisa_penalty   (minimize) — local H-H / L-L cluster penalty

jains_index is intentionally excluded from Pareto sorting: it is collinear
with balance_gap (both measure resource variance), and including a 4th
objective would collapse rank discrimination in Generation 0 (the
"many-objective curse" — most individuals become non-dominated).

SBPCG taxonomy (Togelius et al. 2011):
    Representation  : TI4Map tile permutation (genotype)
    Evaluation      : evaluate_map_multiobjective (theory-driven fitness)
    Search algorithm: NSGA-II with BFS-blob OX1 crossover + swap mutation

Key algorithmic choice — crossover domain:
    Crossover swaths are random BFS-connected blobs on topology.spatial_W
    (NOT radial wedges). This lets the algorithm discover map structures
    that bypass the human "slice" bias, allowing Moran's I and LISA math to
    dictate the spatial arrangement rather than inherited cultural convention.

Population seeding — inoculation strategy:
    10% warm starts (low-iteration greedy HC) anchor the Pareto front in
    high-fitness regions; 90% cold starts (random permutations) supply the
    genetic diversity required by the BFS crossover to splice novel topologies.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from .balance_engine import TI4Map, improve_balance
from .map_topology import MapTopology
from .fast_map_state import FastMapState
from ..data.map_structures import Evaluator
from .spatial_optimizer import MultiObjectiveScore, evaluate_map_multiobjective


# ── Individual ────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    """
    Single member of the NSGA-II population.

    map and fast_state must stay in sync: every system swap applied to one
    must be applied to the other. They share the immutable MapTopology.

    rank and crowding_distance are assigned by the NSGA-II sorting step and
    reset every generation.
    """
    map: TI4Map
    fast_state: FastMapState   # shares immutable topology
    score: MultiObjectiveScore
    rank: int = field(default=0)
    crowding_distance: float = field(default=0.0)


# ── Lookup pre-computation ────────────────────────────────────────────────────

def _build_lookups(topology: MapTopology):
    """
    Pre-compute index mappings for BFS blob generation.

    Returns:
        swappable_space_to_pos : dict  space_index → swappable_position_index
        spatial_row_to_s_pos   : dict  spatial_W_row → swappable_position_index
        s_pos_to_spatial_row   : dict  swappable_position_index → spatial_W_row
    """
    swappable_space_to_pos = {
        int(idx): pos for pos, idx in enumerate(topology.swappable_indices)
    }
    spatial_row_to_s_pos = {}
    for row, space_idx in enumerate(topology.spatial_indices):
        if int(space_idx) in swappable_space_to_pos:
            spatial_row_to_s_pos[row] = swappable_space_to_pos[int(space_idx)]
    s_pos_to_spatial_row = {v: k for k, v in spatial_row_to_s_pos.items()}
    return swappable_space_to_pos, spatial_row_to_s_pos, s_pos_to_spatial_row


# ── BFS blob generator ────────────────────────────────────────────────────────

def _bfs_blob(
    topology: MapTopology,
    s_pos_to_spatial_row: dict,
    spatial_row_to_s_pos: dict,
    rng: random.Random,
    blob_fraction: float = 0.5,
) -> frozenset:
    """
    Select a spatially contiguous subset of swappable positions via BFS.

    Traverses topology.spatial_W filtered to swappable positions only.
    blob_fraction=0.5 → ~S/2 positions (approximately half the board).

    The explicit visited set is required: spatial_W is a cyclic graph and
    naive repeated expansion would loop between two adjacent hexes forever.

    If the BFS exhausts a connected component before reaching the target
    size (isolated hex clusters exist on some maps), the smaller blob is
    returned — the OX1 crossover handles any blob size correctly.

    Returns:
        frozenset of swappable-position indices (s ∈ [0, S))
    """
    S = len(topology.swappable_indices)
    target = max(1, int(S * blob_fraction))

    start = rng.randrange(S)
    visited = {start}
    queue = deque([start])

    while queue and len(visited) < target:
        s = queue.popleft()
        if s not in s_pos_to_spatial_row:
            continue
        row = s_pos_to_spatial_row[s]
        # spatial_W[row] is a 1-row sparse matrix; .nonzero() → (rows, cols)
        _, neighbor_cols = topology.spatial_W[row].nonzero()
        for col in neighbor_cols:
            if col in spatial_row_to_s_pos:
                nb_s = spatial_row_to_s_pos[col]
                if nb_s not in visited:
                    visited.add(nb_s)
                    queue.append(nb_s)

    return frozenset(visited)


# ── OX1 crossover (permutation-safe) ─────────────────────────────────────────

def _ox1_crossover(parent_a_systems, parent_b_systems, blob_positions):
    """
    Order Crossover (OX1) adapted for TI4 tile permutation.

    Copies blob_positions verbatim from parent A, then fills remaining
    positions from parent B in their original relative order, skipping
    tiles already placed.

    Identity is determined by System.id (the domain tile number) — NOT
    Python's id() builtin, which breaks when System objects are deep-copied
    during warm-start greedy runs and would silently allow tile duplication.

    Args:
        parent_a_systems : list[System] indexed by swappable position [0..S-1]
        parent_b_systems : same for parent B
        blob_positions   : frozenset of position indices copied from parent A

    Returns:
        list[System] — valid permutation of the same tile multiset
    """
    S = len(parent_a_systems)
    offspring = [None] * S

    # Copy blob from parent A
    for s in blob_positions:
        offspring[s] = parent_a_systems[s]

    # Identify already-placed tiles by domain ID
    placed_ids = {sys.id for sys in offspring if sys is not None}

    # Fill remaining in OX1 order from parent B
    remaining = [s for s in range(S) if s not in blob_positions]
    b_iter = (
        sys for sys in parent_b_systems
        if sys is not None and sys.id not in placed_ids
    )
    for s in remaining:
        offspring[s] = next(b_iter)

    return offspring


# ── Offspring construction ────────────────────────────────────────────────────

def _build_offspring(parent_map, topology, offspring_systems, evaluator):
    """
    Materialise a crossover/mutation result into a (TI4Map, FastMapState) pair.

    TI4Map.copy() creates fresh MapSpace objects sharing the same System refs
    (consistent with OX1 which propagates System references). FastMapState is
    rebuilt from scratch to ensure system_value matches the new arrangement.
    """
    new_map = parent_map.copy()
    swappable_spaces = [new_map.spaces[idx] for idx in topology.swappable_indices]
    for s, system in enumerate(offspring_systems):
        swappable_spaces[s].system = system
    state = FastMapState.from_ti4_map(topology, new_map, evaluator)
    return new_map, state


# ── 3-objective Pareto dominance ──────────────────────────────────────────────

def _nsga2_dominates(a: Individual, b: Individual) -> bool:
    """
    Strict Pareto dominance over 3 objectives: balance_gap, |morans_i|, lisa_penalty.

    Returns True if a is better-or-equal on all objectives and strictly
    better on at least one.
    """
    a_gap = a.score.balance_gap
    a_mi  = abs(a.score.morans_i)
    a_lp  = a.score.lisa_penalty
    b_gap = b.score.balance_gap
    b_mi  = abs(b.score.morans_i)
    b_lp  = b.score.lisa_penalty

    all_leq = (a_gap <= b_gap) and (a_mi <= b_mi) and (a_lp <= b_lp)
    any_lt  = (a_gap < b_gap)  or  (a_mi < b_mi)  or  (a_lp < b_lp)
    return all_leq and any_lt


# ── NSGA-II sorting ───────────────────────────────────────────────────────────

def _fast_nondominated_sort(individuals: List[Individual]) -> List[List[Individual]]:
    """
    NSGA-II fast non-dominated sort (Deb et al. 2002).

    O(N² × 3) — assigns .rank in-place and returns list of fronts.
    Front index 0 = Pareto-optimal (rank 0).
    """
    n = len(individuals)
    domination_count = [0] * n           # n_p: how many individuals dominate p
    dominated_by = [[] for _ in range(n)]  # S_p: indices that p dominates

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _nsga2_dominates(individuals[i], individuals[j]):
                dominated_by[i].append(j)
            elif _nsga2_dominates(individuals[j], individuals[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            individuals[i].rank = 0

    # Seed first front
    current_front_indices = [i for i in range(n) if domination_count[i] == 0]
    fronts_indices = [current_front_indices]

    current_rank = 0
    while fronts_indices[current_rank]:
        next_front = []
        for i in fronts_indices[current_rank]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    individuals[j].rank = current_rank + 1
                    next_front.append(j)
        fronts_indices.append(next_front)
        current_rank += 1

    # Convert index fronts to Individual fronts, drop empty trailing front
    return [
        [individuals[i] for i in front]
        for front in fronts_indices
        if front
    ]


def _crowding_distance(front: List[Individual]) -> None:
    """
    Assign crowding distances in-place for individuals in one Pareto front.

    3 objectives: balance_gap (minimize), |morans_i| (minimize), lisa_penalty (minimize).
    Boundary individuals (extreme on any objective) receive float('inf') so the
    tournament selector always preserves them to maintain Pareto front spread.
    """
    n = len(front)
    for ind in front:
        ind.crowding_distance = 0.0

    if n <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return

    objective_getters = [
        lambda ind: ind.score.balance_gap,
        lambda ind: abs(ind.score.morans_i),
        lambda ind: ind.score.lisa_penalty,
    ]

    for getter in objective_getters:
        sorted_front = sorted(front, key=getter)
        obj_min = getter(sorted_front[0])
        obj_max = getter(sorted_front[-1])
        obj_range = obj_max - obj_min

        sorted_front[0].crowding_distance = float('inf')
        sorted_front[-1].crowding_distance = float('inf')

        if obj_range == 0.0:
            continue

        for i in range(1, n - 1):
            sorted_front[i].crowding_distance += (
                getter(sorted_front[i + 1]) - getter(sorted_front[i - 1])
            ) / obj_range


# ── Tournament selection ──────────────────────────────────────────────────────

def _tournament_select(pool: List[Individual], rng: random.Random) -> Individual:
    """
    Binary tournament: pick 2 random individuals, return the better one.

    Preference order: lower rank > higher crowding distance.
    """
    a, b = rng.sample(pool, 2)
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding_distance >= b.crowding_distance else b


# ── Population seeding (inoculation) ─────────────────────────────────────────

def _seed_population(
    ti4_map: TI4Map,
    topology: MapTopology,
    evaluator: Evaluator,
    pop_size: int,
    warm_fraction: float,
    rng: random.Random,
    verbose: bool,
) -> List[Individual]:
    """
    Inoculation seeding: warm_fraction warm starts + rest cold starts.

    Cold starts (random permutations) maximise diversity so NSGA-II can
    spread the Pareto front across the objective space.

    Warm starts (low-iteration greedy HC) provide anchor individuals in
    high-fitness regions, preventing Gen 0 from wasting cycles on maps
    that are mathematically catastrophic.
    """
    n_warm = int(pop_size * warm_fraction)  # 0 is valid → pure cold start
    n_cold = pop_size - n_warm
    population: List[Individual] = []

    base_systems = [ti4_map.spaces[i].system for i in topology.swappable_indices]

    # Cold starts: random shuffles of the tile pool
    if verbose:
        print(f"Seeding {n_cold} cold individuals (random permutations)...")
    for _ in range(n_cold):
        shuffled = base_systems.copy()
        rng.shuffle(shuffled)
        new_map, state = _build_offspring(ti4_map, topology, shuffled, evaluator)
        score = evaluate_map_multiobjective(new_map, evaluator, fast_state=state)
        population.append(Individual(map=new_map, fast_state=state, score=score))

    # Warm starts: short greedy HC runs as anchor individuals
    if verbose:
        print(f"Seeding {n_warm} warm individuals (greedy HC, 50 iterations each)...")
    for _ in range(n_warm):
        warm_map = ti4_map.copy()
        improve_balance(
            warm_map, evaluator,
            iterations=50,
            random_seed=rng.randrange(1_000_000),
        )
        state = FastMapState.from_ti4_map(topology, warm_map, evaluator)
        score = evaluate_map_multiobjective(warm_map, evaluator, fast_state=state)
        population.append(Individual(map=warm_map, fast_state=state, score=score))

    return population


# ── Main NSGA-II loop ─────────────────────────────────────────────────────────

def nsga2_optimize(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    generations: int = 200,
    population_size: int = 100,
    blob_fraction: float = 0.5,
    mutation_rate: float = 0.05,
    warm_fraction: float = 0.10,
    random_seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Tuple[TI4Map, MultiObjectiveScore]]:
    """
    NSGA-II with BFS-blob OX1 crossover for TI4 map Pareto optimisation.

    Optimises three objectives simultaneously (SBPCG direct evaluation):
        balance_gap   — max(home_values) - min(home_values)
        abs(morans_i) — global resource clustering
        lisa_penalty  — sum of positive local Moran's I (H-H / L-L hotspots)

    jains_index is computed and stored but excluded from Pareto sorting to
    avoid the many-objective curse (4+ objectives collapse rank discrimination).

    Args:
        ti4_map        : Starting map (defines tile pool and board topology)
        evaluator      : Balance evaluator (resource/influence weights)
        generations    : Number of evolutionary generations
        population_size: N individuals per generation (2N evaluated per gen)
        blob_fraction  : Fraction of swappable positions per BFS crossover blob
        mutation_rate  : Per-position swap probability after crossover
        warm_fraction  : Fraction of Gen 0 seeded with HC-optimised maps
        random_seed    : Reproducibility seed (None = non-deterministic)
        verbose        : Print per-generation progress

    Returns:
        List of (TI4Map, MultiObjectiveScore) tuples representing the Pareto front,
        sorted by balance_gap ascending.
    """
    rng = random.Random(random_seed)

    # Build topology once; all individuals share it.
    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    _, spatial_row_to_s_pos, s_pos_to_spatial_row = _build_lookups(topology)
    S = len(topology.swappable_indices)

    if S < 2:
        raise ValueError("Not enough swappable spaces for NSGA-II (need ≥ 2)")

    # ── Generation 0: seeding ────────────────────────────────────────────────
    population = _seed_population(
        ti4_map, topology, evaluator, population_size, warm_fraction, rng, verbose
    )

    fronts = _fast_nondominated_sort(population)
    for front in fronts:
        _crowding_distance(front)

    if verbose:
        n_rank0 = sum(1 for ind in population if ind.rank == 0)
        print(f"Gen 0 seed complete: {n_rank0}/{population_size} rank-0 individuals")

    # ── Evolutionary loop ────────────────────────────────────────────────────
    for gen in range(1, generations + 1):

        # Generate N offspring via selection → crossover → mutation
        offspring: List[Individual] = []
        while len(offspring) < population_size:
            pa = _tournament_select(population, rng)
            pb = _tournament_select(population, rng)

            pa_systems = [pa.map.spaces[i].system for i in topology.swappable_indices]
            pb_systems = [pb.map.spaces[i].system for i in topology.swappable_indices]

            blob = _bfs_blob(
                topology, s_pos_to_spatial_row, spatial_row_to_s_pos, rng, blob_fraction
            )
            child_systems = _ox1_crossover(pa_systems, pb_systems, blob)

            # Per-position swap mutation (independent Bernoulli trials)
            for s in range(S):
                if rng.random() < mutation_rate:
                    s2 = rng.randrange(S)
                    if s2 != s:
                        child_systems[s], child_systems[s2] = (
                            child_systems[s2], child_systems[s]
                        )

            child_map, child_state = _build_offspring(
                pa.map, topology, child_systems, evaluator
            )
            child_score = evaluate_map_multiobjective(
                child_map, evaluator, fast_state=child_state
            )
            offspring.append(Individual(
                map=child_map, fast_state=child_state, score=child_score
            ))

        # Combine 2N pool, sort, and truncate back to N
        combined = population + offspring
        fronts = _fast_nondominated_sort(combined)
        for front in fronts:
            _crowding_distance(front)

        # Fill new population front-by-front; sort the final partial front by crowding
        new_population: List[Individual] = []
        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend(front)
            else:
                needed = population_size - len(new_population)
                partial = sorted(front, key=lambda ind: -ind.crowding_distance)
                new_population.extend(partial[:needed])
                break

        population = new_population

        if verbose and (gen % 10 == 0 or gen == generations):
            rank0 = [ind for ind in population if ind.rank == 0]
            best_gap  = min(ind.score.balance_gap   for ind in rank0) if rank0 else float('inf')
            best_mi   = min(abs(ind.score.morans_i)  for ind in rank0) if rank0 else float('inf')
            best_lisa = min(ind.score.lisa_penalty   for ind in rank0) if rank0 else float('inf')
            print(
                f"Gen {gen:>{len(str(generations))}}/{generations}: "
                f"{len(rank0):>3} Pareto-front members | "
                f"best gap={best_gap:.2f} | "
                f"best |I|={best_mi:.3f} | "
                f"best LISA={best_lisa:.3f}"
            )

    # ── Return Pareto front sorted by balance_gap ────────────────────────────
    pareto_front = [ind for ind in population if ind.rank == 0]
    pareto_front.sort(key=lambda ind: ind.score.balance_gap)

    if verbose:
        print(f"\nFinal Pareto front: {len(pareto_front)} solutions")
        for i, ind in enumerate(pareto_front[:5]):
            print(f"  Solution {i + 1}: {ind.score}")

    return [(ind.map, ind.score) for ind in pareto_front]
