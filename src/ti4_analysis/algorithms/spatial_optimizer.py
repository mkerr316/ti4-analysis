"""
Multi-objective spatial optimizer for TI4 maps.

Implements Pareto optimization that balances BOTH basic balance metrics
(balance_gap) AND spatial distribution metrics (Moran's I, Jain's Index).

This addresses the "spatial blindness" problem identified in the experimental research.
"""

import random
from typing import Tuple, List, Optional, Dict
import numpy as np

from .balance_engine import TI4Map, get_home_values, get_balance_gap, can_swap_system
from .map_topology import MapTopology
from .fast_map_state import FastMapState
from ..data.map_structures import Evaluator, MapSpace
from ..spatial_stats.spatial_metrics import comprehensive_spatial_analysis


class MultiObjectiveScore:
    """
    Container for multi-objective fitness scores.

    A map's quality is evaluated across multiple dimensions:
    - Balance gap (lower is better)
    - Spatial clustering / Moran's I (lower absolute value is better)
    - Accessibility fairness / Jain's Index (higher is better)
    """

    def __init__(
        self,
        balance_gap: float,
        morans_i: float,
        jains_index: float,
        weights: Optional[Dict[str, float]] = None
    ):
        self.balance_gap = balance_gap
        self.morans_i = morans_i
        self.jains_index = jains_index

        # Default weights
        if weights is None:
            weights = {
                'balance_gap': 1.0,
                'morans_i': 0.5,
                'jains_index': -0.5  # Negative because we want to maximize Jain's
            }

        self.weights = weights

    def composite_score(self) -> float:
        """
        Calculate weighted composite score.

        Lower is better (minimize this).
        """
        score = (
            self.weights['balance_gap'] * self.balance_gap +
            self.weights['morans_i'] * abs(self.morans_i) +
            self.weights['jains_index'] * (1.0 - self.jains_index)  # Convert max to min
        )
        return score

    def dominates(self, other: 'MultiObjectiveScore') -> bool:
        """
        Check if this score Pareto-dominates another score.

        Dominates if it's better or equal on all objectives, and strictly better on at least one.
        """
        better_gap = self.balance_gap <= other.balance_gap
        better_morans = abs(self.morans_i) <= abs(other.morans_i)
        better_jains = self.jains_index >= other.jains_index

        strictly_better_gap = self.balance_gap < other.balance_gap
        strictly_better_morans = abs(self.morans_i) < abs(other.morans_i)
        strictly_better_jains = self.jains_index > other.jains_index

        all_better_or_equal = better_gap and better_morans and better_jains
        at_least_one_strictly_better = strictly_better_gap or strictly_better_morans or strictly_better_jains

        return all_better_or_equal and at_least_one_strictly_better

    def __str__(self) -> str:
        return (
            f"Gap={self.balance_gap:.2f}, "
            f"Moran's I={self.morans_i:+.3f}, "
            f"Jain's={self.jains_index:.3f}, "
            f"Composite={self.composite_score():.2f}"
        )


def evaluate_map_multiobjective(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    weights: Optional[Dict[str, float]] = None,
    fast_state: Optional[FastMapState] = None,
) -> MultiObjectiveScore:
    """
    Evaluate a map across all objectives.

    Args:
        ti4_map: Map to evaluate (must reflect current tile placement for spatial metrics)
        evaluator: Balance evaluator
        weights: Objective weights (uses defaults if None)
        fast_state: Optional pre-initialized FastMapState. When provided, all three
            metrics (balance_gap, Moran's I, Jain's Index) are computed via vectorized
            fast paths — ti4_map is not read. When None, falls back to comprehensive_spatial_analysis().

    Returns:
        MultiObjectiveScore with all metrics
    """
    if fast_state is not None:
        # All three metrics from vectorized fast paths — no TI4Map traversal needed.
        balance_gap = fast_state.balance_gap()
        morans_i = fast_state.morans_i()
        jains_index = fast_state.jains_index()
    else:
        home_values = get_home_values(ti4_map, evaluator)
        balance_gap = get_balance_gap(home_values)
        spatial_analysis = comprehensive_spatial_analysis(ti4_map, evaluator)
        morans_i = spatial_analysis['resource_clustering_morans_i']
        jains_index = spatial_analysis['jains_fairness_index']

    return MultiObjectiveScore(balance_gap, morans_i, jains_index, weights)


def improve_balance_spatial(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    iterations: int = 200,
    weights: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[MultiObjectiveScore, List[Tuple[int, MultiObjectiveScore]]]:
    """
    Improve map balance using multi-objective optimization.

    Uses hill-climbing with a weighted composite fitness function that
    considers both basic balance and spatial distribution.

    Args:
        ti4_map: Map to optimize (modified in-place)
        evaluator: Balance evaluator
        iterations: Maximum number of iterations
        weights: Objective weights (balance_gap, morans_i, jains_index)
        random_seed: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        Tuple of (final_score, history)
        history is a list of (iteration, score) tuples
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Build static topology once; balance_gap evaluations use fast matmul.
    # swappable_spaces[s] corresponds to fast_state.system_value[s].
    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
    swappable_spaces = [ti4_map.spaces[i] for i in topology.swappable_indices]

    if len(swappable_spaces) < 2:
        raise ValueError("Not enough swappable spaces to optimize")

    # Initial evaluation
    current_score = evaluate_map_multiobjective(ti4_map, evaluator, weights, fast_state)
    best_score = current_score
    history = [(0, current_score)]

    if verbose:
        print(f"Initial: {current_score}")

    no_improvement_count = 0
    max_no_improvement = 50  # Early stopping

    for i in range(1, iterations + 1):
        # Pick two random swappable-position indices
        s1, s2 = random.sample(range(len(swappable_spaces)), 2)
        space1 = swappable_spaces[s1]
        space2 = swappable_spaces[s2]

        # Apply swap to both fast state and ti4_map (spatial metrics read ti4_map)
        fast_state.swap(s1, s2)
        space1.system, space2.system = space2.system, space1.system

        # Evaluate new configuration
        new_score = evaluate_map_multiobjective(ti4_map, evaluator, weights, fast_state)

        # Accept if improved (greedy hill-climbing)
        if new_score.composite_score() < current_score.composite_score():
            current_score = new_score
            no_improvement_count = 0

            if new_score.composite_score() < best_score.composite_score():
                best_score = new_score

                if verbose and i % 10 == 0:
                    print(f"Iteration {i}: {new_score}")

        else:
            # Revert both fast state and ti4_map
            fast_state.swap(s1, s2)
            space1.system, space2.system = space2.system, space1.system
            no_improvement_count += 1

        # Record history every 10 iterations
        if i % 10 == 0:
            history.append((i, current_score))

        # Early stopping if no improvement
        if no_improvement_count >= max_no_improvement:
            if verbose:
                print(f"Early stopping at iteration {i} (no improvement for {max_no_improvement} iterations)")
            break

    if verbose:
        print(f"Final: {current_score}")

    history.append((i, current_score))

    return current_score, history


def pareto_optimize(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    iterations: int = 500,
    population_size: int = 10,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> List[Tuple[TI4Map, MultiObjectiveScore]]:
    """
    Find Pareto-optimal set of maps using multi-objective evolutionary algorithm.

    This maintains a population of diverse solutions representing different
    trade-offs between balance and spatial objectives.

    WARNING: This is computationally expensive! Each map evaluation requires
    calculating spatial metrics, which is O(n²) in the number of hexes.

    Args:
        ti4_map: Starting map
        evaluator: Balance evaluator
        iterations: Number of generations
        population_size: Size of population
        random_seed: Random seed
        verbose: Print progress

    Returns:
        List of (map, score) tuples representing the Pareto front
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Build topology once; all individuals share it.
    topology = MapTopology.from_ti4_map(ti4_map, evaluator)

    # Initialize population with variations of the starting map.
    # Each individual is (TI4Map, FastMapState, score) — evaluation uses fast_state.
    population: List[Tuple[TI4Map, FastMapState, MultiObjectiveScore]] = []

    base_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
    swappable_indices_list = list(range(len(topology.swappable_indices)))

    for i in range(population_size):
        map_copy = ti4_map.copy()
        fast_copy = base_state.clone()

        # Randomize by applying 10 random swaps to both in sync.
        swappable_spaces = [map_copy.spaces[idx] for idx in topology.swappable_indices]
        for _ in range(10):
            s1, s2 = random.sample(swappable_indices_list, 2)
            fast_copy.swap(s1, s2)
            swappable_spaces[s1].system, swappable_spaces[s2].system = (
                swappable_spaces[s2].system, swappable_spaces[s1].system
            )

        score = evaluate_map_multiobjective(map_copy, evaluator, fast_state=fast_copy)
        population.append((map_copy, fast_copy, score))

        if verbose:
            print(f"Initialized map {i+1}/{population_size}: {score}")

    # Evolve population
    for generation in range(iterations):
        if verbose and generation % 10 == 0:
            print(f"\nGeneration {generation}/{iterations}")

        # Create offspring by mutation
        offspring: List[Tuple[TI4Map, FastMapState, MultiObjectiveScore]] = []

        for map_obj, fast_obj, score in population:
            child_map = map_obj.copy()
            child_state = fast_obj.clone()
            child_swappable = [child_map.spaces[idx] for idx in topology.swappable_indices]

            num_swaps = random.randint(1, 3)
            for _ in range(num_swaps):
                s1, s2 = random.sample(swappable_indices_list, 2)
                child_state.swap(s1, s2)
                child_swappable[s1].system, child_swappable[s2].system = (
                    child_swappable[s2].system, child_swappable[s1].system
                )

            child_score = evaluate_map_multiobjective(child_map, evaluator, fast_state=child_state)
            offspring.append((child_map, child_state, child_score))

        # Combine and extract Pareto front
        combined = population + offspring
        pareto_front = _extract_pareto_front_triples(combined)
        population = pareto_front[:population_size]

        if verbose and generation % 10 == 0:
            print(f"Pareto front size: {len(pareto_front)}")
            if len(population) > 0:
                print(f"Best composite score: {min(s.composite_score() for _, _, s in population):.2f}")

    # Return final Pareto front — strip FastMapState from tuples (API unchanged)
    final_front_triples = _extract_pareto_front_triples(population)

    if verbose:
        print(f"\nFinal Pareto front: {len(final_front_triples)} solutions")
        for i, (_, _, score) in enumerate(final_front_triples[:5]):
            print(f"  Solution {i+1}: {score}")

    return [(m, s) for m, _, s in final_front_triples]


def _extract_pareto_front_triples(
    population: List[Tuple[TI4Map, 'FastMapState', MultiObjectiveScore]]
) -> List[Tuple[TI4Map, 'FastMapState', MultiObjectiveScore]]:
    """Extract Pareto-optimal front from (map, fast_state, score) triples."""
    pareto_front = []
    for item in population:
        score = item[2]
        if not any(other[2].dominates(score) for other in population):
            pareto_front.append(item)
    return pareto_front


def _extract_pareto_front(
    population: List[Tuple[TI4Map, MultiObjectiveScore]]
) -> List[Tuple[TI4Map, MultiObjectiveScore]]:
    """
    Extract the Pareto-optimal front from a population.

    A solution is Pareto-optimal if no other solution dominates it.

    Args:
        population: List of (map, score) tuples

    Returns:
        List of non-dominated solutions
    """
    pareto_front = []

    for map_obj, score in population:
        dominated = False

        for other_map, other_score in population:
            if other_score.dominates(score):
                dominated = True
                break

        if not dominated:
            pareto_front.append((map_obj, score))

    return pareto_front


def compare_optimizers(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    iterations: int = 200,
    random_seed: Optional[int] = None
) -> Dict[str, Tuple[TI4Map, Dict]]:
    """
    Compare basic optimizer vs spatial optimizer on the same map.

    Args:
        ti4_map: Starting map
        evaluator: Balance evaluator
        iterations: Optimization iterations
        random_seed: Random seed

    Returns:
        Dictionary with results for each optimizer
    """
    from .balance_engine import improve_balance, analyze_balance

    results = {}

    # Basic optimizer
    print("Running BASIC optimizer...")
    basic_map = ti4_map.copy()
    basic_gap, basic_history = improve_balance(
        basic_map, evaluator, iterations=iterations, random_seed=random_seed
    )

    basic_balance = analyze_balance(basic_map, evaluator)
    basic_spatial = comprehensive_spatial_analysis(basic_map, evaluator)

    results['basic'] = (basic_map, {
        'balance': basic_balance,
        'spatial': basic_spatial,
        'history': basic_history
    })

    # Spatial optimizer
    print("\nRunning SPATIAL optimizer...")
    spatial_map = ti4_map.copy()
    spatial_score, spatial_history = improve_balance_spatial(
        spatial_map, evaluator, iterations=iterations, random_seed=random_seed
    )

    spatial_balance = analyze_balance(spatial_map, evaluator)
    spatial_spatial = comprehensive_spatial_analysis(spatial_map, evaluator)

    results['spatial'] = (spatial_map, {
        'balance': spatial_balance,
        'spatial': spatial_spatial,
        'score': spatial_score,
        'history': spatial_history
    })

    # Print comparison
    print("\n" + "=" * 80)
    print("OPTIMIZER COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<30s} | {'Basic':>12s} | {'Spatial':>12s} | {'Difference':>12s}")
    print("-" * 80)

    print(f"{'Balance Gap':<30s} | {basic_balance['balance_gap']:>12.3f} | {spatial_balance['balance_gap']:>12.3f} | {spatial_balance['balance_gap']-basic_balance['balance_gap']:>+12.3f}")
    morans_label = "Moran's I (Clustering)"
    jains_label = "Jain's Index (Fairness)"
    print(f"{morans_label:<30s} | {basic_spatial['resource_clustering_morans_i']:>12.3f} | {spatial_spatial['resource_clustering_morans_i']:>12.3f} | {spatial_spatial['resource_clustering_morans_i']-basic_spatial['resource_clustering_morans_i']:>+12.3f}")
    print(f"{jains_label:<30s} | {basic_spatial['jains_fairness_index']:>12.3f} | {spatial_spatial['jains_fairness_index']:>12.3f} | {spatial_spatial['jains_fairness_index']-basic_spatial['jains_fairness_index']:>+12.3f}")

    print("=" * 80)

    return results
