"""
Multi-objective spatial optimizer for TI4 maps.

Implements Pareto optimization across three canonical objectives:

    1. 1 − jains_index  (minimize) — distributive equity (Jain's Fairness Index)
    2. max(0, morans_i − E[I])  (minimize) — global spatial clustering (one-sided hinge)
    3. lisa_penalty     (minimize) — Local Spatial Autocorrelation Penalty (LSAP)

Decomposition: JFI captures the *magnitude* of resource disparity (distributive
justice); Moran's I and LSAP capture the *spatial topology* of that disparity
(spatial justice). These are orthogonal analytical dimensions. Within the
distributive axis, JFI replaces balance_gap (max−min range) as the axiomatic,
scale-invariant measure recommended by peer-reviewed literature.

LSAP (lisa_penalty) is a continuous heuristic proxy for significance-tested
LISA (Anselin, 1995). It sums all positive variance-normalised local Moran's
I_i values without permutation-test significance filtering, providing a smooth
fitness signal for metaheuristic optimisation. Post-hoc validation confirms
that minimising this proxy eliminates statistically significant clusters.

balance_gap is retained as a stored attribute for display and reporting only.
"""

import math
import random
from typing import Tuple, List, Optional, Dict
import numpy as np

from .balance_engine import TI4Map, get_home_values, get_balance_gap, can_swap_system
from .map_topology import MapTopology
from .fast_map_state import FastMapState
from .objectives_smooth import smooth_min_jain, softplus_hinge, DEFAULT_JAIN_SMOOTH_P, DEFAULT_SOFTPLUS_K
from ..data.map_structures import Evaluator, MapSpace
from ..spatial_stats.spatial_metrics import comprehensive_spatial_analysis

# Keys for normalizer_sigma and raw_objective_terms
NORM_KEY_HINGE = 'morans_hinge'
NORM_KEY_JFI = 'jfi_gap'
NORM_KEY_LISA = 'lisa_norm'
NORM_EPS = 1e-10


class MultiObjectiveScore:
    """
    Container for multi-objective fitness scores.

    A map's quality is evaluated across multiple dimensions:
    - Balance gap (lower is better) — display only
    - Spatial clustering / Moran's I (one-sided hinge: max(0, I − E[I]), lower is better)
    - Multi-Jain Fairness Index (higher is better): min(JFI_resources, JFI_influence)
    - LSAP / lisa_penalty (lower is better) — continuous proxy for LISA
    """

    def __init__(
        self,
        balance_gap: float,
        morans_i: float,
        jains_index: float,
        lisa_penalty: float = 0.0,
        n_spatial: int = 1,
        weights: Optional[Dict[str, float]] = None,
        jfi_resources: float = 1.0,
        jfi_influence: float = 1.0,
        normalizer_sigma: Optional[Dict[str, float]] = None,
        use_smooth_objectives: bool = False,
        smooth_p: float = DEFAULT_JAIN_SMOOTH_P,
        smooth_k: float = DEFAULT_SOFTPLUS_K,
    ):
        self.balance_gap = balance_gap   # retained for display only
        self.morans_i = morans_i
        self.jains_index = jains_index
        self.jfi_resources = jfi_resources
        self.jfi_influence = jfi_influence
        self.lisa_penalty = lisa_penalty
        self.n_spatial = max(1, n_spatial)  # guard against zero-division
        self.normalizer_sigma = normalizer_sigma
        self.use_smooth_objectives = use_smooth_objectives
        self.smooth_p = smooth_p
        self.smooth_k = smooth_k

        # Weights sum to 1.0 for interpretability (equal 1 : 1 : 1).
        if weights is None:
            weights = {
                'morans_i':     1.0 / 3.0,
                'jains_index':  1.0 / 3.0,
                'lisa_penalty': 1.0 / 3.0,
            }

        self.weights = weights

    def composite_score(self) -> float:
        """
        Scalar objective for SA and HC (Weighted Sum Model). Lower is better.

        All three terms are normalized before weighting:
          - max(0, I − E[I]) ∈ [0, 1 + 1/(n-1)] ≈ [0, 1.028] for n=37
            One-sided hinge: fires only when I exceeds the null expectation
            E[I] = -1/(n-1). Negative autocorrelation (spatial dispersion)
            passes at zero cost; only pathological positive clustering is penalized.
          - 1 − jains_index ∈ [0, 1]   (JFI ∈ [1/H, 1])
          - lisa_norm        ∈ [0, 1]   (see derivation below)

        LSAP normalization — n(n-1) upper bound derivation:
            local_I[i] = z_dev[i] * (W @ z_dev)[i] / m2  (variance-normalized).
            For row-standardized W, the maximum positive local_I achievable at
            node i of degree k_i is derived by placing {i} ∪ N(i) at value a
            and the remaining n-k_i-1 nodes at the compensating value b:
                max positive local_I[i]  =  (n - k_i - 1) / (k_i + 1)
            This is maximised at k_i = 1 (leaf node): (n-2)/2 ≈ 17.5 for n=37.
            Summing conservatively over all n nodes gives:
                LSAP  ≤  n × (n-2)/2  ≤  n × (n-1)  =  n(n-1)
            Using n(n-1) as the divisor maps LSAP to [0, 1] across all topologies
            and is tighter than the earlier 'max single = n-1' claim (which
            assumed a leaf surrounded by identical neighbours — the correct leaf
            maximum is (n-2)/2, not n-1).

        Weights sum to 1.0 (ratios 5 : 5 : 3 ≈ 0.385 : 0.385 : 0.231).
        """
        n = self.n_spatial
        lisa_divisor = max(1, n * (n - 1))
        lisa_norm = self.lisa_penalty / lisa_divisor
        # Hinge: one-sided penalize I > E[I] = -1/(n-1). Smooth or hard.
        hinge_raw = self.morans_i + 1.0 / max(1, n - 1)
        if self.use_smooth_objectives:
            morans_hinge = softplus_hinge(hinge_raw, self.smooth_k)
            jfi_term = 1.0 - smooth_min_jain(self.jfi_resources, self.jfi_influence, self.smooth_p)
        else:
            morans_hinge = max(0.0, hinge_raw)
            jfi_term = 1.0 - self.jains_index
        # Optional Gen-0 variance normalization (stationary landscape).
        if self.normalizer_sigma:
            sigma_h = max(self.normalizer_sigma.get(NORM_KEY_HINGE, 1.0), NORM_EPS)
            sigma_j = max(self.normalizer_sigma.get(NORM_KEY_JFI, 1.0), NORM_EPS)
            sigma_l = max(self.normalizer_sigma.get(NORM_KEY_LISA, 1.0), NORM_EPS)
            morans_hinge = morans_hinge / sigma_h
            jfi_term = jfi_term / sigma_j
            lisa_norm = lisa_norm / sigma_l
        return (
            self.weights['morans_i']     * morans_hinge +
            self.weights['jains_index']  * jfi_term +
            self.weights['lisa_penalty'] * lisa_norm
        )

    def raw_objective_terms(self) -> Tuple[float, float, float]:
        """
        Raw normalized terms (hinge, jfi_gap, lisa_norm) with same divisors as composite.
        Used for Gen-0 sigma computation. No smooth operators; no normalizer_sigma applied.
        """
        n = self.n_spatial
        lisa_divisor = max(1, n * (n - 1))
        lisa_norm = self.lisa_penalty / lisa_divisor
        morans_hinge = max(0.0, self.morans_i + 1.0 / max(1, n - 1))
        jfi_gap = 1.0 - self.jains_index
        return (morans_hinge, jfi_gap, lisa_norm)

    def objective_values_for_pareto(self) -> Tuple[float, float, float]:
        """
        Three objectives (minimize) in the form used for Pareto dominance and crowding.
        Respects use_smooth_objectives so NSGA-II stays consistent with Track A composite.
        Returns (f1, f2, f3) = (JFI gap, Moran hinge term, LSAP).
        """
        n = max(1, self.n_spatial - 1)
        if self.use_smooth_objectives:
            f1 = 1.0 - smooth_min_jain(self.jfi_resources, self.jfi_influence, self.smooth_p)
            f2 = softplus_hinge(self.morans_i + 1.0 / n, self.smooth_k)
        else:
            f1 = 1.0 - self.jains_index
            f2 = max(0.0, self.morans_i + 1.0 / n)
        f3 = self.lisa_penalty
        return (f1, f2, f3)

    def lex_key(self) -> tuple:
        """
        Lexicographic fitness key for HC, TS, and SGA. Lower is better.

        Primary: full composite loss (Moran's I hinge + fairness gap + LSAP).
        Secondary: −J_max = −max(JFI_R, JFI_I).

        On zero-delta plateaus where a swap improves the non-bottleneck JFI
        dimension without changing J_min (so composite_score is identical),
        the secondary key breaks the tie in favour of higher J_max, providing
        directional momentum across the flat ridge.

        SA does not use lex_key. SA's Metropolis criterion natively accepts
        equal or worse moves via exp(−Δ/T) — when Δ=0, acceptance probability
        is 1 at any temperature — so SA traverses zero-delta plateaus without
        any modification.

        Track A final-solution comparison uses composite_score() for all
        algorithms; lex_key() affects only internal accept/reject decisions
        of HC, TS, and SGA and does not alter the reported scalar metric.
        """
        j_max = max(self.jfi_resources, self.jfi_influence)
        return (self.composite_score(), -j_max)

    def dominates(self, other: 'MultiObjectiveScore') -> bool:
        """
        Pareto dominance over the canonical 3 objectives (all minimized).
        Uses same smooth/hard formulae as composite_score when use_smooth_objectives is set.
        """
        n_self = max(1, self.n_spatial - 1)
        n_other = max(1, other.n_spatial - 1)
        if self.use_smooth_objectives:
            a_jfi = 1.0 - smooth_min_jain(self.jfi_resources, self.jfi_influence, self.smooth_p)
            b_jfi = 1.0 - smooth_min_jain(other.jfi_resources, other.jfi_influence, other.smooth_p)
            a_mi = softplus_hinge(self.morans_i + 1.0 / n_self, self.smooth_k)
            b_mi = softplus_hinge(other.morans_i + 1.0 / n_other, other.smooth_k)
        else:
            a_jfi = 1.0 - self.jains_index
            b_jfi = 1.0 - other.jains_index
            a_mi = max(0.0, self.morans_i + 1.0 / n_self)
            b_mi = max(0.0, other.morans_i + 1.0 / n_other)
        a_lp = self.lisa_penalty
        b_lp = other.lisa_penalty
        all_leq = (a_jfi <= b_jfi) and (a_mi <= b_mi) and (a_lp <= b_lp)
        any_lt = (a_jfi < b_jfi) or (a_mi < b_mi) or (a_lp < b_lp)
        return all_leq and any_lt

    def __str__(self) -> str:
        return (
            f"Gap={self.balance_gap:.2f}, "
            f"Moran's I={self.morans_i:+.3f}, "
            f"Jain's={self.jains_index:.3f} "
            f"(R={self.jfi_resources:.3f}, I={self.jfi_influence:.3f}), "
            f"LISA={self.lisa_penalty:.3f}, "
            f"Composite={self.composite_score():.2f}"
        )


def evaluate_map_multiobjective(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    weights: Optional[Dict[str, float]] = None,
    fast_state: Optional[FastMapState] = None,
    normalizer_sigma: Optional[Dict[str, float]] = None,
    use_smooth_objectives: bool = False,
    smooth_p: float = DEFAULT_JAIN_SMOOTH_P,
    smooth_k: float = DEFAULT_SOFTPLUS_K,
    use_local_variance_lisa: bool = False,
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
        normalizer_sigma: Optional frozen Gen-0 stds (morans_hinge, jfi_gap, lisa_norm) for stationary composite.
        use_smooth_objectives: If True, use smooth min (JFI) and softplus (hinge) in composite and dominance.
        smooth_p: Exponent for smooth Jain (L_{-p} mean). Default 8.
        smooth_k: Temperature for softplus hinge. Default 10, clamped [5, 20].
        use_local_variance_lisa: If True, LSAP uses sqrt(k_i) correction in lisa_penalty_swappable (used in corrected-landscape path).

    Returns:
        MultiObjectiveScore with all metrics
    """
    if fast_state is not None:
        # All metrics from vectorized fast paths — swappable-only spatial (no static outliers).
        balance_gap    = fast_state.balance_gap()
        morans_i       = fast_state.morans_i_swappable()
        jains_index    = fast_state.jains_index()
        jfi_resources  = fast_state.jfi_resources()
        jfi_influence  = fast_state.jfi_influence()
        lisa_penalty   = fast_state.lisa_penalty_swappable(use_local_variance=use_local_variance_lisa)
        n_spatial      = max(1, len(fast_state.topology.swappable_connected_s_pos))
    else:
        # Slow path: per-dimension JFI unavailable; fall back to combined scalar.
        home_values = get_home_values(ti4_map, evaluator)
        balance_gap = get_balance_gap(home_values)
        spatial_analysis = comprehensive_spatial_analysis(ti4_map, evaluator)
        morans_i       = spatial_analysis['resource_clustering_morans_i']
        jains_index    = spatial_analysis['jains_fairness_index']
        jfi_resources  = jains_index
        jfi_influence  = jains_index
        lisa_penalty   = 0.0
        n_spatial      = 1

    return MultiObjectiveScore(
        balance_gap, morans_i, jains_index, lisa_penalty, n_spatial, weights,
        jfi_resources=jfi_resources, jfi_influence=jfi_influence,
        normalizer_sigma=normalizer_sigma,
        use_smooth_objectives=use_smooth_objectives,
        smooth_p=smooth_p,
        smooth_k=smooth_k,
    )


def compute_gen0_sigma(
    topology: MapTopology,
    evaluator: Evaluator,
    ti4_map_template: TI4Map,
    n_samples: int = 1000,
    random_seed: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    use_local_variance_lisa: bool = False,
    n_swaps_randomize: int = 80,
) -> Dict[str, float]:
    """
    Compute frozen empirical standard deviations of the three objective terms
    from a Gen-0 (uniformly random) sample of map permutations. Used for
    stationary variance normalization so the fitness landscape does not
    shift during optimization.

    Args:
        topology: Pre-built topology for the map template.
        evaluator: Balance evaluator.
        ti4_map_template: Template map; copies are randomized for each sample.
        n_samples: Number of random permutations to sample (default 1000).
        random_seed: Optional seed for reproducibility.
        weights: Objective weights (for score construction; default 1:1:1).
        use_local_variance_lisa: If True, LSAP uses sqrt(k_i) correction (match corrected-landscape benchmark).
        n_swaps_randomize: Random 2-swaps per copy to randomize (default 80).

    Returns:
        Dict with keys NORM_KEY_HINGE, NORM_KEY_JFI, NORM_KEY_LISA and values
        empirical std (at least NORM_EPS to avoid division by zero).
    """
    if random_seed is not None:
        random.seed(random_seed)
    swappable_indices = list(range(len(topology.swappable_indices)))
    if len(swappable_indices) < 2:
        return {NORM_KEY_HINGE: 1.0, NORM_KEY_JFI: 1.0, NORM_KEY_LISA: 1.0}
    terms_list: List[Tuple[float, float, float]] = []
    for _ in range(n_samples):
        map_copy = ti4_map_template.copy()
        swappable_spaces = [map_copy.spaces[i] for i in topology.swappable_indices]
        for _ in range(n_swaps_randomize):
            s1, s2 = random.sample(swappable_indices, 2)
            swappable_spaces[s1].system, swappable_spaces[s2].system = (
                swappable_spaces[s2].system, swappable_spaces[s1].system,
            )
        fast_state = FastMapState.from_ti4_map(topology, map_copy, evaluator)
        score = evaluate_map_multiobjective(
            map_copy, evaluator, weights=weights, fast_state=fast_state,
            use_local_variance_lisa=use_local_variance_lisa,
        )
        terms_list.append(score.raw_objective_terms())
    arr = np.array(terms_list)
    sigma_h = float(np.std(arr[:, 0]))
    sigma_j = float(np.std(arr[:, 1]))
    sigma_l = float(np.std(arr[:, 2]))
    return {
        NORM_KEY_HINGE: max(sigma_h, NORM_EPS),
        NORM_KEY_JFI: max(sigma_j, NORM_EPS),
        NORM_KEY_LISA: max(sigma_l, NORM_EPS),
    }


def improve_balance_spatial(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    iterations: int = 200,
    weights: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True,
    cooling_rate: float = 0.99,
    min_temp: float = 0.01,
    initial_acceptance_rate: float = 0.80,
    normalizer_sigma: Optional[Dict[str, float]] = None,
    use_smooth_objectives: bool = False,
    smooth_p: float = DEFAULT_JAIN_SMOOTH_P,
    smooth_k: float = DEFAULT_SOFTPLUS_K,
    use_local_variance_lisa: bool = False,
) -> Tuple[MultiObjectiveScore, List[Tuple[int, MultiObjectiveScore]]]:
    """
    Improve map balance using Simulated Annealing over a multi-objective
    composite fitness (balance_gap + Moran's I + Jain's Index + LISA penalty).

    SA replaces greedy hill-climbing: the Metropolis criterion allows
    probabilistic acceptance of worse moves at high temperatures, escaping
    local optima. Temperature decays geometrically until min_temp, at which
    point the algorithm collapses to a greedy finisher.

    Initial temperature is calibrated dynamically from the local fitness
    landscape: 10 probe swaps determine the average positive delta, and T₀
    is chosen so the initial acceptance rate equals initial_acceptance_rate.

    Args:
        ti4_map: Map to optimize (modified in-place)
        evaluator: Balance evaluator
        iterations: Maximum number of swap attempts
        weights: Objective weights (morans_i, jains_index, lisa_penalty; sum to 1.0)
        random_seed: Random seed for reproducibility
        verbose: Print progress messages
        cooling_rate: Geometric cooling factor α (T_{k+1} = α·T_k)
        min_temp: Temperature floor; algorithm stops if T falls below this
        initial_acceptance_rate: Target P(accept worse move) at T₀ (e.g. 0.80)

    Returns:
        Tuple of (best_score, history, evals_to_best)
        history is a list of (iteration, score) tuples
        evals_to_best is the iteration at which the incumbent was last improved
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Build static topology once; fitness evaluations use fast matmul.
    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
    swappable_spaces = [ti4_map.spaces[i] for i in topology.swappable_indices]
    sample_indices = list(range(len(swappable_spaces)))

    if len(swappable_spaces) < 2:
        raise ValueError("Not enough swappable spaces to optimize")

    # Initial evaluation
    current_score = evaluate_map_multiobjective(
        ti4_map, evaluator, weights, fast_state,
        normalizer_sigma=normalizer_sigma,
        use_smooth_objectives=use_smooth_objectives,
        smooth_p=smooth_p,
        smooth_k=smooth_k,
        use_local_variance_lisa=use_local_variance_lisa,
    )
    best_score = current_score
    best_eval = 0
    history = [(0, current_score)]

    # ── Calibrate initial temperature ────────────────────────────────────────
    # Sample up to 10 random swaps and collect positive deltas.
    # Solve T₀ = -avg_delta / ln(initial_acceptance_rate) so that the
    # Metropolis criterion accepts ~initial_acceptance_rate of worsening moves.
    sample_deltas = []
    for _ in range(10):
        s1, s2 = random.sample(sample_indices, 2)
        fast_state.swap(s1, s2)
        probe = evaluate_map_multiobjective(
            ti4_map, evaluator, weights, fast_state,
            normalizer_sigma=normalizer_sigma,
            use_smooth_objectives=use_smooth_objectives,
            smooth_p=smooth_p,
            smooth_k=smooth_k,
            use_local_variance_lisa=use_local_variance_lisa,
        )
        delta = probe.composite_score() - current_score.composite_score()
        if delta > 0:
            sample_deltas.append(delta)
        fast_state.swap(s1, s2)  # revert — ti4_map not touched during probes

    avg_delta = float(np.mean(sample_deltas)) if sample_deltas else 1.0
    temperature = -avg_delta / math.log(initial_acceptance_rate)

    # Derive cooling_rate from the iteration budget so that temperature reaches
    # min_temp exactly at iteration `iterations`.  This makes --sa-iter the
    # authoritative budget knob: α = (min_temp / T₀)^(1/N).
    # The user-supplied cooling_rate parameter is intentionally overridden here.
    effective_cooling_rate = (min_temp / temperature) ** (1.0 / max(1, iterations))

    if verbose:
        print(f"Initial: {current_score}")
        print(f"Calibrated T₀={temperature:.4f} (avg_delta={avg_delta:.4f}, "
              f"target_accept={initial_acceptance_rate:.0%})")

    # ── Main SA loop ──────────────────────────────────────────────────────────
    last_i = 0
    for i in range(1, iterations + 1):
        last_i = i
        s1, s2 = random.sample(sample_indices, 2)
        space1 = swappable_spaces[s1]
        space2 = swappable_spaces[s2]

        # Apply swap to both fast state and ti4_map
        fast_state.swap(s1, s2)
        space1.system, space2.system = space2.system, space1.system

        new_score = evaluate_map_multiobjective(
            ti4_map, evaluator, weights, fast_state,
            normalizer_sigma=normalizer_sigma,
            use_smooth_objectives=use_smooth_objectives,
            smooth_p=smooth_p,
            smooth_k=smooth_k,
            use_local_variance_lisa=use_local_variance_lisa,
        )
        delta = new_score.composite_score() - current_score.composite_score()

        if delta < 0:
            # Always accept improvements
            current_score = new_score
            if new_score.composite_score() < best_score.composite_score():
                best_score = new_score
                best_eval = i
                if verbose and i % 10 == 0:
                    print(f"Iteration {i} [T={temperature:.4f}]: {new_score}")
        elif temperature > min_temp and random.random() < math.exp(-delta / temperature):
            # Metropolis criterion: probabilistically accept worse move
            current_score = new_score
        else:
            # Reject — revert both fast state and ti4_map
            fast_state.swap(s1, s2)
            space1.system, space2.system = space2.system, space1.system

        # Geometric cooling (uses budget-derived rate, not user-supplied cooling_rate)
        if temperature > min_temp:
            temperature *= effective_cooling_rate

        # Record history every 10 iterations
        if i % 10 == 0:
            history.append((i, current_score))

        # Natural SA termination: cooling schedule exhausted
        if temperature <= min_temp:
            if verbose:
                print(f"SA converged at iteration {i} (T={temperature:.4f} ≤ min_temp={min_temp})")
            break

    if verbose:
        print(f"Final best: {best_score}")

    history.append((last_i, best_score))
    return best_score, history, best_eval


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
    spatial_score, spatial_history, _ = improve_balance_spatial(
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
