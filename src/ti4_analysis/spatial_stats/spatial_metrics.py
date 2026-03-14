"""
Spatial statistics for TI4 map analysis.

Implements advanced spatial metrics proposed in the research documentation:
- Moran's I (spatial autocorrelation)
- Gravity Model (distance-weighted accessibility)
- Jain's Fairness Index
- Resource clustering metrics

References:
    - docs/Twilight Imperium Map Balance Research.md
    - Anselin, L. (1995). Local indicators of spatial association—LISA
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
import warnings

from ..algorithms.hex_grid import HexCoord, hex_distance
from ..data.map_structures import MapSpace, System, Evaluator, MapSpaceType
from ..algorithms.balance_engine import TI4Map


@dataclass
class SpatialWeightMatrix:
    """
    Spatial weight matrix for hexagonal grid.

    W[i,j] = weight between space i and space j
    Common weighting schemes:
    - Binary adjacency: 1 if adjacent, 0 otherwise
    - Distance decay: 1/d^β where d is distance
    """
    weights: np.ndarray  # NxN matrix
    coords: List[HexCoord]  # Coordinate for each row/col

    def row_standardize(self) -> 'SpatialWeightMatrix':
        """
        Row-standardize weights so each row sums to 1.

        This is standard practice for Moran's I and similar metrics.

        Returns:
            New SpatialWeightMatrix with standardized weights
        """
        row_sums = self.weights.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        standardized = self.weights / row_sums
        return SpatialWeightMatrix(weights=standardized, coords=self.coords)


def create_adjacency_weights(
    ti4_map: TI4Map,
    include_wormholes: bool = False,
    evaluator: Optional[Evaluator] = None,
) -> SpatialWeightMatrix:
    """
    Create binary adjacency weight matrix.

    W[i,j] = 1 if spaces i and j are adjacent, 0 otherwise.
    If evaluator is provided, edges from or to impassable tiles (e.g. Supernova)
    are omitted so W reflects navigable topology only.

    Args:
        ti4_map: TI4 map object
        include_wormholes: Whether to treat wormholes as adjacency
        evaluator: If provided, exclude edges involving impassable systems
            (get_distance_modifier(evaluator) is None). Default None for
            backward compatibility (geometric adjacency only).

    Returns:
        Spatial weight matrix
    """
    spaces = [s for s in ti4_map.spaces if s.space_type == MapSpaceType.SYSTEM]
    n = len(spaces)
    weights = np.zeros((n, n))

    def is_passable(space: MapSpace) -> bool:
        if evaluator is None or space.system is None:
            return True
        return space.system.get_distance_modifier(evaluator) is not None

    for i, space_i in enumerate(spaces):
        if not is_passable(space_i):
            continue
        if include_wormholes:
            neighbors = ti4_map.get_adjacent_spaces_including_wormholes(space_i)
        else:
            neighbors = ti4_map.get_adjacent_spaces(space_i)

        for j, space_j in enumerate(spaces):
            if space_j in neighbors and is_passable(space_j):
                weights[i, j] = 1

    coords = [s.coord for s in spaces]
    return SpatialWeightMatrix(weights=weights, coords=coords)


def create_distance_weights(
    ti4_map: TI4Map,
    beta: float = 1.0,
    max_distance: Optional[int] = None
) -> SpatialWeightMatrix:
    """
    Create distance-decay weight matrix.

    W[i,j] = 1 / d^β  (inverse distance decay)

    Args:
        ti4_map: TI4 map object
        beta: Distance decay exponent
        max_distance: Maximum distance to consider (None = unlimited)

    Returns:
        Spatial weight matrix
    """
    spaces = [s for s in ti4_map.spaces if s.space_type == MapSpaceType.SYSTEM]
    n = len(spaces)
    weights = np.zeros((n, n))

    for i, space_i in enumerate(spaces):
        for j, space_j in enumerate(spaces):
            if i == j:
                continue

            dist = hex_distance(space_i.coord, space_j.coord)

            if max_distance is not None and dist > max_distance:
                continue

            if dist > 0:
                weights[i, j] = 1 / (dist ** beta)

    coords = [s.coord for s in spaces]
    return SpatialWeightMatrix(weights=weights, coords=coords)


def morans_i(
    values: np.ndarray,
    weights: SpatialWeightMatrix,
    row_standardized: bool = True
) -> Tuple[float, float, float]:
    """
    Calculate Moran's I spatial autocorrelation statistic.

    Moran's I measures spatial clustering:
    - I > 0: Positive spatial autocorrelation (similar values cluster)
    - I ≈ 0: Random spatial pattern
    - I < 0: Negative spatial autocorrelation (dissimilar values cluster)

    Formula:
        I = (N/W) × Σᵢ Σⱼ wᵢⱼ(xᵢ - x̄)(xⱼ - x̄) / Σᵢ(xᵢ - x̄)²

    where:
        N = number of observations
        W = sum of all weights
        wᵢⱼ = spatial weight between i and j
        xᵢ = value at location i
        x̄ = mean of all values

    Args:
        values: Array of values at each location
        weights: Spatial weight matrix
        row_standardized: Whether to row-standardize weights first

    Returns:
        Tuple of (I, expected_I, variance_I)
        - I: Moran's I statistic
        - expected_I: Expected value under null hypothesis of no correlation
        - variance_I: Variance of I under null hypothesis

    References:
        Anselin, L. (1995). Local indicators of spatial association—LISA.
        Geographical analysis, 27(2), 93-115.
    """
    if row_standardized:
        weights = weights.row_standardize()

    W_matrix = weights.weights
    n = len(values)

    # Demean values
    x_mean = values.mean()
    x_dev = values - x_mean

    # Calculate Moran's I
    numerator = 0.0
    for i in range(n):
        for j in range(n):
            numerator += W_matrix[i, j] * x_dev[i] * x_dev[j]

    denominator = (x_dev ** 2).sum()
    W_sum = W_matrix.sum()

    if denominator == 0 or W_sum == 0:
        return 0.0, -1/(n-1), 0.0

    I = (n / W_sum) * (numerator / denominator)

    # Expected value under null hypothesis
    expected_I = -1 / (n - 1)

    # Analytical variance under normality assumption (Cliff & Ord, 1981).
    # S0 = sum of all weights, S1 and S2 are weight-matrix cross-products.
    S0 = W_matrix.sum()
    S1 = 0.5 * float(np.sum((W_matrix + W_matrix.T) ** 2))
    row_plus_col = (
        np.asarray(W_matrix.sum(axis=1)).ravel()
        + np.asarray(W_matrix.sum(axis=0)).ravel()
    )
    S2 = float(np.sum(row_plus_col ** 2))
    # Sample kurtosis of deviations
    m2 = float((x_dev ** 2).sum()) / n
    m4 = float((x_dev ** 4).sum()) / n
    kurtosis = m4 / (m2 ** 2) if m2 != 0 else 3.0

    A = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    B = kurtosis * (n * (n - 1) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    C = (n - 1) * (n - 2) * (n - 3) * S0 ** 2

    variance_I = (A - B) / C - expected_I ** 2 if C != 0 else 0.0

    return I, expected_I, variance_I


def local_morans_i(
    values: np.ndarray,
    weights: SpatialWeightMatrix,
    row_standardized: bool = True
) -> np.ndarray:
    """
    Calculate Local Moran's I (LISA) for each location.

    Local Moran's I identifies local spatial clusters:
    - I > 0: Location similar to neighbors (high-high or low-low cluster)
    - I < 0: Location dissimilar to neighbors (spatial outlier)

    Formula for location i:
        Iᵢ = (xᵢ - x̄) Σⱼ wᵢⱼ(xⱼ - x̄) / m2
        where m2 = Σ(xᵢ - x̄)² / n  (spatial variance)

    Normalising by m2 makes values dimensionless; Σ Iᵢ ≈ n × I_global.

    Args:
        values: Array of values at each location
        weights: Spatial weight matrix
        row_standardized: Whether to row-standardize weights

    Returns:
        Array of local Moran's I values (dimensionless)
    """
    if row_standardized:
        weights = weights.row_standardize()

    W_matrix = weights.weights
    n = len(values)

    x_mean = values.mean()
    x_dev = values - x_mean
    m2 = float(x_dev @ x_dev) / n
    if m2 == 0.0:
        return np.zeros(n)

    local_I = np.zeros(n)
    for i in range(n):
        local_I[i] = x_dev[i] * (W_matrix[i, :] * x_dev).sum() / m2

    return local_I


def gravity_model_accessibility(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    home_space: MapSpace,
    beta: float = 2.0
) -> float:
    """
    Calculate accessibility using a continuous inverse-distance gravity model.

    .. deprecated::
        This function uses a continuous power-law decay (1/d^beta) which
        contradicts the discrete step-function decay used in the main
        optimization loop (see ``Evaluator.get_distance_multiplier``).
        It is retained **only** for post-hoc comparison against the
        step-function approach in ``comprehensive_spatial_analysis`` and
        must NOT be used inside the fitness evaluation loop.

    Formula:
        Accessibility = Σⱼ (Vⱼ / dᵢⱼ^β)

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluator for system values
        home_space: Home system space
        beta: Distance decay exponent (higher = distance matters more)

    Returns:
        Accessibility score
    """
    warnings.warn(
        "gravity_model_accessibility uses continuous 1/d^beta decay. "
        "The optimization loop uses discrete step-function decay via "
        "Evaluator.DISTANCE_MULTIPLIER. This function is for post-hoc "
        "analysis only.",
        DeprecationWarning,
        stacklevel=2,
    )
    accessibility = 0.0

    for space in ti4_map.spaces:
        if space.space_type != MapSpaceType.SYSTEM or space.system is None:
            continue

        if space.coord == home_space.coord:
            continue

        value = space.system.evaluate(evaluator)
        if value <= 0:
            continue

        distance = hex_distance(home_space.coord, space.coord)
        if distance > 0:
            accessibility += value / (distance ** beta)

    return accessibility


def jains_fairness_index(values: np.ndarray) -> float:
    """
    Calculate Jain's Fairness Index.

    Measures equality of resource distribution:
    - Range: [1/n, 1]
    - J = 1: Perfect fairness (all equal)
    - J = 1/n: Maximum unfairness (one gets all)

    Formula:
        J(x) = (Σxᵢ)² / (n × Σxᵢ²)

    Args:
        values: Array of values to measure fairness

    Returns:
        Fairness index (0 to 1)

    References:
        Jain, R., Chiu, D. M., & Hawe, W. R. (1984). A quantitative measure
        of fairness and discrimination for resource allocation in shared
        computer systems. DEC Research Report TR-301.
    """
    if len(values) == 0:
        return 1.0

    n = len(values)
    sum_x = values.sum()
    sum_x_sq = (values ** 2).sum()

    if sum_x_sq == 0:
        return 1.0

    J = (sum_x ** 2) / (n * sum_x_sq)
    return J


def resource_clustering_coefficient(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    include_wormholes: bool = False
) -> float:
    """
    Calculate resource clustering coefficient using Moran's I.

    High values indicate resources are spatially clustered.
    Low/negative values indicate resources are dispersed.

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluator for system values
        include_wormholes: Whether wormholes count as adjacency

    Returns:
        Moran's I statistic for resource distribution
    """
    # Get system spaces and their values
    spaces = [s for s in ti4_map.spaces if s.space_type == MapSpaceType.SYSTEM and s.system]

    if len(spaces) < 3:
        return 0.0

    values = np.array([s.system.evaluate(evaluator) for s in spaces])

    # Create weight matrix
    weights = create_adjacency_weights(ti4_map, include_wormholes)

    # Calculate Moran's I
    I, expected_I, variance_I = morans_i(values, weights)

    return I


def calculate_spatial_inequality(
    home_accessibilities: List[float]
) -> Dict[str, float]:
    """
    Calculate multiple inequality metrics for spatial accessibility.

    Args:
        home_accessibilities: Accessibility scores for each home position

    Returns:
        Dictionary with inequality metrics:
        - gini_coefficient: Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        - jains_index: Jain's fairness index
        - coefficient_of_variation: Standard deviation / mean
        - range_ratio: max / min
    """
    values = np.array(home_accessibilities)

    if len(values) == 0:
        return {
            'gini_coefficient': 0.0,
            'jains_index': 1.0,
            'coefficient_of_variation': 0.0,
            'range_ratio': 1.0
        }

    # Gini coefficient
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n

    # Jain's index
    jains = jains_fairness_index(values)

    # Coefficient of variation
    cv = values.std() / values.mean() if values.mean() > 0 else 0.0

    # Range ratio
    range_ratio = values.max() / values.min() if values.min() > 0 else np.inf

    return {
        'gini_coefficient': gini,
        'jains_index': jains,
        'coefficient_of_variation': cv,
        'range_ratio': range_ratio
    }


def comprehensive_spatial_analysis(
    ti4_map: TI4Map,
    evaluator: Evaluator
) -> Dict[str, any]:
    """
    Perform comprehensive spatial statistical analysis of map.

    This implements the advanced metrics proposed in the research documentation.

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluator parameters

    Returns:
        Dictionary with all spatial statistics
    """
    # Get home spaces
    home_spaces = ti4_map.get_home_spaces()

    # Calculate accessibility for each home
    accessibilities = []
    for home in home_spaces:
        acc = gravity_model_accessibility(ti4_map, evaluator, home, beta=2.0)
        accessibilities.append(acc)

    # Resource clustering
    clustering = resource_clustering_coefficient(ti4_map, evaluator, include_wormholes=True)

    # Inequality metrics
    inequality = calculate_spatial_inequality(accessibilities)

    return {
        'home_accessibilities': accessibilities,
        'resource_clustering_morans_i': clustering,
        'hotspots': [],
        'num_hotspots': 0,
        'num_coldspots': 0,
        'inequality_metrics': inequality,
        'jains_fairness_index': inequality['jains_index'],
        'gini_coefficient': inequality['gini_coefficient']
    }
