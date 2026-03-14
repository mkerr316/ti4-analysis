"""
Statistical analysis tools for experimental results.

Provides functions for:
- Paired t-tests (before vs after)
- Effect size calculation (Cohen's d)
- Correlation analysis
- Summary statistics
- Identifying "smoking gun" cases
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class PairedTestResult:
    """Results from a paired statistical test."""
    metric_name: str
    mean_before: float
    mean_after: float
    mean_difference: float
    std_difference: float
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool
    sample_size: int

    def __str__(self) -> str:
        sig_marker = "***" if self.significant else "   "
        return (
            f"{self.metric_name:30s} | "
            f"Δ={self.mean_difference:+7.3f} | "
            f"t={self.t_statistic:+7.3f} | "
            f"p={self.p_value:.4f} {sig_marker} | "
            f"d={self.cohens_d:+6.3f}"
        )


@dataclass
class CorrelationResult:
    """Results from a correlation analysis."""
    metric_x: str
    metric_y: str
    correlation: float
    p_value: float
    significant: bool
    sample_size: int

    def __str__(self) -> str:
        sig_marker = "***" if self.significant else "   "
        return (
            f"{self.metric_x:25s} vs {self.metric_y:25s} | "
            f"r={self.correlation:+6.3f} | "
            f"p={self.p_value:.4f} {sig_marker}"
        )


def compute_cohens_d(before: np.ndarray, after: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Cohen's d is a standardized measure of effect size:
    - |d| < 0.2: negligible
    - |d| < 0.5: small
    - |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        before: Values before treatment
        after: Values after treatment

    Returns:
        Cohen's d effect size
    """
    diff = after - before
    return np.mean(diff) / np.std(diff, ddof=1)


def paired_t_test(
    df: pd.DataFrame,
    metric_name: str,
    before_col: str,
    after_col: str,
    alpha: float = 0.05
) -> PairedTestResult:
    """
    Perform a paired t-test for a single metric.

    Tests the null hypothesis that the mean difference between
    before and after is zero.

    Args:
        df: DataFrame with experimental results
        metric_name: Descriptive name for the metric
        before_col: Column name for "before" values
        after_col: Column name for "after" values
        alpha: Significance level (default 0.05)

    Returns:
        PairedTestResult with statistical details
    """
    before = df[before_col].values
    after = df[after_col].values

    # Remove any NaN pairs
    mask = ~(np.isnan(before) | np.isnan(after))
    before = before[mask]
    after = after[mask]

    # Compute statistics
    mean_before = np.mean(before)
    mean_after = np.mean(after)
    diff = after - before
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(before, after)

    # Effect size
    cohens_d = compute_cohens_d(before, after)

    return PairedTestResult(
        metric_name=metric_name,
        mean_before=mean_before,
        mean_after=mean_after,
        mean_difference=mean_diff,
        std_difference=std_diff,
        t_statistic=t_stat,
        p_value=p_val,
        cohens_d=cohens_d,
        significant=p_val < alpha,
        sample_size=len(before)
    )


def analyze_experiment_results(
    df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, PairedTestResult]:
    """
    Perform comprehensive paired analysis on all metrics.

    Tests whether optimization significantly changes:
    - Balance gap (expected: decrease)
    - Spatial metrics (hypothesis: no change)

    Args:
        df: DataFrame from run_batch_experiment
        alpha: Significance level

    Returns:
        Dictionary mapping metric name -> PairedTestResult
    """
    results = {}

    # Define metrics to test
    metrics = [
        ("Balance Gap", "naive_balance_gap", "optimized_balance_gap"),
        ("Mean Home Value", "naive_mean_value", "optimized_mean_value"),
        ("Std Home Value", "naive_std_value", "optimized_std_value"),
        ("Fairness Index", "naive_fairness_index", "optimized_fairness_index"),
        ("Moran's I (Clustering)", "naive_morans_i", "optimized_morans_i"),
        ("Jain's Index (Accessibility)", "naive_jains_index", "optimized_jains_index"),
        ("Gini Coefficient", "naive_gini_coefficient", "optimized_gini_coefficient"),
        ("Accessibility Variance", "naive_accessibility_variance", "optimized_accessibility_variance"),
    ]

    for metric_name, before_col, after_col in metrics:
        if before_col in df.columns and after_col in df.columns:
            results[metric_name] = paired_t_test(df, metric_name, before_col, after_col, alpha)

    return results


def correlation_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_x_name: str,
    metric_y_name: str,
    alpha: float = 0.05
) -> CorrelationResult:
    """
    Compute Pearson correlation between two metrics.

    This is used to test the "spatial blindness" hypothesis:
    If optimized balance_gap is uncorrelated with spatial metrics,
    it proves the optimizer is spatially blind.

    Args:
        df: DataFrame with results
        x_col: Column name for X variable
        y_col: Column name for Y variable
        metric_x_name: Descriptive name for X
        metric_y_name: Descriptive name for Y
        alpha: Significance level

    Returns:
        CorrelationResult
    """
    x = df[x_col].values
    y = df[y_col].values

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Pearson correlation
    r, p_val = stats.pearsonr(x, y)

    return CorrelationResult(
        metric_x=metric_x_name,
        metric_y=metric_y_name,
        correlation=r,
        p_value=p_val,
        significant=p_val < alpha,
        sample_size=len(x)
    )


def test_spatial_blindness(
    df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, CorrelationResult]:
    """
    Test the spatial blindness hypothesis.

    Core hypothesis: In optimized maps, balance_gap should be uncorrelated
    with spatial metrics (Moran's I, Jain's Index, etc.).

    A weak/non-significant correlation proves the basic optimizer
    doesn't account for spatial distribution.

    Args:
        df: DataFrame with experimental results
        alpha: Significance level

    Returns:
        Dictionary of correlation results
    """
    correlations = {}

    # Test balance_gap vs spatial metrics (in optimized maps)
    tests = [
        ("optimized_balance_gap", "optimized_morans_i", "Balance Gap", "Moran's I"),
        ("optimized_balance_gap", "optimized_jains_index", "Balance Gap", "Jain's Index"),
        ("optimized_balance_gap", "optimized_gini_coefficient", "Balance Gap", "Gini Coefficient"),
        ("optimized_balance_gap", "optimized_accessibility_variance", "Balance Gap", "Accessibility Variance"),
    ]

    for x_col, y_col, x_name, y_name in tests:
        if x_col in df.columns and y_col in df.columns:
            key = f"{x_name} vs {y_name}"
            correlations[key] = correlation_analysis(df, x_col, y_col, x_name, y_name, alpha)

    return correlations


def find_smoking_gun_cases(
    df: pd.DataFrame,
    max_gap: float = 1.0,
    min_morans_i: float = 0.3,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Identify "smoking gun" maps that prove spatial blindness.

    These are maps that achieved excellent balance (low gap)
    but have severe spatial clustering (high Moran's I).

    Args:
        df: Experimental results
        max_gap: Maximum optimized balance gap
        min_morans_i: Minimum Moran's I threshold
        top_n: Number of worst cases to return

    Returns:
        DataFrame with top smoking gun cases
    """
    # Filter for well-balanced maps
    filtered = df[df['optimized_balance_gap'] <= max_gap].copy()

    # Find maps with high clustering
    filtered = filtered[filtered['optimized_morans_i'] >= min_morans_i]

    # Sort by worst clustering
    smoking_guns = filtered.nlargest(top_n, 'optimized_morans_i')

    return smoking_guns[[
        'map_id', 'random_seed',
        'naive_balance_gap', 'optimized_balance_gap',
        'naive_morans_i', 'optimized_morans_i',
        'optimized_jains_index', 'optimized_gini_coefficient',
    ]]


def compute_summary_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics for all metrics.

    Args:
        df: Experimental results

    Returns:
        Nested dictionary: {metric_name: {stat_name: value}}
    """
    metrics = [
        ('balance_gap', ['naive_balance_gap', 'optimized_balance_gap']),
        ('morans_i', ['naive_morans_i', 'optimized_morans_i']),
        ('jains_index', ['naive_jains_index', 'optimized_jains_index']),
        ('gini_coefficient', ['naive_gini_coefficient', 'optimized_gini_coefficient']),
        ('accessibility_variance', ['naive_accessibility_variance', 'optimized_accessibility_variance']),
    ]

    summary = {}

    for metric_name, cols in metrics:
        for col in cols:
            if col in df.columns:
                summary[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'q25': float(df[col].quantile(0.25)),
                    'median': float(df[col].median()),
                    'q75': float(df[col].quantile(0.75)),
                    'max': float(df[col].max()),
                }

    return summary


def print_analysis_summary(
    paired_results: Dict[str, PairedTestResult],
    correlation_results: Dict[str, CorrelationResult],
    smoking_guns: Optional[pd.DataFrame] = None
) -> None:
    """
    Print a formatted summary of statistical analysis.

    Args:
        paired_results: Results from analyze_experiment_results
        correlation_results: Results from test_spatial_blindness
        smoking_guns: Results from find_smoking_gun_cases
    """
    print("\n" + "=" * 100)
    print("PAIRED T-TEST RESULTS (Naive vs Optimized)")
    print("=" * 100)
    print(f"{'Metric':<30s} | {'Mean Δ':>7s} | {'t-stat':>7s} | {'p-value':>10s} | {'d':>6s}")
    print("-" * 100)

    for result in paired_results.values():
        print(result)

    print("\n" + "=" * 100)
    print("SPATIAL BLINDNESS TEST (Correlations in Optimized Maps)")
    print("=" * 100)
    print(f"{'Metric X':<25s} vs {'Metric Y':<25s} | {'r':>6s} | {'p-value':>10s}")
    print("-" * 100)

    for result in correlation_results.values():
        print(result)

    if smoking_guns is not None and len(smoking_guns) > 0:
        print("\n" + "=" * 100)
        print("SMOKING GUN CASES (Perfect Balance, Poor Spatial Distribution)")
        print("=" * 100)
        print(smoking_guns.to_string())

    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE:")
    print("=" * 100)
    print("Paired t-tests:")
    print("  - Balance Gap should DECREASE (negative Δ, p < 0.05) ← Proves optimizer works")
    print("  - Spatial metrics should NOT CHANGE (p > 0.05) ← Proves spatial blindness")
    print("")
    print("Correlations (in optimized maps):")
    print("  - Balance Gap vs Spatial metrics should be WEAK (|r| < 0.3) ← Proves independence")
    print("  - Non-significant p-values (p > 0.05) further support spatial blindness")
    print("")
    print("Effect sizes (Cohen's d):")
    print("  - |d| < 0.2: negligible")
    print("  - |d| < 0.5: small")
    print("  - |d| < 0.8: medium")
    print("  - |d| >= 0.8: large")
    print("=" * 100)
