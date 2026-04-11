"""
Advanced visualization tools for experimental analysis.

Creates publication-quality plots for comparing naive vs optimized maps:
- Distribution comparisons (before/after)
- Scatter plots (spatial blindness tests)
- Case study heatmaps
- Convergence analysis
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

from .map_viz import plot_hex_map, plot_value_heatmap
from ..algorithms.balance_engine import TI4Map
from ..data.map_structures import Evaluator


# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_distribution_comparison(
    df: pd.DataFrame,
    metric_before: str,
    metric_after: str,
    metric_name: str,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot before/after distribution comparison using KDE plots.

    This visualization proves the optimizer works by showing
    the shift in balance_gap distribution.

    Args:
        df: Experimental results DataFrame
        metric_before: Column name for "before" values
        metric_after: Column name for "after" values
        metric_name: Display name for metric
        ax: Matplotlib axes (creates new figure if None)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot distributions
    sns.kdeplot(
        data=df[metric_before],
        ax=ax,
        label='Naive (Before)',
        fill=True,
        alpha=0.5,
        color='#e74c3c',
        linewidth=2
    )

    sns.kdeplot(
        data=df[metric_after],
        ax=ax,
        label='Optimized (After)',
        fill=True,
        alpha=0.5,
        color='#2ecc71',
        linewidth=2
    )

    # Add mean lines
    mean_before = df[metric_before].mean()
    mean_after = df[metric_after].mean()

    ax.axvline(mean_before, color='#c0392b', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(mean_after, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)

    # Labels and title
    ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{metric_name} Distribution: Naive vs Optimized\n'
        f'Mean: {mean_before:.2f} → {mean_after:.2f} (Δ = {mean_after - mean_before:+.2f})',
        fontsize=14,
        fontweight='bold'
    )

    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_spatial_blindness_scatter(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    correlation: Optional[float] = None,
    p_value: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create scatter plot to test spatial blindness hypothesis.

    A wide vertical spread of spatial metrics (Y) at low balance gaps (X)
    proves the optimizer is spatially blind.

    Args:
        df: Experimental results
        x_metric: Column for X-axis (typically optimized_balance_gap)
        y_metric: Column for Y-axis (spatial metric)
        x_label: X-axis label
        y_label: Y-axis label
        title: Plot title
        correlation: Pearson r (displayed if provided)
        p_value: Statistical significance (displayed if provided)
        ax: Matplotlib axes
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Create scatter plot with density coloring
    x = df[x_metric]
    y = df[y_metric]

    # Plot scatter
    scatter = ax.scatter(
        x, y,
        alpha=0.6,
        s=50,
        c='#3498db',
        edgecolors='#2c3e50',
        linewidth=0.5
    )

    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')

    # Add horizontal reference line at y=0 if applicable
    if y.min() < 0 < y.max():
        ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.3)

    # Labels and title
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    # Add correlation info to title
    title_text = title
    if correlation is not None and p_value is not None:
        sig_marker = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
        title_text += f'\nr = {correlation:.3f} ({sig_marker}), p = {p_value:.4f}'

    ax.set_title(title_text, fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_multi_metric_comparison(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a multi-panel figure comparing multiple metrics.

    Shows 2x2 grid:
    - Balance Gap distribution
    - Moran's I distribution
    - Gap vs Moran's I scatter
    - Gap vs Jain's Index scatter

    Args:
        df: Experimental results
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Balance Gap distribution
    plot_distribution_comparison(
        df,
        'naive_balance_gap',
        'optimized_balance_gap',
        'Balance Gap',
        ax=axes[0, 0]
    )

    # Panel 2: Moran's I distribution
    plot_distribution_comparison(
        df,
        'naive_morans_i',
        'optimized_morans_i',
        "Moran's I (Resource Clustering)",
        ax=axes[0, 1]
    )

    # Panel 3: Gap vs Moran's I scatter
    from scipy.stats import pearsonr
    r, p = pearsonr(df['optimized_balance_gap'], df['optimized_morans_i'])
    plot_spatial_blindness_scatter(
        df,
        'optimized_balance_gap',
        'optimized_morans_i',
        'Optimized Balance Gap',
        "Moran's I (Clustering)",
        "Spatial Blindness Test: Balance vs Clustering",
        correlation=r,
        p_value=p,
        ax=axes[1, 0]
    )

    # Panel 4: Gap vs Jain's Index scatter
    r2, p2 = pearsonr(df['optimized_balance_gap'], df['optimized_jains_index'])
    plot_spatial_blindness_scatter(
        df,
        'optimized_balance_gap',
        'optimized_jains_index',
        'Optimized Balance Gap',
        "Jain's Fairness Index",
        "Spatial Blindness Test: Balance vs Accessibility",
        correlation=r2,
        p_value=p2,
        ax=axes[1, 1]
    )

    plt.suptitle(
        f'Spatial Blindness Experimental Results (N={len(df)})',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_convergence_comparison(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot how optimization affects different metric categories.

    Shows box plots comparing naive vs optimized for:
    - Balance metrics
    - Spatial metrics

    Args:
        df: Experimental results
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Prepare data for box plots
    balance_data = pd.DataFrame({
        'Naive Gap': df['naive_balance_gap'],
        'Optimized Gap': df['optimized_balance_gap'],
    })

    spatial_data = pd.DataFrame({
        'Naive Moran\'s I': df['naive_morans_i'],
        'Optimized Moran\'s I': df['optimized_morans_i'],
        'Naive Jain\'s Index': df['naive_jains_index'],
        'Optimized Jain\'s Index': df['optimized_jains_index'],
    })

    # Panel 1: Balance metrics
    balance_data.boxplot(ax=axes[0], grid=False)
    axes[0].set_title('Balance Metrics: Before vs After', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Panel 2: Spatial metrics
    spatial_data.boxplot(ax=axes[1], grid=False)
    axes[1].set_title('Spatial Metrics: Before vs After', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Value', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def create_case_study_report(
    naive_map: TI4Map,
    optimized_map: TI4Map,
    evaluator: Evaluator,
    map_id: int,
    metrics_before: dict,
    metrics_after: dict,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a detailed case study visualization for a "smoking gun" map.

    Shows side-by-side comparison:
    - Naive map layout and heatmap
    - Optimized map layout and heatmap
    - Metric comparison table

    Args:
        naive_map: Original random map
        optimized_map: After basic optimization
        evaluator: Balance evaluator
        map_id: Map identifier
        metrics_before: Dict with naive metrics
        metrics_after: Dict with optimized metrics
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Top row: Map layouts
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])

    # Bottom row: Heatmaps
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4 = fig.add_subplot(gs[1, 2:4])

    # Plot naive map
    plot_hex_map(naive_map, title='Naive (Random) Map', ax=ax1)

    # Plot optimized map
    plot_hex_map(optimized_map, title='Optimized (Basic Balance) Map', ax=ax2)

    # Plot heatmaps
    plot_value_heatmap(naive_map, evaluator, title='Naive Value Heatmap', ax=ax3)
    plot_value_heatmap(optimized_map, evaluator, title='Optimized Value Heatmap', ax=ax4)

    # Add overall title with metrics
    balance_gap_before = metrics_before.get('balance_gap', 0)
    balance_gap_after = metrics_after.get('balance_gap', 0)
    morans_i_before = metrics_before.get('resource_clustering_morans_i', 0)
    morans_i_after = metrics_after.get('resource_clustering_morans_i', 0)

    fig.suptitle(
        f'Case Study: Map {map_id} - "Smoking Gun" Example\n'
        f'Balance Gap: {balance_gap_before:.2f} → {balance_gap_after:.2f} | '
        f'Moran\'s I: {morans_i_before:.3f} → {morans_i_after:.3f}',
        fontsize=16,
        fontweight='bold'
    )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_effect_size_comparison(
    paired_results: dict,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize Cohen's d effect sizes for all metrics.

    Creates a horizontal bar chart showing effect sizes,
    color-coded by magnitude (small/medium/large).

    Args:
        paired_results: Dict from analyze_experiment_results
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data
    metrics = []
    cohens_d_values = []
    colors = []

    for metric_name, result in paired_results.items():
        metrics.append(metric_name)
        d = result.cohens_d
        cohens_d_values.append(d)

        # Color by effect size magnitude
        abs_d = abs(d)
        if abs_d < 0.2:
            colors.append('#95a5a6')  # Gray - negligible
        elif abs_d < 0.5:
            colors.append('#3498db')  # Blue - small
        elif abs_d < 0.8:
            colors.append('#f39c12')  # Orange - medium
        else:
            colors.append('#e74c3c')  # Red - large

    # Create horizontal bar chart
    y_pos = np.arange(len(metrics))
    ax.barh(y_pos, cohens_d_values, color=colors, edgecolor='black', linewidth=0.5)

    # Add reference lines
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.8, color='gray', linestyle='--', alpha=0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
    ax.set_title(
        "Effect Sizes: Naive → Optimized\n"
        "Negative = Decrease, Positive = Increase",
        fontsize=14,
        fontweight='bold'
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color='#95a5a6', label='Negligible (|d| < 0.2)'),
        mpatches.Patch(color='#3498db', label='Small (0.2 ≤ |d| < 0.5)'),
        mpatches.Patch(color='#f39c12', label='Medium (0.5 ≤ |d| < 0.8)'),
        mpatches.Patch(color='#e74c3c', label='Large (|d| ≥ 0.8)'),
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, shadow=True)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_raincloud(
    data: pd.DataFrame,
    x: str,
    y: str,
    palette: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    orient: str = "h",
    baseline_val: Optional[float] = None,
    baseline_label: str = "Random Baseline"
) -> plt.Axes:
    """
    Create a raincloud plot (stripplot + violinplot + boxplot) with optional baseline anchor.
    
    Args:
        data: DataFrame containing the data
        x: Column name for the categorical variable
        y: Column name for the numeric variable
        palette: Color palette mapping
        ax: Matplotlib axes
        title: Plot title
        orient: Orientation ('h' or 'v')
        baseline_val: Optional value to draw a reference line
        baseline_label: Label for the reference line
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Determine order if it's a categorical column
    order = None
    if data[x].dtype.name == 'category':
        order = data[x].cat.categories

    # Fallback to violin + stripplot + boxplot
    sns.violinplot(
        data=data, x=y if orient == "h" else x, y=x if orient == "h" else y,
        palette=palette, ax=ax, inner=None, alpha=0.3, order=order, orient=orient
    )
    sns.stripplot(
        data=data, x=y if orient == "h" else x, y=x if orient == "h" else y,
        palette=palette, ax=ax, size=2, alpha=0.3, order=order, jitter=0.2, orient=orient
    )
    sns.boxplot(
        data=data, x=y if orient == "h" else x, y=x if orient == "h" else y,
        palette=palette, ax=ax, width=0.1, boxprops={'zorder': 2}, 
        showfliers=False, order=order, orient=orient
    )
    
    # Add baseline anchor
    if baseline_val is not None:
        if orient == "h":
            ax.axvline(baseline_val, color='red', linestyle='--', linewidth=2, alpha=0.6, label=baseline_label)
        else:
            ax.axhline(baseline_val, color='red', linestyle='--', linewidth=2, alpha=0.6, label=baseline_label)
        ax.legend()

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    return ax


def plot_pareto_projections(
    df: pd.DataFrame,
    objectives: List[Tuple[str, str, str]], # (col, label, short_label)
    hue_metric: str = "condition",
    palette: Optional[Dict] = None,
    save_path: Optional[Path] = None,
    utopia_points: Optional[Dict[str, float]] = None
) -> plt.Figure:
    """
    Create a grid of 2D Pareto projections with optional Utopia point anchors.
    
    Args:
        df: DataFrame containing objective values
        objectives: List of (col_name, label, short_label)
        hue_metric: Column to use for color
        palette: Color palette
        save_path: Where to save
        utopia_points: Optional dict mapping column names to their ideal values
        
    Returns:
        Matplotlib figure
    """
    n = len(objectives)
    
    # Pairwise combinations for 3 objectives
    combos = [
        (objectives[0], objectives[1]), # JFI vs Moran
        (objectives[0], objectives[2]), # JFI vs LSAP
        (objectives[1], objectives[2]), # Moran vs LSAP
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (obj1, obj2) in enumerate(combos):
        ax = axes[i]
        sns.scatterplot(
            data=df, x=obj1[0], y=obj2[0], hue=hue_metric, 
            palette=palette, ax=ax, alpha=0.5, s=20
        )
        
        # Add Utopia point if provided
        if utopia_points and obj1[0] in utopia_points and obj2[0] in utopia_points:
            ux = utopia_points[obj1[0]]
            uy = utopia_points[obj2[0]]
            ax.scatter([ux], [uy], marker='*', s=200, color='gold', 
                       edgecolors='black', label='Utopia Point', zorder=5)

        ax.set_xlabel(obj1[1], fontweight='bold')
        ax.set_ylabel(obj2[1], fontweight='bold')
        ax.set_title(f"{obj1[2]} vs {obj2[2]}", fontweight='bold')
        
        if i < 2:
            legend = ax.get_legend()
            if legend: legend.remove()
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_core_objective_distributions(
    df: pd.DataFrame,
    metrics: List[Tuple[str, str]], # (col, label)
    condition_col: str = "condition",
    palette: Optional[Dict] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Streamlined distribution plots for core objectives using rainclouds.
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1: axes = [axes]
    
    for i, (metric, label) in enumerate(metrics):
        plot_raincloud(
            df, x=condition_col, y=metric, 
            palette=palette, ax=axes[i], title=label, orient="v"
        )
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=15)
        axes[i].set_xlabel("")
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_ridgeline(
    data: pd.DataFrame,
    x: str,
    y: str,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
    floor: Optional[float] = None
) -> sns.FacetGrid:
    """
    Create a ridgeline plot using Seaborn FacetGrid.
    """
    # Joypy-style ridgeline using Seaborn
    g = sns.FacetGrid(data, row=y, hue=y, aspect=5, height=1.2, palette=palette)
    g.map(sns.kdeplot, x, bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, x, clip_on=False, color="w", lw=2, bw_adjust=.5)
    
    # Passing color=None to refline() to use the hue mapping or a fixed color
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, x)
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    
    if title:
        g.figure.suptitle(title, y=1.02)
    
    if floor is not None:
        for ax in g.axes.flat:
            ax.axvline(floor, color="red", linestyle="--", alpha=0.5)
            # Annotate floor on the last axis
            if ax == g.axes.flat[-1]:
                ax.text(floor, 0, f"Floor: {floor}", color="red", ha="right", va="bottom", rotation=90)
    
    return g


def create_all_experiment_visualizations(
    df: pd.DataFrame,
    paired_results: dict,
    correlation_results: dict,
    output_dir: Path,
    smoking_gun_maps: Optional[List[Tuple[TI4Map, TI4Map, int]]] = None
) -> List[Path]:
    """
    Generate all experimental visualizations and save to directory.

    Args:
        df: Experimental results DataFrame
        paired_results: Results from analyze_experiment_results
        correlation_results: Results from test_spatial_blindness
        output_dir: Directory to save figures
        smoking_gun_maps: List of (naive_map, optimized_map, map_id) tuples

    Returns:
        List of paths to saved figures
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    print("Generating experimental visualizations...")

    # 1. Multi-metric comparison
    print("  - Multi-metric comparison panel...")
    path = figures_dir / "multi_metric_comparison.png"
    plot_multi_metric_comparison(df, save_path=path)
    saved_paths.append(path)

    # 2. Balance gap distribution
    print("  - Balance gap distribution...")
    path = figures_dir / "distribution_balance_gap.png"
    plot_distribution_comparison(
        df, 'naive_balance_gap', 'optimized_balance_gap',
        'Balance Gap', save_path=path
    )
    saved_paths.append(path)

    # 3. Moran's I distribution
    print("  - Moran's I distribution...")
    path = figures_dir / "distribution_morans_i.png"
    plot_distribution_comparison(
        df, 'naive_morans_i', 'optimized_morans_i',
        "Moran's I (Resource Clustering)", save_path=path
    )
    saved_paths.append(path)

    # 4. Scatter plots
    print("  - Scatter plot: Gap vs Moran's I...")
    from scipy.stats import pearsonr
    r, p = pearsonr(df['optimized_balance_gap'], df['optimized_morans_i'])
    path = figures_dir / "scatter_gap_vs_morans.png"
    plot_spatial_blindness_scatter(
        df, 'optimized_balance_gap', 'optimized_morans_i',
        'Optimized Balance Gap', "Moran's I",
        "Spatial Blindness: Balance vs Clustering",
        correlation=r, p_value=p, save_path=path
    )
    saved_paths.append(path)

    print("  - Scatter plot: Gap vs Jain's Index...")
    r2, p2 = pearsonr(df['optimized_balance_gap'], df['optimized_jains_index'])
    path = figures_dir / "scatter_gap_vs_jains.png"
    plot_spatial_blindness_scatter(
        df, 'optimized_balance_gap', 'optimized_jains_index',
        'Optimized Balance Gap', "Jain's Fairness Index",
        "Spatial Blindness: Balance vs Accessibility",
        correlation=r2, p_value=p2, save_path=path
    )
    saved_paths.append(path)

    # 5. Effect sizes
    print("  - Effect size comparison...")
    path = figures_dir / "effect_sizes.png"
    plot_effect_size_comparison(paired_results, save_path=path)
    saved_paths.append(path)

    # 6. Convergence comparison
    print("  - Convergence comparison...")
    path = figures_dir / "convergence_comparison.png"
    plot_convergence_comparison(df, save_path=path)
    saved_paths.append(path)

    print(f"✓ Generated {len(saved_paths)} visualizations in {figures_dir}")

    return saved_paths
