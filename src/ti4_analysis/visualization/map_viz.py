"""
Visualization tools for TI4 maps and balance analysis.

Provides functions for:
- Hexagonal map rendering
- Balance statistics plots
- Optimization convergence plots
- Spatial heatmaps
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple

from ti4_analysis.algorithms.hex_grid import HexCoord, hex_distance
from ti4_analysis.data.map_structures import MapSpace, MapSpaceType, Evaluator
from ti4_analysis.algorithms.balance_engine import TI4Map
from ti4_analysis.algorithms.balance_engine import HomeValue, get_home_values


# Hexagon visualization constants
HEX_RADIUS = 1.0
HEX_HEIGHT = np.sqrt(3) * HEX_RADIUS
HEX_WIDTH = 2 * HEX_RADIUS


def cube_to_pixel(coord: HexCoord, size: float = HEX_RADIUS) -> Tuple[float, float]:
    """
    Convert cube coordinates to pixel coordinates for visualization.

    Uses flat-top hexagon orientation.

    Args:
        coord: Hex coordinate
        size: Hexagon size (radius)

    Returns:
        (x, y) pixel coordinates
    """
    x = size * (3/2 * coord.x)
    y = size * (np.sqrt(3)/2 * coord.x + np.sqrt(3) * coord.y)
    return x, y


def create_hexagon_patch(coord: HexCoord, size: float = HEX_RADIUS) -> mpatches.RegularPolygon:
    """
    Create a matplotlib hexagon patch at given coordinate.

    Args:
        coord: Hex coordinate
        size: Hexagon size (radius)

    Returns:
        RegularPolygon patch
    """
    x, y = cube_to_pixel(coord, size)
    return mpatches.RegularPolygon(
        (x, y), 6, radius=size,
        orientation=np.pi/6,  # Rotate 30 degrees to get flat top/bottom
        edgecolor='black',
        linewidth=1
    )


def plot_hex_map(
    ti4_map: 'TI4Map',
    ax: Optional[plt.Axes] = None,
    color_by: str = 'type',
    value_map: Optional[Dict[HexCoord, float]] = None,
    show_coords: bool = False,
    show_system_ids: bool = True,
    title: Optional[str] = None,
    size: float = HEX_RADIUS,
    cmap: str = 'magma',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_legend: bool = True
) -> plt.Axes:
    """
    Plot TI4 hexagonal map.

    Args:
        ti4_map: TI4 map object
        ax: Matplotlib axes (creates new if None)
        color_by: How to color hexes ('type', 'value', 'resources')
        value_map: Optional dict mapping coords to values for coloring
        show_coords: Whether to show hex coordinates
        show_system_ids: Whether to show system IDs
        title: Optional plot title
        size: Hexagon size
        cmap: Matplotlib colormap to use when color_by='value'
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        show_legend: Whether to show the legend

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    patches = []
    colors = []

    # Define colors for basic space types
    type_colors = {
        MapSpaceType.OPEN: '#f0f0f0',
        MapSpaceType.SYSTEM: '#e0e0e0',  # Base gray for generic systems
        MapSpaceType.HOME: '#ccffcc',    # Home systems are green
        MapSpaceType.CLOSED: '#333333',
        MapSpaceType.WARP: '#ffcc99'
    }

    # Create patches for hexes
    patches = []
    colors = []
    impassable_patches = [] # Cross-hatch + Thick Solid Border
    hazardous_patches = []  # Single-hatch + Thick Dashed Border

    for space in ti4_map.spaces:
        patches.append(create_hexagon_patch(space.coord, size))
        
        # Identify structural constraints (Anomalies)
        if space.system and space.system.anomalies:
            is_impassable = any(a.value in ["supernova", "asteroid_field"] for a in space.system.anomalies)
            if is_impassable:
                impassable_patches.append(create_hexagon_patch(space.coord, size))
            else:
                hazardous_patches.append(create_hexagon_patch(space.coord, size))

        # Determine color
        if color_by == 'type':
            color = type_colors.get(space.space_type, '#ffffff')
            if space.space_type == MapSpaceType.SYSTEM and space.system:
                if space.system.is_mecatol_rex():
                    color = '#ffff99'  # Mecatol yellow
                elif space.system.is_red():
                    color = '#ff9999'  # Red system
                elif space.system.is_blue():
                    color = '#99ccff'  # Blue system
        elif color_by == 'value' and value_map:
            val = value_map.get(space.coord, 0)
            color = val  # Will use colormap
        else:
            color = '#ffffff'

        colors.append(color)

    # Create patch collection
    if color_by == 'value' and value_map:
        collection = PatchCollection(patches, cmap=cmap, edgecolors='black', linewidths=1)
        collection.set_array(np.array(colors))
        if vmin is not None and vmax is not None:
            collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        
        # Triple Encoding Overlays
        # 1. Impassable (Stop-like): Dense cross-hatch + Bold Solid Border
        if impassable_patches:
            stop_coll = PatchCollection(impassable_patches, facecolors='none', edgecolors='black', 
                                       linewidths=3.0, linestyle='solid', hatch='xxx', alpha=0.6)
            ax.add_collection(stop_coll)
        
        # 2. Hazardous (Yield-like): Single diagonal hatch + Bold Dashed Border
        if hazardous_patches:
            yield_coll = PatchCollection(hazardous_patches, facecolors='none', edgecolors='black', 
                                        linewidths=2.5, linestyle='--', hatch='///', alpha=0.6)
            ax.add_collection(yield_coll)

        cbar = plt.colorbar(collection, ax=ax, label='Relative Strategic Value')
        from matplotlib.ticker import FuncFormatter
        def custom_format(x, pos):
            if x < 0: return f"(-{abs(x):g})"
            return f"{x:g}"
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_format))
    else:
        collection = PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=1)
        ax.add_collection(collection)
        
        if impassable_patches:
            stop_coll = PatchCollection(impassable_patches, facecolors='none', edgecolors='black', 
                                       linewidths=3.0, linestyle='solid', hatch='xxx', alpha=0.6)
            ax.add_collection(stop_coll)
        if hazardous_patches:
            yield_coll = PatchCollection(hazardous_patches, facecolors='none', edgecolors='black', 
                                        linewidths=2.5, linestyle='--', hatch='///', alpha=0.6)
            ax.add_collection(yield_coll)

    # Add Academic Legend
    if show_legend:
        from matplotlib.lines import Line2D
        
        # 1. Base constraints (Always relevant for these research plots)
        legend_elements = [
            Line2D([0], [0], color='black', lw=3, linestyle='solid', label='Movement Blocked (xxx)'),
            Line2D([0], [0], color='black', lw=2.5, linestyle='--', label='Movement Restricted (///)')
        ]
        
        # 2. Dynamic Wormholes (Only if they form a connection on the map)
        wh_counts = {}
        for space in ti4_map.spaces:
            if space.system and space.system.wormhole:
                val = space.system.wormhole.value
                wh_counts[val] = wh_counts.get(val, 0) + 1
                
        if wh_counts.get("alpha", 0) > 1:
            legend_elements.append(Line2D([0], [0], color='#D55E00', lw=2.5, linestyle='--', 
                                          dash_capstyle='round', label=r'Wormhole $\alpha$'))
        if wh_counts.get("beta", 0) > 1:
            legend_elements.append(Line2D([0], [0], color='#0072B2', lw=2.5, linestyle=':', 
                                          dash_capstyle='round', label=r'Wormhole $\beta$'))
        if wh_counts.get("delta", 0) > 1:
            legend_elements.append(Line2D([0], [0], color='#009E73', lw=2.5, linestyle='-.', 
                                          dash_capstyle='round', label=r'Wormhole $\delta$'))

        # Place legend inside the plot area in an empty corner to avoid colorbar collisions
        leg = ax.legend(handles=legend_elements, loc='upper right', 
                  title="Legend", frameon=True, shadow=True, ncol=1, fontsize=9)
        plt.setp(leg.get_title(), fontweight='bold')

    # Add text annotations
    for space in ti4_map.spaces:
        x, y = cube_to_pixel(space.coord, size)

        if show_coords:
            ax.text(x, y - 0.35, f"({space.coord.x},{space.coord.y})",
                   ha='center', va='center', fontsize=6, color='gray')

        if space.system:
            # Main label: ID or Anomaly/Wormhole
            labels = []
            if show_system_ids:
                labels.append(f"{space.system.id}")

                # Add anomaly indicators
                if space.system.anomalies:
                    for anomaly in space.system.anomalies:
                        # Short codes for anomalies
                        code = anomaly.value[0].upper()
                        if anomaly.value == "asteroid_field": code = "AF"
                        elif anomaly.value == "gravity_rift": code = "GR"
                        labels.append(code)
                
                # Add wormhole indicators
                if space.system.wormhole:
                    labels.append(f"WH-{space.system.wormhole.value[0].upper()}")

            if labels:
                main_text = "\n".join(labels)
                ax.text(x, y, main_text,
                       ha='center', va='center', fontsize=8, 
                       fontweight='bold', color='black')

        # Show home markers
        if space.space_type == MapSpaceType.HOME:
            ax.scatter([x], [y], s=250, c='red', marker='*', zorder=10, alpha=0.8, edgecolors='black')

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    return ax


def plot_balance_convergence(
    history: List[Tuple[int, float]],
    ax: Optional[plt.Axes] = None,
    title: str = "Balance Optimization Convergence"
) -> plt.Axes:
    """
    Plot balance gap convergence over iterations.

    Args:
        history: List of (iteration, balance_gap) tuples
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    iterations, gaps = zip(*history)

    ax.plot(iterations, gaps, linewidth=2, color='#2E86AB')
    ax.fill_between(iterations, gaps, alpha=0.3, color='#2E86AB')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Balance Gap', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add improvement annotation
    initial_gap = gaps[0]
    final_gap = gaps[-1]
    improvement = ((initial_gap - final_gap) / initial_gap) * 100

    ax.text(0.95, 0.95,
           f'Initial: {initial_gap:.2f}\nFinal: {final_gap:.2f}\nImprovement: {improvement:.1f}%',
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def plot_balance_distribution(
    home_values: List[HomeValue],
    ax: Optional[plt.Axes] = None,
    title: str = "Player Position Value Distribution"
) -> plt.Axes:
    """
    Plot distribution of home values across players.

    Args:
        home_values: List of HomeValue objects
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    values = [hv.value for hv in home_values]
    positions = [f"P{i+1}" for i in range(len(values))]

    # Bar plot
    bars = ax.bar(positions, values, color='#A23B72', alpha=0.7, edgecolor='black')

    # Add mean line
    mean_val = np.mean(values)
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Player Position', fontsize=12)
    ax.set_ylabel('Home Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    return ax


def plot_balance_comparison(
    before_values: List[float],
    after_values: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Balance Before vs After Optimization"
) -> plt.Axes:
    """
    Compare balance before and after optimization.

    Args:
        before_values: Home values before optimization
        after_values: Home values after optimization
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(before_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, before_values, width, label='Before',
                   color='#E63946', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, after_values, width, label='After',
                   color='#06A77D', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Player Position', fontsize=12)
    ax.set_ylabel('Home Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}' for i in range(len(before_values))])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add gap annotations
    before_gap = max(before_values) - min(before_values)
    after_gap = max(after_values) - min(after_values)

    ax.text(0.02, 0.98,
           f'Gap Before: {before_gap:.2f}\nGap After: {after_gap:.2f}\nReduction: {before_gap - after_gap:.2f}',
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def plot_value_heatmap(
    ti4_map: 'TI4Map',
    evaluator: Evaluator,
    ax: Optional[plt.Axes] = None,
    title: str = "System Value Heatmap",
    cmap: str = 'magma',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_legend: bool = True
) -> plt.Axes:
    """
    Create heatmap of system values across the map.

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluator for system values
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap for values
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        show_legend: Whether to show the legend

    Returns:
        Matplotlib axes
    """
    # Calculate values for each space
    value_map = {}
    for space in ti4_map.spaces:
        # The balance algorithm ignores the home system tiles themselves (MapSpaceType.HOME)
        # but IT DOES include Mecatol Rex (MapSpaceType.SYSTEM) in the value calculations
        # since it is accessible to all players.
        if space.space_type == MapSpaceType.SYSTEM and space.system:
            value_map[space.coord] = space.system.evaluate(evaluator)
        else:
            value_map[space.coord] = 0

    return plot_hex_map(
        ti4_map,
        ax=ax,
        color_by='value',
        value_map=value_map,
        show_system_ids=False,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_legend=show_legend
    )


def plot_delta_heatmap(
    map_a: 'TI4Map',
    map_b: 'TI4Map',
    evaluator: Evaluator,
    ax: Optional[plt.Axes] = None,
    title: str = "Spatial Difference Map (B - A)",
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_legend: bool = True
) -> plt.Axes:
    """
    Create spatial delta map showing differences between two maps.

    Args:
        map_a: Baseline map
        map_b: Target map
        evaluator: Evaluator for system values
        ax: Matplotlib axes
        title: Plot title
        cmap: Diverging colormap for differences
        vmin: Minimum value for colormap (forces symmetry if center=0)
        vmax: Maximum value for colormap
        show_legend: Whether to show the legend

    Returns:
        Matplotlib axes
    """
    # Calculate values for each space in both maps
    val_a = {s.coord: s.system.evaluate(evaluator) if s.system else 0 for s in map_a.spaces}
    val_b = {s.coord: s.system.evaluate(evaluator) if s.system else 0 for s in map_b.spaces}
    
    # Calculate difference
    delta_map = {coord: val_b.get(coord, 0) - val_a.get(coord, 0) for coord in val_b.keys()}
    
    # Determine bounds if not provided
    if vmin is None or vmax is None:
        max_abs_delta = max(abs(v) for v in delta_map.values()) if delta_map else 1.0
        if max_abs_delta == 0: max_abs_delta = 1.0
        vmin = -max_abs_delta
        vmax = max_abs_delta
    
    # Create the plot
    return plot_hex_map(
        map_b, # Use map_b as the layout scaffold
        ax=ax,
        color_by='value',
        value_map=delta_map,
        show_system_ids=False,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_legend=show_legend
    )


def annotate_player_slices(
    ti4_map: 'TI4Map',
    evaluator: Evaluator,
    ax: plt.Axes,
    size: float = HEX_RADIUS
) -> None:
    """
    Annotate player slices with aggregate Resource/Influence totals.
    
    Draws boundaries around systems closest to each home system and
    labels the slice with its total value.
    
    Args:
        ti4_map: TI4 map object
        evaluator: Evaluator for calculating values
        ax: Matplotlib axes to draw on
        size: Hexagon size
    """
    home_spaces = ti4_map.get_home_spaces()
    if not home_spaces:
        return

    # Assign each system space to the closest home system (Voronoi)
    slices: Dict[HexCoord, List[MapSpace]] = {h.coord: [] for h in home_spaces}
    
    for space in ti4_map.spaces:
        if space.space_type != MapSpaceType.SYSTEM or space.system is None:
            continue
            
        # Find closest home system by hex distance
        closest_home = min(home_spaces, key=lambda h: hex_distance(h.coord, space.coord))
        slices[closest_home.coord].append(space)

    # Calculate and draw totals for each slice
    for i, (home_coord, slice_spaces) in enumerate(slices.items()):
        total_res = sum(sum(p.resources for p in s.system.planets) for s in slice_spaces)
        total_inf = sum(sum(p.influence for p in s.system.planets) for s in slice_spaces)
        
        # Calculate pixel coordinates for the home system
        hx, hy = cube_to_pixel(home_coord, size)
        
        # Push label outward radially from the center (0,0) to be near the home tile
        dist = np.hypot(hx, hy)
        if dist > 0:
            lx = hx + (hx / dist) * (1.2 * size)
            ly = hy + (hy / dist) * (1.2 * size)
        else:
            lx, ly = hx, hy + 1.2 * size
        
        label_text = f"P{i+1}\nR:{total_res} I:{total_inf}"
        ax.text(lx, ly, label_text,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

        # Optional: Draw slice boundaries by finding edges between different slice assignments
        # This is complex for matplotlib, skipping for now in favor of clear labeling


def highlight_defects(
    coords: List[HexCoord],
    ax: plt.Axes,
    color: str = 'red',
    label: Optional[str] = None,
    size: float = HEX_RADIUS
) -> None:
    """
    Circle specific hexes to highlight topology defects or hotspots.
    
    Args:
        coords: List of hex coordinates to highlight
        ax: Matplotlib axes
        color: Circle color
        label: Optional text label
        size: Hexagon size
    """
    for coord in coords:
        x, y = cube_to_pixel(coord, size)
        circle = plt.Circle((x, y), size * 0.8, color=color, fill=False, linewidth=3, linestyle='--')
        ax.add_patch(circle)
        
        if label:
            ax.text(x, y + size, label, color=color, fontweight='bold', ha='center', va='bottom')


def plot_comparison_diptych(
    map_a: 'TI4Map',
    map_b: 'TI4Map',
    evaluator: Evaluator,
    title_a: str = "Baseline Map",
    title_b: str = "Optimized Map",
    figsize: Tuple[int, int] = (20, 10),
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None
) -> plt.Figure:
    """
    Create a two-panel side-by-side comparison of two maps with standardized scaling.
    
    Args:
        map_a: First map (usually baseline/random)
        map_b: Second map (usually optimized)
        evaluator: Evaluator for calculating values
        title_a: Title for left plot
        title_b: Title for right plot
        figsize: Figure size
        global_vmin: Global minimum for color scaling
        global_vmax: Global maximum for color scaling
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # If bounds not provided, calculate from both maps to ensure consistency
    if global_vmin is None or global_vmax is None:
        vals_a = [s.system.evaluate(evaluator) for s in map_a.spaces if s.system]
        vals_b = [s.system.evaluate(evaluator) for s in map_b.spaces if s.system]
        all_vals = vals_a + vals_b
        if not all_vals: all_vals = [0, 1]
        global_vmin = min(all_vals)
        global_vmax = max(all_vals)

    plot_value_heatmap(map_a, evaluator, ax=ax1, title=title_a, vmin=global_vmin, vmax=global_vmax, show_legend=False)
    plot_value_heatmap(map_b, evaluator, ax=ax2, title=title_b, vmin=global_vmin, vmax=global_vmax, show_legend=True)
    
    plt.tight_layout()
    return fig


def create_balance_report(
    ti4_map: 'TI4Map',
    evaluator: Evaluator,
    history: Optional[List[Tuple[int, float]]] = None,
    figsize: Tuple[int, int] = (16, 12),
    global_vmin: Optional[float] = None,
    global_vmax: Optional[float] = None
) -> plt.Figure:
    """
    Create comprehensive balance analysis report with standardized scaling.

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluator parameters
        history: Optional optimization history
        figsize: Figure size
        global_vmin: Global minimum for color scaling
        global_vmax: Global maximum for color scaling

    Returns:
        Matplotlib figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Hex map with slice annotations
    ax1 = fig.add_subplot(gs[0, 0])
    plot_hex_map(ti4_map, ax=ax1, title="Map Layout & Slices")
    annotate_player_slices(ti4_map, evaluator, ax=ax1)

    # Plot 2: Value heatmap with global normalization
    ax2 = fig.add_subplot(gs[0, 1])
    plot_value_heatmap(ti4_map, evaluator, ax=ax2, vmin=global_vmin, vmax=global_vmax)

    # Plot 3: Balance distribution
    ax3 = fig.add_subplot(gs[1, 0])
    home_values = get_home_values(ti4_map, evaluator)
    plot_balance_distribution(home_values, ax=ax3)

    # Plot 4: Convergence (if history provided)
    if history:
        ax4 = fig.add_subplot(gs[1, 1])
        plot_balance_convergence(history, ax=ax4)
    else:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.text(0.5, 0.5, 'No optimization history',
                ha='center', va='center', fontsize=14)
        ax4.axis('off')

    fig.suptitle('TI4 Map Balance Analysis (Standardized Scaling)', fontsize=16, fontweight='bold', y=0.98)

    return fig


def plot_fairness_metrics(
    analysis_results: Dict,
    ax: Optional[plt.Axes] = None,
    title: str = "Balance Fairness Metrics"
) -> plt.Axes:
    """
    Visualize fairness metrics from balance analysis.

    Args:
        analysis_results: Results from analyze_balance()
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    metrics = {
        'Balance Gap': analysis_results['balance_gap'],
        'Std Dev': analysis_results['std'],
        'Mean': analysis_results['mean'],
        'Fairness Index': analysis_results['fairness_index'] * 100  # Scale to 0-100
    }

    y_pos = np.arange(len(metrics))
    values = list(metrics.values())
    labels = list(metrics.keys())

    bars = ax.barh(y_pos, values, color=['#E63946', '#F77F00', '#06A77D', '#118AB2'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f' {val:.2f}', va='center', fontweight='bold')

    ax.grid(True, axis='x', alpha=0.3)

    return ax
