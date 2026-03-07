#!/usr/bin/env python3
"""
Side-by-side comparison of SLV for 5-player vs 6-player maps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.algorithms.balance_engine import create_joebrew_evaluator
from ti4_analysis.spatial_stats.spatial_metrics import compute_slice_level_variance
from ti4_analysis.data.tile_loader import load_tile_database
from ti4_analysis.data.map_structures import MapSpaceType
import numpy as np

def analyze_map_slices(map_obj, evaluator, player_count, template):
    """Analyze slice distribution for all homes."""
    from ti4_analysis.algorithms.balance_engine import create_ti4_cost_function
    from heapq import heappush, heappop
    
    cost_fn = create_ti4_cost_function(evaluator, map_obj._coord_to_space)
    max_cost = 5 * 2.0
    
    homes = map_obj.get_home_spaces()
    
    print(f"\n{'='*70}")
    print(f"{player_count}-Player Map (template: {template})")
    print(f"{'='*70}")
    
    systems = [s for s in map_obj.spaces if s.space_type == MapSpaceType.SYSTEM]
    print(f"Map has {len(systems)} systems, {len(homes)} homes")
    
    all_home_variances = []
    
    for home_idx, home in enumerate(homes):
        slice_totals = {}
        frontier = [(0.0, 0, home.coord, None, (home.coord,))]
        best_costs = {home.coord: 0.0}
        processed = {home.coord}
        
        counter = 0
        while frontier:
            cost, _, current_coord, first_step, path = heappop(frontier)
            
            if current_coord in processed and current_coord != home.coord:
                continue
            
            if current_coord != home.coord:
                processed.add(current_coord)
            
            if cost > max_cost:
                continue
            
            current_space = map_obj.get_space(current_coord)
            if current_space is None:
                continue
            
            if (current_coord != home.coord and 
                current_space.space_type == MapSpaceType.SYSTEM and
                current_space.system is not None and 
                first_step is not None):
                
                system_value = current_space.system.evaluate(evaluator)
                if system_value > 0:
                    weight = evaluator.get_distance_multiplier(cost)
                    if weight > 0:
                        weighted_value = weight * system_value
                        slice_totals[first_step] = slice_totals.get(first_step, 0.0) + weighted_value
            
            neighbors = map_obj.get_adjacent_spaces_including_wormholes(current_space)
            for neighbor in neighbors:
                if neighbor.coord in processed:
                    continue
                
                move_cost = cost_fn(current_coord, neighbor.coord, cost, list(path))
                if move_cost is None:
                    continue
                
                new_cost = cost + move_cost
                if new_cost > max_cost:
                    continue
                
                best_known = best_costs.get(neighbor.coord)
                if best_known is not None and new_cost >= best_known - 1e-9:
                    continue
                
                next_first_step = first_step if first_step is not None else neighbor.coord
                best_costs[neighbor.coord] = new_cost
                counter += 1
                heappush(frontier, (new_cost, counter, neighbor.coord, next_first_step, path + (neighbor.coord,)))
        
        # Calculate variance for this home
        if slice_totals:
            values = np.array(list(slice_totals.values()), dtype=float)
            variance = float(np.var(values))
            all_home_variances.append(variance)
            
            print(f"\nHome {home_idx} at {home.coord}:")
            print(f"  Slices: {len(slice_totals)}")
            print(f"  Processed systems: {len(processed) - 1}")
            print(f"  Slice distribution:")
            
            sorted_slices = sorted(slice_totals.items(), key=lambda x: x[1], reverse=True)
            total = sum(slice_totals.values())
            for first_step, value in sorted_slices:
                pct = (value / total * 100) if total > 0 else 0
                print(f"    {first_step}: {value:.1f} ({pct:.1f}%)")
            
            print(f"  Variance: {variance:.2f}")
            print(f"  Mean: {np.mean(values):.2f}")
            print(f"  Ratio (max/min): {np.max(values)/np.min(values):.2f}x")
    
    # Overall SLV
    mean_variance = np.mean(all_home_variances) if all_home_variances else 0.0
    print(f"\n{'='*70}")
    print(f"Overall SLV: {mean_variance:.2f}")
    print(f"{'='*70}")
    
    return mean_variance

def compare_maps():
    project_root = Path(__file__).parent.parent
    tile_db = load_tile_database(project_root=project_root)
    evaluator = create_joebrew_evaluator()
    
    print("\n" + "="*70)
    print("COMPARING 5-PLAYER vs 6-PLAYER SLV")
    print("="*70)
    
    # Generate both maps with same seed for fair comparison
    seed = 42
    
    # 5-player warp
    map_5p = generate_random_map(
        player_count=5,
        template_name="warp",
        include_pok=True,
        include_thunders_edge=True,
        random_seed=seed,
        tile_db=tile_db,
        project_root=project_root
    )
    
    # 6-player normal
    map_6p = generate_random_map(
        player_count=6,
        template_name="normal",
        include_pok=True,
        include_thunders_edge=True,
        random_seed=seed,
        tile_db=tile_db,
        project_root=project_root
    )
    
    # Analyze both
    slv_5p = analyze_map_slices(map_5p, evaluator, 5, "warp")
    slv_6p = analyze_map_slices(map_6p, evaluator, 6, "normal")
    
    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"5-Player SLV: {slv_5p:.2f}")
    print(f"6-Player SLV: {slv_6p:.2f}")
    print(f"Ratio (5p/6p): {slv_5p/slv_6p:.2f}x")
    
    # Verify with built-in function
    print(f"\nVerifying with built-in compute_slice_level_variance():")
    slv_5p_builtin = compute_slice_level_variance(map_5p, evaluator)
    slv_6p_builtin = compute_slice_level_variance(map_6p, evaluator)
    print(f"5-Player: {slv_5p_builtin:.2f}")
    print(f"6-Player: {slv_6p_builtin:.2f}")
    
    # Expected values from comprehensive analysis
    print(f"\nExpected values from comprehensive analysis:")
    print(f"  Random maps: 2,000 - 2,900")
    print(f"  Optimized maps: 600 - 900")
    
    if slv_5p > 10000:
        print(f"\n❌ 5-Player SLV is EXTREMELY HIGH ({slv_5p:.0f})")
        print(f"   This suggests severe imbalance in map generation")
    elif slv_5p > 3000:
        print(f"\n⚠️  5-Player SLV is HIGH ({slv_5p:.0f})")
        print(f"   Worse than random maps from comprehensive analysis")
    else:
        print(f"\n✓ 5-Player SLV seems reasonable ({slv_5p:.0f})")

if __name__ == "__main__":
    compare_maps()
