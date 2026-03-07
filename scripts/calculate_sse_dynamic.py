"""
Dynamic SSE Calculator - Tile-Based Economy

This calculates SSE by dynamically computing economy from tile counts,
which is the foundation for optimizing the number of tiles per player count.

METHODOLOGY (Updated for academic rigor):
- Use TILE-TYPE-SPECIFIC averages from the full tile pool (not blended averages)
- Blue tiles: High value (avg 2.26 res, 2.50 inf per system)
- Red tiles: Low value (avg 0.26 res, 0.05 inf per system - mostly empty/anomalies)
- Home systems: Calculate from faction data
- Mecatol: Fixed at 1 res, 6 inf

This approach is more accurate than blended averages because it properly accounts
for the different characteristics of tile types when the blue/red ratio changes.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')

# Load tile pool statistics from systems.json
with open(DATA_DIR / 'systems.json') as f:
    _systems_data = json.load(f)
    _pool = _systems_data['pool_statistics']

# TILE-TYPE-SPECIFIC AVERAGES (from full pool)
# These are the CORRECT values for calculating expected map composition
BLUE_AVG_RESOURCES = _pool['blue']['avg_resources_per_system']  # ~2.26
BLUE_AVG_INFLUENCE = _pool['blue']['avg_influence_per_system']  # ~2.50
BLUE_AVG_PLANETS = _pool['blue']['avg_planets_per_system']      # ~1.58

RED_AVG_RESOURCES = _pool['red']['total_resources'] / _pool['red']['count']  # ~0.26
RED_AVG_INFLUENCE = _pool['red']['total_influence'] / _pool['red']['count']  # ~0.05
RED_AVG_PLANETS = _pool['red']['total_planets'] / _pool['red']['count']      # ~0.11

# Mecatol Rex (fixed values)
MECATOL_RESOURCES = 1
MECATOL_INFLUENCE = 6
MECATOL_PLANETS = 1

# Home system averages (calculated from faction data)
# Most factions have 1-2 planets totaling 3-5 resources
HOME_AVG_RESOURCES = 3.7  # Average across all factions
HOME_AVG_INFLUENCE = 2.1
HOME_AVG_PLANETS = 1.5

# Minor faction systems (Trade Wars variant)
MINOR_FACTION_AVG_PLANETS = 2.5  # Approximate

# Load faction data for commodities
df_factions = pd.read_csv(DATA_DIR / 'combined_faction_clusters.csv')
AVG_COMMODITIES = df_factions['commodities'].mean()

# Load objectives
with open(DATA_DIR / 'objectives.json') as f:
    objectives_data = json.load(f)


def calculate_map_stats(blue_tiles, red_tiles, player_count,
                        minor_factions=0, space_stations=0, trade_wars=False):
    """
    Calculate map statistics from tile counts.

    This is the DYNAMIC calculation that allows optimization.

    METHODOLOGY: Use tile-type-specific averages for accuracy.
    - Blue tiles contribute high resources/influence
    - Red tiles contribute minimal resources (mostly empty)
    - Home systems and Mecatol are calculated separately
    """
    # System counts
    home_systems = player_count
    total_systems = blue_tiles + red_tiles + home_systems + 1 + minor_factions  # +1 for Mecatol
    non_home_systems = blue_tiles + red_tiles + 1 + minor_factions

    # TILE-TYPE-SPECIFIC RESOURCE CALCULATION
    # This is more accurate than a blended average
    blue_resources = blue_tiles * BLUE_AVG_RESOURCES
    blue_influence = blue_tiles * BLUE_AVG_INFLUENCE
    blue_planets = blue_tiles * BLUE_AVG_PLANETS

    red_resources = red_tiles * RED_AVG_RESOURCES
    red_influence = red_tiles * RED_AVG_INFLUENCE
    red_planets = red_tiles * RED_AVG_PLANETS

    home_resources = player_count * HOME_AVG_RESOURCES
    home_influence = player_count * HOME_AVG_INFLUENCE
    home_planets = player_count * HOME_AVG_PLANETS

    minor_resources = minor_factions * BLUE_AVG_RESOURCES  # Minor factions are valuable
    minor_influence = minor_factions * BLUE_AVG_INFLUENCE
    minor_planets = minor_factions * MINOR_FACTION_AVG_PLANETS

    # Total expected resources and influence
    total_resources = (blue_resources + red_resources + home_resources +
                       MECATOL_RESOURCES + minor_resources)
    total_influence = (blue_influence + red_influence + home_influence +
                       MECATOL_INFLUENCE + minor_influence)

    # Non-home planets (for planet-control objectives)
    non_home_planets = blue_planets + red_planets + MECATOL_PLANETS + minor_planets
    total_planets = non_home_planets + home_planets

    # Per-player values
    resources_per_player = total_resources / player_count
    influence_per_player = total_influence / player_count

    # Trade goods calculation
    if trade_wars:
        # Space stations enable full trade meta + boost commodities
        boosted_commodities = AVG_COMMODITIES + space_stations  # +1 per space station
        tg_per_round = (3 + boosted_commodities * player_count) / player_count
    elif player_count == 2:
        tg_per_round = 3  # No trade meta in 2P base
    elif player_count == 3:
        tg_per_round = 2.5  # Reluctant trade in 3P base
    else:
        # Full X-1 meta for 4-8P
        tg_per_round = (3 + AVG_COMMODITIES * player_count) / player_count

    # Total economy per player
    total_economy = resources_per_player + influence_per_player + tg_per_round

    # Edge systems (for objectives)
    edge_systems = blue_tiles + red_tiles  # Approximate: outer ring tiles
    if trade_wars:
        edge_systems += minor_factions

    # Systems without planets (for patrol objectives)
    # ~89% of red tiles are empty (17 out of 19 in pool)
    empty_red_fraction = (_pool['red']['count'] - 2) / _pool['red']['count']  # 17/19
    systems_without_planets = red_tiles * empty_red_fraction

    return {
        'player_count': player_count,
        'blue_tiles': blue_tiles,
        'red_tiles': red_tiles,
        'minor_factions': minor_factions,
        'space_stations': space_stations,
        'trade_wars': trade_wars,
        'total_systems': total_systems,
        'non_home_systems': non_home_systems,
        'non_home_planets': non_home_planets,
        'total_planets': total_planets,
        'total_resources': total_resources,
        'total_influence': total_influence,
        'resources_per_player': resources_per_player,
        'influence_per_player': influence_per_player,
        'tg_per_round': tg_per_round,
        'total_economy': total_economy,
        'edge_systems': edge_systems,
        'edge_systems_per_player': edge_systems / player_count,
        'systems_without_planets': systems_without_planets,
        'systems_without_planets_per_player': systems_without_planets / player_count,
        'non_home_planets_per_player': non_home_planets / player_count,
        'mecatol_adjacent_per_player': 1.0,  # Always 1 for fair maps
    }


def calculate_objective_difficulty(obj, stats):
    """Calculate difficulty for one objective given map stats."""
    req_type = obj.get('requirement_type')
    req_value = obj.get('requirement_value')
    pc = stats['player_count']

    if req_type == 'spend_resources':
        avail = stats['resources_per_player'] + stats['tg_per_round']
        return req_value / avail if avail > 0 else None
    elif req_type == 'spend_influence':
        avail = stats['influence_per_player'] + stats['tg_per_round']
        return req_value / avail if avail > 0 else None
    elif req_type == 'spend_trade_goods':
        avail = stats['tg_per_round']
        return req_value / avail if avail > 0 else None
    elif req_type == 'spend_combined':
        if isinstance(req_value, dict):
            total_req = sum(req_value.values())
            return total_req / stats['total_economy']
        return None
    elif req_type == 'edge_systems':
        return req_value / stats['edge_systems_per_player']
    elif req_type == 'adjacent_to_mecatol':
        return req_value / stats['mecatol_adjacent_per_player']
    elif req_type == 'empty_systems':
        return req_value / stats['systems_without_planets_per_player']
    elif req_type == 'non_home_planets':
        return req_value / stats['non_home_planets_per_player']
    elif req_type == 'control_resources':
        return req_value / stats['resources_per_player']
    elif req_type == 'control_influence':
        return req_value / stats['influence_per_player']
    elif req_type == 'total_systems':
        return req_value / stats['edge_systems_per_player']
    else:
        return None


def calculate_sse(stats, baseline_stats):
    """
    Calculate SSE using EXACT Excel methodology.

    Formula: AVERAGE(((scenario - baseline) / baseline)^2) for each category
    """
    errors = {
        'stage_1': [],
        'stage_2': [],
        'secret': [],
    }

    # Calculate objective errors
    for stage in ['stage_1', 'stage_2', 'secret']:
        for obj in objectives_data[stage]:
            if not obj.get('map_dependent', False):
                continue

            scenario_diff = calculate_objective_difficulty(obj, stats)
            baseline_diff = calculate_objective_difficulty(obj, baseline_stats)

            if scenario_diff is not None and baseline_diff is not None and baseline_diff > 0:
                normalized_error = ((scenario_diff - baseline_diff) / baseline_diff) ** 2
                errors[stage].append(normalized_error)

    # Average errors per stage
    stage_errors = {
        'stage_1': np.mean(errors['stage_1']) if errors['stage_1'] else 0,
        'stage_2': np.mean(errors['stage_2']) if errors['stage_2'] else 0,
        'secret': np.mean(errors['secret']) if errors['secret'] else 0,
    }

    # Economy error (normalized)
    economy_error = ((stats['total_economy'] - baseline_stats['total_economy']) /
                     baseline_stats['total_economy']) ** 2

    # Total SSE
    total_sse = economy_error + stage_errors['stage_1'] + stage_errors['stage_2'] + stage_errors['secret']

    return {
        'economy_error': economy_error,
        'stage_1_error': stage_errors['stage_1'],
        'stage_2_error': stage_errors['stage_2'],
        'secret_error': stage_errors['secret'],
        'total_sse': total_sse,
    }


# Define configurations matching user's spreadsheet
def create_configs():
    """Create configurations matching user's Excel data."""
    configs = {}

    # Standard configs (4-8P) - from spreadsheet row 3-4
    tile_counts = {
        8: (24, 16),
        7: (21, 14),
        6: (18, 12),
        5: (15, 10),
        4: (14, 10),
    }

    for pc, (blue, red) in tile_counts.items():
        configs[f'{pc}p_base'] = calculate_map_stats(blue, red, pc)

    # 3P and 2P with Trade Wars (space stations, minor factions)
    # From spreadsheet: 3 space stations, 3 minor faction systems
    configs['3p_trade_wars'] = calculate_map_stats(
        blue_tiles=10, red_tiles=8, player_count=3,
        minor_factions=3, space_stations=3, trade_wars=True
    )
    configs['2p_trade_wars'] = calculate_map_stats(
        blue_tiles=8, red_tiles=7, player_count=2,
        minor_factions=3, space_stations=3, trade_wars=True
    )

    # 3P and 2P BASE (without Trade Wars) for comparison
    configs['3p_base'] = calculate_map_stats(
        blue_tiles=10, red_tiles=8, player_count=3
    )
    configs['2p_base'] = calculate_map_stats(
        blue_tiles=8, red_tiles=7, player_count=2
    )

    return configs


if __name__ == '__main__':
    print("=" * 80)
    print("DYNAMIC SSE CALCULATOR - Tile-Based Economy")
    print("=" * 80)
    print()

    configs = create_configs()
    baseline = configs['6p_base']

    # Show map statistics
    print("MAP STATISTICS (Per Player)")
    print("-" * 80)
    print(f"{'Config':<18} {'Blue':>5} {'Red':>5} {'Res/P':>7} {'Inf/P':>7} {'TG':>6} {'Total':>7}")
    print("-" * 80)

    for name, stats in configs.items():
        print(f"{name:<18} {stats['blue_tiles']:>5} {stats['red_tiles']:>5} "
              f"{stats['resources_per_player']:>7.1f} {stats['influence_per_player']:>7.1f} "
              f"{stats['tg_per_round']:>6.1f} {stats['total_economy']:>7.1f}")
    print()

    # Calculate and show SSE
    print("SSE ANALYSIS (vs 6P Baseline)")
    print("-" * 80)
    print(f"{'Config':<18} {'Econ Err':>9} {'Stage1':>8} {'Stage2':>8} {'Secret':>8} {'Total SSE':>10}")
    print("-" * 80)

    results = []
    for name, stats in configs.items():
        sse = calculate_sse(stats, baseline)
        results.append({
            'name': name,
            'stats': stats,
            'sse': sse,
        })
        print(f"{name:<18} {sse['economy_error']:>9.4f} {sse['stage_1_error']:>8.4f} "
              f"{sse['stage_2_error']:>8.4f} {sse['secret_error']:>8.4f} {sse['total_sse']:>10.4f}")
    print()

    # Comparison with user's Excel values
    print("COMPARISON WITH EXCEL VALUES")
    print("-" * 80)
    excel_sse = {
        '8p_base': 0.17,
        '7p_base': 0.05,
        '6p_base': 0.00,
        '5p_base': 0.09,
        '4p_base': 0.34,
        '3p_trade_wars': 1.11,  # User's spreadsheet has Trade Wars for 2P/3P
        '2p_trade_wars': 2.73,
    }

    print(f"{'Config':<18} {'Calculated':>12} {'Excel':>12} {'Diff':>12}")
    print("-" * 80)
    for name, excel_val in excel_sse.items():
        if name in configs:
            calc_sse = calculate_sse(configs[name], baseline)['total_sse']
            diff = calc_sse - excel_val
            print(f"{name:<18} {calc_sse:>12.4f} {excel_val:>12.2f} {diff:>+12.4f}")
    print()

    # Show ranking
    print("RANKING (Lower = Better)")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda x: x['sse']['total_sse'])
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r['name']}: SSE = {r['sse']['total_sse']:.4f}")
    print()

    # Save results
    output = {
        'configs': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                       for kk, vv in v.items()} for k, v in configs.items()},
        'sse_results': {r['name']: {kk: float(vv) for kk, vv in r['sse'].items()} for r in results},
    }
    with open(DATA_DIR / 'dynamic_sse_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {DATA_DIR / 'dynamic_sse_results.json'}")
