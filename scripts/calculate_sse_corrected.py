"""
CORRECTED SSE Calculation - Dynamic Objective Difficulties

Key insight: Difficulty should be calculated based on ACTUAL economy available
in each scenario, not pre-calculated values.

For spend objectives:
- difficulty = requirement / economy_available_per_round
- Lower difficulty = easier (can be done in fraction of a round)
- Higher difficulty = harder (needs >1 round of economy)

The 6P baseline is our target - we want all scenarios to have
similar difficulties to 6P.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')

# Load data
df_factions = pd.read_csv(DATA_DIR / 'combined_faction_clusters.csv')
AVG_COMMODITIES = df_factions['commodities'].mean()

with open(DATA_DIR / 'objectives.json') as f:
    objectives_data = json.load(f)

print("=" * 80)
print("CORRECTED SSE CALCULATION - DYNAMIC OBJECTIVE DIFFICULTIES")
print("=" * 80)
print()

ROUNDS = 5

# Define all scenario configurations
def create_configs():
    configs = {}

    # 2P BASE GAME
    pc = 2
    blue = 6
    resources_per_round = (blue * 1.5) / pc  # 4.5
    influence_per_round = (blue * 1.5) / pc  # 4.5
    tg_per_round = 3  # No trade meta, only organic
    configs['2p_base'] = {
        'name': '2P Base',
        'player_count': 2,
        'variant': 'base',
        'blue_tiles': blue,
        'red_tiles': 4,
        'system_tiles': 10,
        'resources_per_round': resources_per_round,
        'influence_per_round': influence_per_round,
        'tg_per_round': tg_per_round,
        'total_economy_per_round': resources_per_round + influence_per_round + tg_per_round,
        'edge_systems': 6,
        'mecatol_adjacent': 2,
        'empty_systems': 4,  # red tiles estimate
        'total_planets': blue * 1.5,  # estimate ~1.5 planets per blue tile
    }

    # 2P TRADE WARS
    pc = 2
    blue = 9
    minor_faction_planets = 7.5  # 3 systems × 2.5 planets avg
    total_blue_equivalent = blue + minor_faction_planets
    resources_per_round = (total_blue_equivalent * 1.5) / pc
    influence_per_round = (total_blue_equivalent * 1.5) / pc
    # Full Trade meta with space station boost
    boosted_commodities = AVG_COMMODITIES + (3 / pc)  # +1.5 from space stations
    tg_per_round = (3 + boosted_commodities * pc) / pc + (1.5 / pc)  # + monument bonus
    configs['2p_trade_wars'] = {
        'name': '2P Trade Wars',
        'player_count': 2,
        'variant': 'trade_wars',
        'blue_tiles': blue,
        'red_tiles': 8,
        'system_tiles': 17,
        'resources_per_round': resources_per_round,
        'influence_per_round': influence_per_round,
        'tg_per_round': tg_per_round,
        'total_economy_per_round': resources_per_round + influence_per_round + tg_per_round,
        'edge_systems': 9,
        'mecatol_adjacent': 2,
        'empty_systems': 8,
        'total_planets': total_blue_equivalent * 1.5,
        'minor_faction_systems': 3,
    }

    # 3P BASE GAME
    pc = 3
    blue = 9
    resources_per_round = (blue * 1.5) / pc
    influence_per_round = (blue * 1.5) / pc
    tg_per_round = 2.5  # Reluctant trade
    configs['3p_base'] = {
        'name': '3P Base',
        'player_count': 3,
        'variant': 'base',
        'blue_tiles': blue,
        'red_tiles': 6,
        'system_tiles': 15,
        'resources_per_round': resources_per_round,
        'influence_per_round': influence_per_round,
        'tg_per_round': tg_per_round,
        'total_economy_per_round': resources_per_round + influence_per_round + tg_per_round,
        'edge_systems': 9,
        'mecatol_adjacent': 3,
        'empty_systems': 6,
        'total_planets': blue * 1.5,
    }

    # 3P TRADE WARS
    pc = 3
    blue = 10
    minor_faction_planets = 7.5
    total_blue_equivalent = blue + minor_faction_planets
    resources_per_round = (total_blue_equivalent * 1.5) / pc
    influence_per_round = (total_blue_equivalent * 1.5) / pc
    boosted_commodities = AVG_COMMODITIES + (3 / pc)
    tg_per_round = (3 + boosted_commodities * pc) / pc + (1.5 / pc)
    configs['3p_trade_wars'] = {
        'name': '3P Trade Wars',
        'player_count': 3,
        'variant': 'trade_wars',
        'blue_tiles': blue,
        'red_tiles': 8,
        'system_tiles': 18,
        'resources_per_round': resources_per_round,
        'influence_per_round': influence_per_round,
        'tg_per_round': tg_per_round,
        'total_economy_per_round': resources_per_round + influence_per_round + tg_per_round,
        'edge_systems': 10,
        'mecatol_adjacent': 3,
        'empty_systems': 8,
        'total_planets': total_blue_equivalent * 1.5,
        'minor_faction_systems': 3,
    }

    # 4P-8P BASE GAME (universal scaling)
    for pc in [4, 5, 6, 7, 8]:
        blue = pc * 3
        red = pc * 2
        resources_per_round = (blue * 1.5) / pc
        influence_per_round = (blue * 1.5) / pc
        tg_per_round = (3 + AVG_COMMODITIES * pc) / pc  # Full Trade meta
        configs[f'{pc}p_base'] = {
            'name': f'{pc}P Base',
            'player_count': pc,
            'variant': 'base',
            'blue_tiles': blue,
            'red_tiles': red,
            'system_tiles': blue + red,
            'resources_per_round': resources_per_round,
            'influence_per_round': influence_per_round,
            'tg_per_round': tg_per_round,
            'total_economy_per_round': resources_per_round + influence_per_round + tg_per_round,
            'edge_systems': pc * 3,
            'mecatol_adjacent': pc * 1,
            'empty_systems': red,
            'total_planets': blue * 1.5,
        }

    return configs

configs = create_configs()

# 6P baseline for comparison
BASELINE = configs['6p_base']
print(f"6P Baseline (per round):")
print(f"  Resources: {BASELINE['resources_per_round']:.2f}")
print(f"  Influence: {BASELINE['influence_per_round']:.2f}")
print(f"  Trade Goods: {BASELINE['tg_per_round']:.2f}")
print(f"  Total: {BASELINE['total_economy_per_round']:.2f}")
print()

# Calculate difficulty for each objective in each scenario
def calculate_objective_difficulty(obj, config):
    """
    Calculate difficulty for an objective in a given scenario.

    Difficulty = requirement / available_per_round
    - <1.0: Can be done with less than 1 round of economy (easy)
    - 1.0: Requires exactly 1 round of economy
    - >1.0: Requires more than 1 round of economy (hard/impossible per round)
    """
    req_type = obj.get('requirement_type')
    req_value = obj.get('requirement_value')
    pc = config['player_count']

    # SPEND OBJECTIVES - use per-round economy
    if req_type == 'spend_resources':
        available = config['resources_per_round'] + config['tg_per_round']  # TG can substitute
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'spend_influence':
        available = config['influence_per_round'] + config['tg_per_round']  # TG can substitute
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'spend_trade_goods':
        available = config['tg_per_round']
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'spend_combined':
        if isinstance(req_value, dict):
            req_r = req_value.get('resources', 0)
            req_i = req_value.get('influence', 0)
            req_tg = req_value.get('trade_goods', 0)
            total_required = req_r + req_i + req_tg
            total_available = config['total_economy_per_round']
            return total_required / total_available if total_available > 0 else float('inf')
        return None

    # MAP STRUCTURE OBJECTIVES - use map availability
    elif req_type == 'edge_systems':
        # Assume equal sharing of map
        available = config['edge_systems'] / pc
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'adjacent_to_mecatol':
        available = config['mecatol_adjacent'] / pc
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'empty_systems':
        available = config['empty_systems'] / pc
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'non_home_planets':
        # Total non-home planets available per player
        available = config['total_planets'] / pc
        return req_value / available if available > 0 else float('inf')

    # CONTROL OBJECTIVES - use total controlled planets
    elif req_type == 'control_resources':
        # Total resources controlled (assume ~1.5 resources per planet)
        total_resources = config['total_planets'] * 1.5 / pc
        return req_value / total_resources if total_resources > 0 else float('inf')

    elif req_type == 'control_influence':
        total_influence = config['total_planets'] * 1.5 / pc
        return req_value / total_influence if total_influence > 0 else float('inf')

    elif req_type == 'total_systems':
        # Systems you can have ships in
        available = config['system_tiles'] / pc
        return req_value / available if available > 0 else float('inf')

    # Non-map-dependent objectives - return None (can't calculate from map)
    else:
        return None

# Calculate difficulties for all map-dependent objectives
print("=" * 80)
print("OBJECTIVE DIFFICULTY BY SCENARIO (per-round basis)")
print("=" * 80)
print()

# Collect all map-dependent objectives
map_dependent_objectives = []
for stage in ['stage_1', 'stage_2', 'secret']:
    for obj in objectives_data[stage]:
        if obj.get('map_dependent', False):
            obj['stage'] = stage
            map_dependent_objectives.append(obj)

print(f"Map-dependent objectives: {len(map_dependent_objectives)}")
print()

# Calculate difficulties
all_difficulties = {}
for config_name, config in configs.items():
    all_difficulties[config_name] = {}
    for obj in map_dependent_objectives:
        diff = calculate_objective_difficulty(obj, config)
        if diff is not None:
            all_difficulties[config_name][obj['name']] = diff

# Show spend objectives comparison (most impacted by economy)
print("=" * 80)
print("SPEND OBJECTIVES - DIFFICULTY COMPARISON")
print("(Lower = easier to achieve per round)")
print("=" * 80)
print()

spend_objectives = [
    ("Negotiate Trade Routes", "spend_trade_goods", 5),
    ("Erect a Monument", "spend_resources", 8),
    ("Sway the Council", "spend_influence", 8),
    ("Amass Wealth", "spend_combined", {"influence": 3, "resources": 3, "trade_goods": 3}),
    ("Centralize Galactic Trade", "spend_trade_goods", 10),
    ("Found a Golden Age", "spend_resources", 16),
    ("Manipulate Galactic Law", "spend_influence", 16),
    ("Hold Vast Reserves", "spend_combined", {"influence": 6, "resources": 6, "trade_goods": 6}),
]

spend_rows = []
for obj_name, req_type, req_value in spend_objectives:
    row = {'Objective': obj_name}
    for config_name, config in configs.items():
        if obj_name in all_difficulties[config_name]:
            diff = all_difficulties[config_name][obj_name]
            baseline_diff = all_difficulties['6p_base'].get(obj_name, 1.0)
            pct_vs_baseline = ((diff - baseline_diff) / baseline_diff * 100) if baseline_diff > 0 else 0
            row[config['name']] = f"{diff:.2f}"
        else:
            row[config['name']] = "N/A"
    spend_rows.append(row)

spend_df = pd.DataFrame(spend_rows)
print(spend_df.to_string(index=False))
print()

# Show map structure objectives
print("=" * 80)
print("MAP STRUCTURE OBJECTIVES - DIFFICULTY COMPARISON")
print("=" * 80)
print()

structure_objectives = [
    "Populate the Outer Rim",
    "Intimidate Council",
    "Explore Deep Space",
    "Expand Borders",
    "Control the Borderlands",
    "Patrol Vast Territories",
    "Subdue the Galaxy",
]

structure_rows = []
for obj_name in structure_objectives:
    row = {'Objective': obj_name}
    for config_name, config in configs.items():
        if obj_name in all_difficulties[config_name]:
            diff = all_difficulties[config_name][obj_name]
            row[config['name']] = f"{diff:.2f}"
        else:
            row[config['name']] = "N/A"
    structure_rows.append(row)

structure_df = pd.DataFrame(structure_rows)
print(structure_df.to_string(index=False))
print()

# Calculate SSE for each scenario
print("=" * 80)
print("CORRECTED SSE CALCULATION")
print("=" * 80)
print()

def calculate_corrected_sse(config_name, all_difficulties):
    """Calculate SSE based on difficulty deviation from 6P baseline."""
    baseline_diffs = all_difficulties['6p_base']
    scenario_diffs = all_difficulties[config_name]

    squared_errors = []
    for obj_name, scenario_diff in scenario_diffs.items():
        if obj_name in baseline_diffs and baseline_diffs[obj_name] > 0:
            baseline_diff = baseline_diffs[obj_name]
            # Normalized squared error
            error = ((scenario_diff - baseline_diff) / baseline_diff) ** 2
            squared_errors.append(error)

    return np.mean(squared_errors) if squared_errors else 0.0

sse_results = []
for config_name, config in configs.items():
    sse = calculate_corrected_sse(config_name, all_difficulties)
    sse_results.append({
        'Scenario': config['name'],
        'Players': config['player_count'],
        'Variant': config['variant'].upper(),
        'Resources/round': round(config['resources_per_round'], 2),
        'Influence/round': round(config['influence_per_round'], 2),
        'TG/round': round(config['tg_per_round'], 2),
        'Total/round': round(config['total_economy_per_round'], 2),
        'SSE': round(sse, 4),
    })

sse_df = pd.DataFrame(sse_results)
sse_df = sse_df.sort_values(['Players', 'Variant'])

print(sse_df.to_string(index=False))
print()

# Key findings
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

# Compare 2P scenarios
base_2p_sse = [r for r in sse_results if r['Scenario'] == '2P Base'][0]['SSE']
tw_2p_sse = [r for r in sse_results if r['Scenario'] == '2P Trade Wars'][0]['SSE']

print(f"2P: Base SSE = {base_2p_sse:.4f}, Trade Wars SSE = {tw_2p_sse:.4f}")
if tw_2p_sse < base_2p_sse:
    improvement = ((base_2p_sse - tw_2p_sse) / base_2p_sse) * 100
    print(f"    Trade Wars IMPROVES balance by {improvement:.1f}%")
else:
    degradation = ((tw_2p_sse - base_2p_sse) / base_2p_sse) * 100
    print(f"    Trade Wars WORSENS balance by {degradation:.1f}%")
print()

# Compare 3P scenarios
base_3p_sse = [r for r in sse_results if r['Scenario'] == '3P Base'][0]['SSE']
tw_3p_sse = [r for r in sse_results if r['Scenario'] == '3P Trade Wars'][0]['SSE']

print(f"3P: Base SSE = {base_3p_sse:.4f}, Trade Wars SSE = {tw_3p_sse:.4f}")
if tw_3p_sse < base_3p_sse:
    improvement = ((base_3p_sse - tw_3p_sse) / base_3p_sse) * 100
    print(f"    Trade Wars IMPROVES balance by {improvement:.1f}%")
else:
    degradation = ((tw_3p_sse - base_3p_sse) / base_3p_sse) * 100
    print(f"    Trade Wars WORSENS balance by {degradation:.1f}%")
print()

# Overall ranking
print("Overall Balance Ranking (lower SSE = better):")
sorted_results = sorted(sse_results, key=lambda x: x['SSE'])
for i, r in enumerate(sorted_results, 1):
    print(f"  {i}. {r['Scenario']}: SSE = {r['SSE']:.4f}")
print()

# Save results
output_data = {
    'configs': {k: {kk: vv for kk, vv in v.items()} for k, v in configs.items()},
    'difficulties': all_difficulties,
    'sse_results': sse_results,
}

def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

output_data = convert_to_json_serializable(output_data)

output_file = DATA_DIR / 'corrected_sse_analysis.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")
