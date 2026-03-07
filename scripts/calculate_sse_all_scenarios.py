"""
Calculate SSE for all player count scenarios including Trade Wars variant.

Scenarios:
1. 2P base game (6 blue + 4 red)
2. 2P Trade Wars (9 blue + 8 red + 3 minor factions)
3. 3P base game (9 blue + 6 red)
4. 3P Trade Wars (10 blue + 8 red + 3 minor factions)
5-8. 4P-8P base game (universal scaling: pc × 5 system tiles)

SSE Components (Excel method):
1. Economy Error
2-4. Per-Board Objectives (Stage1, Stage2, Secret)
5-7. Per-Slice Objectives (Stage1, Stage2, Secret)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')

# Load faction data
df_factions = pd.read_csv(DATA_DIR / 'combined_faction_clusters.csv')
AVG_COMMODITIES = df_factions['commodities'].mean()

# Load objectives
with open(DATA_DIR / 'objective_difficulty.json') as f:
    obj_difficulty = json.load(f)

print("=" * 80)
print("SSE CALCULATION: ALL SCENARIOS (BASE + TRADE WARS)")
print("=" * 80)
print()

# 6P baseline
BASELINE_6P = {
    'blue_tiles': 18,
    'red_tiles': 12,
    'system_tiles': 30,
    'homes': 6,
    'total_tiles': 37,
    'edge_systems': 18,
    'mecatol_adjacent': 6,
}

def calculate_tg_per_player(player_count, rounds=5):
    """Calculate TG with updated values."""
    if player_count == 2:
        return 3 * rounds  # 15 TG
    elif player_count == 3:
        return 2.5 * rounds  # 12.5 TG
    else:
        tg_per_round = 3 + (AVG_COMMODITIES * player_count)
        return (tg_per_round / player_count) * rounds

def estimate_resources_per_player(blue_tiles, player_count):
    return (blue_tiles * 1.5) / player_count

def estimate_influence_per_player(blue_tiles, player_count):
    return (blue_tiles * 1.5) / player_count

# Calculate baseline economy
BASELINE_6P['resources_per_player'] = estimate_resources_per_player(18, 6)
BASELINE_6P['influence_per_player'] = estimate_influence_per_player(18, 6)
BASELINE_6P['tg_per_player'] = calculate_tg_per_player(6)
BASELINE_6P['total_economy_per_player'] = (
    BASELINE_6P['resources_per_player'] +
    BASELINE_6P['influence_per_player'] +
    BASELINE_6P['tg_per_player']
)

# Define all configurations
configs = {}

# 2P base game
configs['2p_base'] = {
    'player_count': 2,
    'variant': 'base',
    'blue_tiles': 6,
    'red_tiles': 4,
    'system_tiles': 10,
    'homes': 2,
    'total_tiles': 13,
    'edge_systems': 6,
    'mecatol_adjacent': 2,
}

# 2P Trade Wars
configs['2p_trade_wars'] = {
    'player_count': 2,
    'variant': 'trade_wars',
    'blue_tiles': 9,
    'red_tiles': 8,
    'system_tiles': 17,
    'homes': 2,
    'total_tiles': 23,  # 17 + 3 minor + 2 homes + 1 Mecatol
    'edge_systems': 9,  # Estimate
    'mecatol_adjacent': 2,
    'minor_faction_systems': 3,
    'minor_faction_planets': 7.5,  # 3 × 2.5 avg
}

# 3P base game
configs['3p_base'] = {
    'player_count': 3,
    'variant': 'base',
    'blue_tiles': 9,
    'red_tiles': 6,
    'system_tiles': 15,
    'homes': 3,
    'total_tiles': 19,
    'edge_systems': 9,
    'mecatol_adjacent': 3,
}

# 3P Trade Wars
configs['3p_trade_wars'] = {
    'player_count': 3,
    'variant': 'trade_wars',
    'blue_tiles': 10,
    'red_tiles': 8,
    'system_tiles': 18,
    'homes': 3,
    'total_tiles': 25,  # 18 + 3 minor + 3 homes + 1 Mecatol
    'edge_systems': 10,  # Estimate
    'mecatol_adjacent': 3,
    'minor_faction_systems': 3,
    'minor_faction_planets': 7.5,
}

# 4P-8P base game (universal scaling)
for pc in [4, 5, 6, 7, 8]:
    configs[f'{pc}p_base'] = {
        'player_count': pc,
        'variant': 'base',
        'blue_tiles': pc * 3,
        'red_tiles': pc * 2,
        'system_tiles': pc * 5,
        'homes': pc,
        'total_tiles': pc * 5 + pc + 1,
        'edge_systems': pc * 3,
        'mecatol_adjacent': pc * 1,
    }

# Calculate economy for each config
for config_name, config in configs.items():
    pc = config['player_count']

    # Base resources/influence from blue tiles
    base_resources = estimate_resources_per_player(config['blue_tiles'], pc)
    base_influence = estimate_influence_per_player(config['blue_tiles'], pc)

    # Add minor faction contribution if Trade Wars
    if config.get('variant') == 'trade_wars':
        minor_faction_planets = config.get('minor_faction_planets', 0)
        minor_faction_resources = (minor_faction_planets * 1.5) / pc
        minor_faction_influence = (minor_faction_planets * 1.5) / pc

        resources = base_resources + minor_faction_resources
        influence = base_influence + minor_faction_influence

        # Trade Wars: Space stations enable FULL Trade meta!
        # +3 commodity value from space stations
        boosted_avg_commodities = AVG_COMMODITIES + (3 / pc)  # +1.5 for 2P, +1 for 3P

        # Use full X-1 meta with boosted commodity value
        tg_per_round = (3 + boosted_avg_commodities * pc) / pc

        # Monuments: ~1.5 TG/round average
        monument_tg_per_round = 1.5 / pc

        # Total TG
        total_tg_per_round = tg_per_round + monument_tg_per_round
        tg = total_tg_per_round * 5  # 5 rounds
    else:
        resources = base_resources
        influence = base_influence
        tg = calculate_tg_per_player(pc)

    config['resources_per_player'] = resources
    config['influence_per_player'] = influence
    config['tg_per_player'] = tg
    config['total_economy_per_player'] = resources + influence + tg

# Extract objectives by stage
stage_1_objectives = []
stage_2_objectives = []
secret_objectives = []

for stage, objs in obj_difficulty.items():
    for name, data in objs.items():
        obj_info = {
            'name': name,
            'type': data['requirement_type'],
            'value': data['requirement_value'],
            'difficulties': data['difficulties_by_player_count'],
        }

        if stage == 'stage_1':
            stage_1_objectives.append(obj_info)
        elif stage == 'stage_2':
            stage_2_objectives.append(obj_info)
        elif stage == 'secret':
            secret_objectives.append(obj_info)

def calculate_excel_sse(config):
    """Calculate SSE using Excel method."""
    pc = config['player_count']

    # Component 1: Economy Error
    economy_error = (
        (config['total_economy_per_player'] - BASELINE_6P['total_economy_per_player']) /
        BASELINE_6P['total_economy_per_player']
    ) ** 2

    # Component 2-4: Per-Board Objectives
    def calc_per_board_error(objectives):
        errors = []
        for obj in objectives:
            pc_diff = obj['difficulties'].get(str(pc), 1.0)
            baseline_diff = obj['difficulties'].get('6', 1.0)

            if baseline_diff > 0:
                error = ((pc_diff - baseline_diff) / baseline_diff) ** 2
                errors.append(error)

        return np.mean(errors) if errors else 0.0

    stage1_perboard = calc_per_board_error(stage_1_objectives)
    stage2_perboard = calc_per_board_error(stage_2_objectives)
    secret_perboard = calc_per_board_error(secret_objectives)

    # Component 5-7: Per-Slice Objectives
    stage1_perslice = stage1_perboard
    stage2_perslice = stage2_perboard
    secret_perslice = secret_perboard

    # Total SSE
    total_sse = (
        economy_error +
        stage1_perboard + stage2_perboard + secret_perboard +
        stage1_perslice + stage2_perslice + secret_perslice
    )

    return {
        'total_sse': total_sse,
        'economy_error': economy_error,
        'stage1_perboard': stage1_perboard,
        'stage2_perboard': stage2_perboard,
        'secret_perboard': secret_perboard,
    }

# Calculate SSE for all configs
print("Calculating SSE for all scenarios...")
print()

sse_results = []

for config_name, config in configs.items():
    sse = calculate_excel_sse(config)

    result = {
        'Scenario': config_name.replace('_', ' ').upper(),
        'Players': config['player_count'],
        'Variant': config.get('variant', 'base').upper(),
        'Total Economy': round(config['total_economy_per_player'], 2),
        'Economy Error': round(sse['economy_error'], 4),
        'Total SSE': round(sse['total_sse'], 4),
        'System Tiles': config['system_tiles'],
        'Blue': config['blue_tiles'],
        'Red': config['red_tiles'],
    }

    sse_results.append(result)

# Create DataFrame
sse_df = pd.DataFrame(sse_results)

# Sort by player count, then variant
sse_df = sse_df.sort_values(['Players', 'Variant'])

print("=" * 80)
print("SSE COMPARISON: ALL SCENARIOS")
print("=" * 80)
print()
print(sse_df[['Scenario', 'Players', 'Variant', 'Total Economy', 'Total SSE']].to_string(index=False))
print()

# Detailed breakdown
print("=" * 80)
print("DETAILED SSE BREAKDOWN")
print("=" * 80)
print()
print(sse_df.to_string(index=False))
print()

# Analysis
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

# Find best config for 2P and 3P
base_2p_sse = sse_df[sse_df['Scenario'] == '2P BASE']['Total SSE'].values[0]
tw_2p_sse = sse_df[sse_df['Scenario'] == '2P TRADE WARS']['Total SSE'].values[0]

base_3p_sse = sse_df[sse_df['Scenario'] == '3P BASE']['Total SSE'].values[0]
tw_3p_sse = sse_df[sse_df['Scenario'] == '3P TRADE WARS']['Total SSE'].values[0]

print(f"2P: Base SSE = {base_2p_sse:.4f}, Trade Wars SSE = {tw_2p_sse:.4f}")
if tw_2p_sse < base_2p_sse:
    improvement = ((base_2p_sse - tw_2p_sse) / base_2p_sse) * 100
    print(f"    Trade Wars IMPROVES balance by {improvement:.1f}%")
else:
    degradation = ((tw_2p_sse - base_2p_sse) / base_2p_sse) * 100
    print(f"    Trade Wars WORSENS balance by {degradation:.1f}%")
print()

print(f"3P: Base SSE = {base_3p_sse:.4f}, Trade Wars SSE = {tw_3p_sse:.4f}")
if tw_3p_sse < base_3p_sse:
    improvement = ((base_3p_sse - tw_3p_sse) / base_3p_sse) * 100
    print(f"    Trade Wars IMPROVES balance by {improvement:.1f}%")
else:
    degradation = ((tw_3p_sse - base_3p_sse) / base_3p_sse) * 100
    print(f"    Trade Wars WORSENS balance by {degradation:.1f}%")
print()

# Overall variance
print("Overall SSE variance across all player counts:")
print(f"  Mean SSE: {sse_df['Total SSE'].mean():.4f}")
print(f"  Std Dev: {sse_df['Total SSE'].std():.4f}")
print(f"  CV: {(sse_df['Total SSE'].std() / sse_df['Total SSE'].mean() * 100):.1f}%")
print()

# Compare base game only
base_only = sse_df[sse_df['Variant'] == 'BASE']
print("Base game only (2P-8P) SSE variance:")
print(f"  Mean SSE: {base_only['Total SSE'].mean():.4f}")
print(f"  Std Dev: {base_only['Total SSE'].std():.4f}")
print(f"  CV: {(base_only['Total SSE'].std() / base_only['Total SSE'].mean() * 100):.1f}%")
print()

# Save results
output_data = {
    'configs': configs,
    'sse_results': sse_results,
}

# Convert to JSON-serializable
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

output_file = DATA_DIR / 'all_scenarios_sse.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")
