"""
Analyze TI4 objectives with proper economic modeling.

Key insights from game structure:
- Game is to 10 points
- Players typically score:
  * 3 secret objectives (3 VP)
  * 1-2 stage II objectives (2-4 VP)
  * 3-4 stage I objectives (3-4 VP)
  * Often Mecatol Rex for 1-2 rounds (1-2 VP)
- Players cherry-pick easiest objectives
- Objectives are random each game

Economic Framework (from unified_economic_analysis.md):
- 1 CC = 3i = 2r = 1.5-1.8 TG
- TG can substitute for BOTH resources OR influence (fungible)
- For spend objectives: total_economy = resources + influence + TG
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path('data')

# Load data
with open(DATA_DIR / 'objectives.json') as f:
    objectives_data = json.load(f)

# Load faction data for commodity values
df_factions = pd.read_csv(DATA_DIR / 'combined_faction_clusters.csv')
AVG_COMMODITIES = df_factions['commodities'].mean()  # ~3.17

print("=" * 80)
print("OBJECTIVE ANALYSIS WITH ECONOMIC FRAMEWORK")
print("=" * 80)
print()
print("Economic Conversion Rates:")
print("  1 CC = 3 influence = 2 resources = 1.5-1.8 trade goods")
print("  Trade goods are FUNGIBLE (can be spent as resources OR influence)")
print()
print(f"Average commodity value: {AVG_COMMODITIES:.2f}")
print()
print("Trade Strategy Card Model:")
print("  2P: No Trade (minimal value with only 1 other player)")
print("  3P: Reluctant Trade (~50% of optimal due to kingmaking concerns)")
print("  4P+: Full X-1 meta (stable coalition building)")
print()

# 6-Player baseline configuration
BASELINE_6P = {
    'blue_tiles': 18,
    'red_tiles': 12,
    'system_tiles': 30,
    'homes': 6,
    'total_tiles': 37,
}

def calculate_tg_per_player(player_count, rounds=5):
    """
    Calculate expected TG per player using X-1 meta.

    Note: Low player counts have limited trading:
    - 2P: No Trade (minimal value with only 1 other player)
    - 3P: Reluctant Trade (kingmaking concerns, zero-sum dynamics)
    - 4P+: Full X-1 meta (stable coalition building)
    """
    if player_count == 2:
        # No Trade meta at 2P - only organic TG from other sources
        # Estimate: ~3 TG per round from action cards, faction abilities, etc.
        return 3 * rounds  # 15 TG over 5 rounds
    elif player_count == 3:
        # Limited Trade at 3P - kingmaking concerns prevent consistent trading
        # Estimate: Partial trade (50% of optimal) + organic TG
        # Optimal: (3 + 3.17*3) / 3 = 4.17 TG/round
        # Actual: ~50% trading + organic = 2.5 TG/round
        return 2.5 * rounds  # 12.5 TG over 5 rounds
    else:
        # Standard X-1 meta (4P+)
        tg_per_round = 3 + (AVG_COMMODITIES * player_count)
        return (tg_per_round / player_count) * rounds

def estimate_resources_per_player(blue_tiles, player_count):
    """Rough estimate: each blue tile has ~1.5 resources on average."""
    return (blue_tiles * 1.5) / player_count

def estimate_influence_per_player(blue_tiles, player_count):
    """Rough estimate: each blue tile has ~1.5 influence on average."""
    return (blue_tiles * 1.5) / player_count

def get_config_for_player_count(pc):
    """Get optimal configuration for player count."""
    # From our optimization results - universal scaling
    if pc == 6:
        return BASELINE_6P
    else:
        return {
            'blue_tiles': pc * 3,
            'red_tiles': pc * 2,
            'system_tiles': pc * 5,
            'homes': pc,
            'total_tiles': pc * 5 + pc + 1,
        }

def calculate_total_economy(player_count):
    """Calculate total economy per player (resources + influence + TG)."""
    config = get_config_for_player_count(player_count)

    resources = estimate_resources_per_player(config['blue_tiles'], player_count)
    influence = estimate_influence_per_player(config['blue_tiles'], player_count)
    tg = calculate_tg_per_player(player_count)

    return {
        'resources': resources,
        'influence': influence,
        'trade_goods': tg,
        'total': resources + influence + tg,
        'config': config,
    }

# Calculate economy for all player counts
print("=" * 80)
print("TOTAL ECONOMY PER PLAYER (5 rounds)")
print("=" * 80)
print()

economy_by_pc = {}
for pc in [2, 3, 4, 5, 6, 7, 8]:
    economy_by_pc[pc] = calculate_total_economy(pc)

# Display table
economy_rows = []
for pc in sorted(economy_by_pc.keys()):
    econ = economy_by_pc[pc]
    economy_rows.append({
        'Players': pc,
        'Resources': round(econ['resources'], 2),
        'Influence': round(econ['influence'], 2),
        'Trade Goods': round(econ['trade_goods'], 2),
        'Total Economy': round(econ['total'], 2),
        'vs 6P': f"{((econ['total'] - economy_by_pc[6]['total']) / economy_by_pc[6]['total'] * 100):+.1f}%",
    })

economy_df = pd.DataFrame(economy_rows)
print(economy_df.to_string(index=False))
print()

# Analyze map-dependent objectives
print("=" * 80)
print("MAP-DEPENDENT OBJECTIVES ANALYSIS")
print("=" * 80)
print()

def analyze_objective_difficulty(objective, player_count):
    """
    Calculate difficulty for an objective at a given player count.

    Returns difficulty score (0-1, where 0 = impossible, 1 = trivial).
    """
    req_type = objective.get('requirement_type')
    req_value = objective.get('requirement_value')

    if not objective.get('map_dependent', False):
        return None  # Can't calculate from map alone

    econ = economy_by_pc[player_count]
    config = econ['config']

    # Spend objectives with TG substitution
    if req_type == 'spend_resources':
        # Resources + TG can both count
        available = econ['resources'] + econ['trade_goods']
        difficulty = req_value / available if available > 0 else float('inf')
        return {'type': 'spend', 'required': req_value, 'available': available, 'difficulty': difficulty}

    elif req_type == 'spend_influence':
        # Influence + TG can both count
        available = econ['influence'] + econ['trade_goods']
        difficulty = req_value / available if available > 0 else float('inf')
        return {'type': 'spend', 'required': req_value, 'available': available, 'difficulty': difficulty}

    elif req_type == 'spend_trade_goods':
        # Only TG count
        available = econ['trade_goods']
        difficulty = req_value / available if available > 0 else float('inf')
        return {'type': 'spend', 'required': req_value, 'available': available, 'difficulty': difficulty}

    elif req_type == 'spend_combined':
        # Need resources + influence + TG
        if isinstance(req_value, dict):
            req_r = req_value.get('resources', 0)
            req_i = req_value.get('influence', 0)
            req_tg = req_value.get('trade_goods', 0)

            # TG can substitute for either r or i
            total_available = econ['resources'] + econ['influence'] + econ['trade_goods']
            total_required = req_r + req_i + req_tg

            difficulty = total_required / total_available if total_available > 0 else float('inf')
            return {'type': 'spend', 'required': total_required, 'available': total_available, 'difficulty': difficulty}

    # Control objectives
    elif req_type == 'control_resources':
        # Need to control planets with total resources >= req_value
        # Estimate: each player controls ~blue_tiles/player_count planets
        # Each planet averages ~1.5 resources
        planets_per_player = config['blue_tiles'] / player_count
        total_r_available = planets_per_player * 1.5
        difficulty = req_value / total_r_available if total_r_available > 0 else float('inf')
        return {'type': 'control', 'required': req_value, 'available': total_r_available, 'difficulty': difficulty}

    elif req_type == 'control_influence':
        # Similar to resources
        planets_per_player = config['blue_tiles'] / player_count
        total_i_available = planets_per_player * 1.5
        difficulty = req_value / total_i_available if total_i_available > 0 else float('inf')
        return {'type': 'control', 'required': req_value, 'available': total_i_available, 'difficulty': difficulty}

    # Planet count objectives
    elif req_type == 'non_home_planets':
        planets_per_player = config['blue_tiles'] / player_count
        difficulty = req_value / planets_per_player if planets_per_player > 0 else float('inf')
        return {'type': 'control', 'required': req_value, 'available': planets_per_player, 'difficulty': difficulty}

    # Map structure objectives
    elif req_type == 'edge_systems':
        # Estimate: ~3 edge systems per player
        edge_per_player = player_count * 3 / player_count  # 3
        difficulty = req_value / edge_per_player if edge_per_player > 0 else float('inf')
        return {'type': 'map_structure', 'required': req_value, 'available': edge_per_player, 'difficulty': difficulty}

    elif req_type == 'adjacent_to_mecatol':
        # Estimate: ~1 Mecatol-adjacent per player
        mecatol_per_player = player_count * 1 / player_count  # 1
        difficulty = req_value / mecatol_per_player if mecatol_per_player > 0 else float('inf')
        return {'type': 'map_structure', 'required': req_value, 'available': mecatol_per_player, 'difficulty': difficulty}

    # Planet trait objectives - harder to estimate without actual tile distribution
    # For now, return None (would need actual map data)
    elif req_type in ['planet_trait', 'tech_specialty_planets', 'cultural_planets',
                       'hazardous_planets', 'industrial_planets']:
        return None

    # Other map-dependent types
    else:
        return None

# Analyze all objectives
stage_1_objectives = []
stage_2_objectives = []
secret_objectives = []

for obj in objectives_data['stage_1']:
    if obj.get('map_dependent', False):
        stage_1_objectives.append(obj)

for obj in objectives_data['stage_2']:
    if obj.get('map_dependent', False):
        stage_2_objectives.append(obj)

for obj in objectives_data['secret']:
    if obj.get('map_dependent', False):
        secret_objectives.append(obj)

print(f"Map-dependent objectives:")
print(f"  Stage I: {len(stage_1_objectives)}")
print(f"  Stage II: {len(stage_2_objectives)}")
print(f"  Secret: {len(secret_objectives)}")
print()

# Analyze Stage I objectives
print("=" * 80)
print("STAGE I OBJECTIVE DIFFICULTIES (by player count)")
print("=" * 80)
print()

stage_1_analysis = {}
for obj in stage_1_objectives:
    obj_name = obj['name']
    stage_1_analysis[obj_name] = {}

    for pc in [2, 3, 4, 5, 6, 7, 8]:
        result = analyze_objective_difficulty(obj, pc)
        if result:
            stage_1_analysis[obj_name][pc] = result['difficulty']

# Create DataFrame for Stage I
if stage_1_analysis:
    stage_1_df = pd.DataFrame(stage_1_analysis).T
    stage_1_df = stage_1_df.round(3)

    # Calculate statistics
    stage_1_df['Mean'] = stage_1_df.mean(axis=1)
    stage_1_df['Std'] = stage_1_df.std(axis=1)
    stage_1_df['CV'] = (stage_1_df['Std'] / stage_1_df['Mean'] * 100).round(1)

    print(stage_1_df.to_string())
    print()

# Analyze Stage II objectives
print("=" * 80)
print("STAGE II OBJECTIVE DIFFICULTIES (by player count)")
print("=" * 80)
print()

stage_2_analysis = {}
for obj in stage_2_objectives:
    obj_name = obj['name']
    stage_2_analysis[obj_name] = {}

    for pc in [2, 3, 4, 5, 6, 7, 8]:
        result = analyze_objective_difficulty(obj, pc)
        if result:
            stage_2_analysis[obj_name][pc] = result['difficulty']

# Create DataFrame for Stage II
if stage_2_analysis:
    stage_2_df = pd.DataFrame(stage_2_analysis).T
    stage_2_df = stage_2_df.round(3)

    # Calculate statistics
    stage_2_df['Mean'] = stage_2_df.mean(axis=1)
    stage_2_df['Std'] = stage_2_df.std(axis=1)
    stage_2_df['CV'] = (stage_2_df['Std'] / stage_2_df['Mean'] * 100).round(1)

    print(stage_2_df.to_string())
    print()

# Key insight: Players cherry-pick easiest objectives
print("=" * 80)
print("CHERRY-PICKING ANALYSIS")
print("=" * 80)
print()
print("Players typically score 3-4 easiest Stage I objectives.")
print("Let's analyze the difficulty of the Nth easiest objective at each player count:")
print()

# For each player count, find the 3rd and 4th easiest objectives
if stage_1_analysis:
    cherry_pick_data = []

    for pc in [2, 3, 4, 5, 6, 7, 8]:
        difficulties = []
        for obj_name, pc_difficulties in stage_1_analysis.items():
            if pc in pc_difficulties:
                difficulties.append(pc_difficulties[pc])

        difficulties.sort()

        if len(difficulties) >= 4:
            cherry_pick_data.append({
                'Players': pc,
                'Easiest': round(difficulties[0], 3),
                '2nd Easiest': round(difficulties[1], 3),
                '3rd Easiest': round(difficulties[2], 3),
                '4th Easiest': round(difficulties[3], 3),
                'Available': len(difficulties),
            })

    cherry_df = pd.DataFrame(cherry_pick_data)
    print(cherry_df.to_string(index=False))
    print()

    # Calculate variance in cherry-picking difficulty
    print("Variance Analysis (lower is better):")
    print(f"  3rd easiest objective CV: {cherry_df['3rd Easiest'].std() / cherry_df['3rd Easiest'].mean() * 100:.1f}%")
    print(f"  4th easiest objective CV: {cherry_df['4th Easiest'].std() / cherry_df['4th Easiest'].mean() * 100:.1f}%")
    print()

# Save results
output_data = {
    'economy_by_player_count': {str(pc): econ for pc, econ in economy_by_pc.items()},
    'stage_1_difficulties': stage_1_analysis,
    'stage_2_difficulties': stage_2_analysis,
}

# Convert numpy types to native Python types for JSON serialization
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

output_file = DATA_DIR / 'objective_difficulties_with_economy.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to: {output_file}")
