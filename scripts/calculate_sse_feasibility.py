"""
SSE Calculation with Feasibility Focus

Key insight from user: Being unable to score an objective is MUCH WORSE
than having it be too easy.

Metrics:
1. Mean difficulty per stage (Stage I, Stage II, Secret)
2. Standard deviation of difficulties per stage
3. Feasibility rate: % of objectives that can be scored in reasonable time
4. Asymmetric penalty: Impossible objectives penalized more than easy ones
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
print("SSE CALCULATION WITH FEASIBILITY FOCUS")
print("=" * 80)
print()

ROUNDS = 5

# Define configurations (same as before)
def create_configs():
    configs = {}

    # 2P BASE GAME
    pc = 2
    blue = 6
    resources_per_round = (blue * 1.5) / pc
    influence_per_round = (blue * 1.5) / pc
    tg_per_round = 3  # No trade meta
    configs['2p_base'] = {
        'name': '2P Base',
        'player_count': 2,
        'variant': 'base',
        'blue_tiles': blue,
        'red_tiles': 4,
        'resources_per_round': resources_per_round,
        'influence_per_round': influence_per_round,
        'tg_per_round': tg_per_round,
        'total_economy_per_round': resources_per_round + influence_per_round + tg_per_round,
        'edge_systems': 6,
        'mecatol_adjacent': 2,
        'empty_systems': 4,
        'total_planets': blue * 1.5,
    }

    # 2P TRADE WARS
    pc = 2
    blue = 9
    minor_faction_planets = 7.5
    total_blue_equivalent = blue + minor_faction_planets
    resources_per_round = (total_blue_equivalent * 1.5) / pc
    influence_per_round = (total_blue_equivalent * 1.5) / pc
    boosted_commodities = AVG_COMMODITIES + (3 / pc)
    tg_per_round = (3 + boosted_commodities * pc) / pc + (1.5 / pc)
    configs['2p_trade_wars'] = {
        'name': '2P Trade Wars',
        'player_count': 2,
        'variant': 'trade_wars',
        'blue_tiles': blue,
        'red_tiles': 8,
        'resources_per_round': resources_per_round,
        'influence_per_round': influence_per_round,
        'tg_per_round': tg_per_round,
        'total_economy_per_round': resources_per_round + influence_per_round + tg_per_round,
        'edge_systems': 9,
        'mecatol_adjacent': 2,
        'empty_systems': 8,
        'total_planets': total_blue_equivalent * 1.5,
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
        'resources_per_round': resources_per_round,
        'influence_per_round': influence_per_round,
        'tg_per_round': tg_per_round,
        'total_economy_per_round': resources_per_round + influence_per_round + tg_per_round,
        'edge_systems': 10,
        'mecatol_adjacent': 3,
        'empty_systems': 8,
        'total_planets': total_blue_equivalent * 1.5,
    }

    # 4P-8P BASE GAME
    for pc in [4, 5, 6, 7, 8]:
        blue = pc * 3
        red = pc * 2
        resources_per_round = (blue * 1.5) / pc
        influence_per_round = (blue * 1.5) / pc
        tg_per_round = (3 + AVG_COMMODITIES * pc) / pc
        configs[f'{pc}p_base'] = {
            'name': f'{pc}P Base',
            'player_count': pc,
            'variant': 'base',
            'blue_tiles': blue,
            'red_tiles': red,
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
BASELINE = configs['6p_base']

def calculate_objective_difficulty(obj, config):
    """Calculate difficulty for an objective."""
    req_type = obj.get('requirement_type')
    req_value = obj.get('requirement_value')
    pc = config['player_count']

    if req_type == 'spend_resources':
        available = config['resources_per_round'] + config['tg_per_round']
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'spend_influence':
        available = config['influence_per_round'] + config['tg_per_round']
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

    elif req_type == 'edge_systems':
        available = config['edge_systems'] / pc
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'adjacent_to_mecatol':
        available = config['mecatol_adjacent'] / pc
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'empty_systems':
        available = config['empty_systems'] / pc
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'non_home_planets':
        available = config['total_planets'] / pc
        return req_value / available if available > 0 else float('inf')

    elif req_type == 'control_resources':
        total_resources = config['total_planets'] * 1.5 / pc
        return req_value / total_resources if total_resources > 0 else float('inf')

    elif req_type == 'control_influence':
        total_influence = config['total_planets'] * 1.5 / pc
        return req_value / total_influence if total_influence > 0 else float('inf')

    elif req_type == 'total_systems':
        available = config.get('system_tiles', 30) / pc
        return req_value / available if available > 0 else float('inf')

    else:
        return None

# Calculate difficulties by stage
def analyze_by_stage(configs, objectives_data):
    """Analyze difficulties by stage for each configuration."""
    results = {}

    for config_name, config in configs.items():
        results[config_name] = {
            'stage_1': [],
            'stage_2': [],
            'secret': [],
        }

        for stage in ['stage_1', 'stage_2', 'secret']:
            for obj in objectives_data[stage]:
                if obj.get('map_dependent', False):
                    diff = calculate_objective_difficulty(obj, config)
                    if diff is not None and diff != float('inf'):
                        results[config_name][stage].append({
                            'name': obj['name'],
                            'difficulty': diff,
                        })

    return results

stage_results = analyze_by_stage(configs, objectives_data)

# Calculate statistics per stage
print("=" * 80)
print("STAGE I OBJECTIVES - STATISTICS BY SCENARIO")
print("=" * 80)
print()

stage_stats = {}
for config_name, config in configs.items():
    stage_stats[config_name] = {}

    for stage in ['stage_1', 'stage_2', 'secret']:
        diffs = [obj['difficulty'] for obj in stage_results[config_name][stage]]
        if diffs:
            stage_stats[config_name][stage] = {
                'count': len(diffs),
                'mean': np.mean(diffs),
                'std': np.std(diffs),
                'min': np.min(diffs),
                'max': np.max(diffs),
                'feasible': sum(1 for d in diffs if d <= 2.0),  # Can do in ~2 rounds
                'feasible_pct': sum(1 for d in diffs if d <= 2.0) / len(diffs) * 100,
            }

# Print Stage I stats
stage1_rows = []
for config_name, config in configs.items():
    stats = stage_stats[config_name].get('stage_1', {})
    if stats:
        stage1_rows.append({
            'Scenario': config['name'],
            'Count': stats['count'],
            'Mean Diff': round(stats['mean'], 3),
            'Std Dev': round(stats['std'], 3),
            'Min': round(stats['min'], 3),
            'Max': round(stats['max'], 3),
            'Feasible': f"{stats['feasible']}/{stats['count']}",
            'Feasible %': f"{stats['feasible_pct']:.0f}%",
        })

print(pd.DataFrame(stage1_rows).to_string(index=False))
print()

# Print Stage II stats
print("=" * 80)
print("STAGE II OBJECTIVES - STATISTICS BY SCENARIO")
print("=" * 80)
print()

stage2_rows = []
for config_name, config in configs.items():
    stats = stage_stats[config_name].get('stage_2', {})
    if stats:
        stage2_rows.append({
            'Scenario': config['name'],
            'Count': stats['count'],
            'Mean Diff': round(stats['mean'], 3),
            'Std Dev': round(stats['std'], 3),
            'Min': round(stats['min'], 3),
            'Max': round(stats['max'], 3),
            'Feasible': f"{stats['feasible']}/{stats['count']}",
            'Feasible %': f"{stats['feasible_pct']:.0f}%",
        })

print(pd.DataFrame(stage2_rows).to_string(index=False))
print()

# Print Secret stats
print("=" * 80)
print("SECRET OBJECTIVES - STATISTICS BY SCENARIO")
print("=" * 80)
print()

secret_rows = []
for config_name, config in configs.items():
    stats = stage_stats[config_name].get('secret', {})
    if stats:
        secret_rows.append({
            'Scenario': config['name'],
            'Count': stats['count'],
            'Mean Diff': round(stats['mean'], 3),
            'Std Dev': round(stats['std'], 3),
            'Min': round(stats['min'], 3),
            'Max': round(stats['max'], 3),
            'Feasible': f"{stats['feasible']}/{stats['count']}",
            'Feasible %': f"{stats['feasible_pct']:.0f}%",
        })

print(pd.DataFrame(secret_rows).to_string(index=False))
print()

# Calculate normalized SSE like Excel
print("=" * 80)
print("NORMALIZED SSE CALCULATION (Excel Method)")
print("=" * 80)
print()

def calculate_normalized_sse(config_name, stage_stats, baseline_stats):
    """
    Calculate SSE with proper normalization.

    Components:
    1. Economy error (normalized)
    2. Stage I: Mean + StdDev errors
    3. Stage II: Mean + StdDev errors
    4. Secret: Mean + StdDev errors
    """
    config = configs[config_name]
    baseline = configs['6p_base']

    # Economy error (normalized)
    economy_error = ((config['total_economy_per_round'] - baseline['total_economy_per_round']) /
                     baseline['total_economy_per_round']) ** 2

    # Per-stage errors
    stage_errors = {}
    for stage in ['stage_1', 'stage_2', 'secret']:
        if stage in stage_stats[config_name] and stage in baseline_stats:
            config_stats = stage_stats[config_name][stage]
            base_stats = baseline_stats[stage]

            # Mean error (normalized)
            if base_stats['mean'] > 0:
                mean_error = ((config_stats['mean'] - base_stats['mean']) / base_stats['mean']) ** 2
            else:
                mean_error = 0

            # StdDev error (normalized)
            if base_stats['std'] > 0:
                std_error = ((config_stats['std'] - base_stats['std']) / base_stats['std']) ** 2
            else:
                std_error = config_stats['std'] ** 2 if config_stats['std'] > 0 else 0

            stage_errors[stage] = {
                'mean_error': mean_error,
                'std_error': std_error,
                'combined': mean_error + std_error,
            }

    # Total SSE (7 components: economy + 2 per stage × 3 stages)
    total_sse = economy_error
    for stage in ['stage_1', 'stage_2', 'secret']:
        if stage in stage_errors:
            total_sse += stage_errors[stage]['combined']

    return {
        'economy_error': economy_error,
        'stage_errors': stage_errors,
        'total_sse': total_sse,
    }

# Get baseline stats
baseline_stats = stage_stats['6p_base']

# Calculate SSE for all scenarios
sse_results = []
for config_name, config in configs.items():
    sse_data = calculate_normalized_sse(config_name, stage_stats, baseline_stats)

    sse_results.append({
        'Scenario': config['name'],
        'Economy/round': round(config['total_economy_per_round'], 2),
        'Econ Error': round(sse_data['economy_error'], 4),
        'Stage1 Error': round(sse_data['stage_errors'].get('stage_1', {}).get('combined', 0), 4),
        'Stage2 Error': round(sse_data['stage_errors'].get('stage_2', {}).get('combined', 0), 4),
        'Secret Error': round(sse_data['stage_errors'].get('secret', {}).get('combined', 0), 4),
        'Total SSE': round(sse_data['total_sse'], 4),
    })

sse_df = pd.DataFrame(sse_results)
print(sse_df.to_string(index=False))
print()

# Final ranking
print("=" * 80)
print("FINAL RANKING (Lower SSE = Better Balance)")
print("=" * 80)
print()

sorted_results = sorted(sse_results, key=lambda x: x['Total SSE'])
for i, r in enumerate(sorted_results, 1):
    print(f"  {i}. {r['Scenario']}: SSE = {r['Total SSE']:.4f}")
print()

# Key findings
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

base_2p = [r for r in sse_results if r['Scenario'] == '2P Base'][0]
tw_2p = [r for r in sse_results if r['Scenario'] == '2P Trade Wars'][0]
base_3p = [r for r in sse_results if r['Scenario'] == '3P Base'][0]
tw_3p = [r for r in sse_results if r['Scenario'] == '3P Trade Wars'][0]

print(f"2P Comparison:")
print(f"  Base: SSE = {base_2p['Total SSE']:.4f}")
print(f"  Trade Wars: SSE = {tw_2p['Total SSE']:.4f}")
if tw_2p['Total SSE'] < base_2p['Total SSE']:
    print(f"  -> Trade Wars IMPROVES balance by {((base_2p['Total SSE'] - tw_2p['Total SSE']) / base_2p['Total SSE'] * 100):.1f}%")
else:
    print(f"  -> Trade Wars WORSENS balance by {((tw_2p['Total SSE'] - base_2p['Total SSE']) / base_2p['Total SSE'] * 100):.1f}%")
print()

print(f"3P Comparison:")
print(f"  Base: SSE = {base_3p['Total SSE']:.4f}")
print(f"  Trade Wars: SSE = {tw_3p['Total SSE']:.4f}")
if tw_3p['Total SSE'] < base_3p['Total SSE']:
    print(f"  -> Trade Wars IMPROVES balance by {((base_3p['Total SSE'] - tw_3p['Total SSE']) / base_3p['Total SSE'] * 100):.1f}%")
else:
    print(f"  -> Trade Wars WORSENS balance by {((tw_3p['Total SSE'] - base_3p['Total SSE']) / base_3p['Total SSE'] * 100):.1f}%")
print()

# Save results
output_file = DATA_DIR / 'feasibility_sse_analysis.json'
with open(output_file, 'w') as f:
    json.dump({
        'stage_stats': {k: {kk: {kkk: float(vvv) if isinstance(vvv, (np.floating, np.integer)) else vvv
                                  for kkk, vvv in vv.items()}
                            for kk, vv in v.items()}
                        for k, v in stage_stats.items()},
        'sse_results': sse_results,
    }, f, indent=2)

print(f"Results saved to: {output_file}")
