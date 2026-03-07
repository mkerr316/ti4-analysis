"""
SSE Calculation - EXACT Excel Method

User's formula: =AVERAGE((((D49:D55)-$E$49:$E$55)/$E$49:$E$55)^2)

For EACH objective:
1. Calculate difficulty in scenario
2. Calculate difficulty in 6P baseline
3. Normalized error = ((scenario - baseline) / baseline)^2
4. Average all errors

This is Normalized Mean Squared Error (NMSE) - a gold standard approach.
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
print("SSE CALCULATION - EXACT EXCEL METHOD")
print("Formula: AVERAGE((((scenario - baseline) / baseline)^2))")
print("=" * 80)
print()

# Define configurations
def create_configs():
    configs = {}

    # 2P BASE
    pc, blue = 2, 6
    configs['2p_base'] = {
        'name': '2P Base', 'player_count': pc, 'variant': 'base',
        'resources_per_round': (blue * 1.5) / pc,
        'influence_per_round': (blue * 1.5) / pc,
        'tg_per_round': 3,
        'edge_systems': 6, 'mecatol_adjacent': 2, 'empty_systems': 4,
        'total_planets': blue * 1.5,
    }
    configs['2p_base']['total_economy_per_round'] = (
        configs['2p_base']['resources_per_round'] +
        configs['2p_base']['influence_per_round'] +
        configs['2p_base']['tg_per_round']
    )

    # 2P TRADE WARS
    pc, blue = 2, 9
    minor_planets = 7.5
    boosted_comm = AVG_COMMODITIES + (3 / pc)
    tg = (3 + boosted_comm * pc) / pc + (1.5 / pc)
    configs['2p_trade_wars'] = {
        'name': '2P Trade Wars', 'player_count': pc, 'variant': 'trade_wars',
        'resources_per_round': ((blue + minor_planets) * 1.5) / pc,
        'influence_per_round': ((blue + minor_planets) * 1.5) / pc,
        'tg_per_round': tg,
        'edge_systems': 9, 'mecatol_adjacent': 2, 'empty_systems': 8,
        'total_planets': (blue + minor_planets) * 1.5,
    }
    configs['2p_trade_wars']['total_economy_per_round'] = (
        configs['2p_trade_wars']['resources_per_round'] +
        configs['2p_trade_wars']['influence_per_round'] +
        configs['2p_trade_wars']['tg_per_round']
    )

    # 3P BASE
    pc, blue = 3, 9
    configs['3p_base'] = {
        'name': '3P Base', 'player_count': pc, 'variant': 'base',
        'resources_per_round': (blue * 1.5) / pc,
        'influence_per_round': (blue * 1.5) / pc,
        'tg_per_round': 2.5,
        'edge_systems': 9, 'mecatol_adjacent': 3, 'empty_systems': 6,
        'total_planets': blue * 1.5,
    }
    configs['3p_base']['total_economy_per_round'] = (
        configs['3p_base']['resources_per_round'] +
        configs['3p_base']['influence_per_round'] +
        configs['3p_base']['tg_per_round']
    )

    # 3P TRADE WARS
    pc, blue = 3, 10
    minor_planets = 7.5
    boosted_comm = AVG_COMMODITIES + (3 / pc)
    tg = (3 + boosted_comm * pc) / pc + (1.5 / pc)
    configs['3p_trade_wars'] = {
        'name': '3P Trade Wars', 'player_count': pc, 'variant': 'trade_wars',
        'resources_per_round': ((blue + minor_planets) * 1.5) / pc,
        'influence_per_round': ((blue + minor_planets) * 1.5) / pc,
        'tg_per_round': tg,
        'edge_systems': 10, 'mecatol_adjacent': 3, 'empty_systems': 8,
        'total_planets': (blue + minor_planets) * 1.5,
    }
    configs['3p_trade_wars']['total_economy_per_round'] = (
        configs['3p_trade_wars']['resources_per_round'] +
        configs['3p_trade_wars']['influence_per_round'] +
        configs['3p_trade_wars']['tg_per_round']
    )

    # 4P-8P BASE
    for pc in [4, 5, 6, 7, 8]:
        blue = pc * 3
        configs[f'{pc}p_base'] = {
            'name': f'{pc}P Base', 'player_count': pc, 'variant': 'base',
            'resources_per_round': (blue * 1.5) / pc,
            'influence_per_round': (blue * 1.5) / pc,
            'tg_per_round': (3 + AVG_COMMODITIES * pc) / pc,
            'edge_systems': pc * 3, 'mecatol_adjacent': pc, 'empty_systems': pc * 2,
            'total_planets': blue * 1.5,
        }
        configs[f'{pc}p_base']['total_economy_per_round'] = (
            configs[f'{pc}p_base']['resources_per_round'] +
            configs[f'{pc}p_base']['influence_per_round'] +
            configs[f'{pc}p_base']['tg_per_round']
        )

    return configs

configs = create_configs()
BASELINE = configs['6p_base']

def calc_difficulty(obj, config):
    """Calculate difficulty for one objective."""
    req_type = obj.get('requirement_type')
    req_value = obj.get('requirement_value')
    pc = config['player_count']

    if req_type == 'spend_resources':
        avail = config['resources_per_round'] + config['tg_per_round']
        return req_value / avail if avail > 0 else None
    elif req_type == 'spend_influence':
        avail = config['influence_per_round'] + config['tg_per_round']
        return req_value / avail if avail > 0 else None
    elif req_type == 'spend_trade_goods':
        avail = config['tg_per_round']
        return req_value / avail if avail > 0 else None
    elif req_type == 'spend_combined':
        if isinstance(req_value, dict):
            total_req = sum(req_value.values())
            return total_req / config['total_economy_per_round']
        return None
    elif req_type == 'edge_systems':
        return req_value / (config['edge_systems'] / pc)
    elif req_type == 'adjacent_to_mecatol':
        return req_value / (config['mecatol_adjacent'] / pc)
    elif req_type == 'empty_systems':
        return req_value / (config['empty_systems'] / pc)
    elif req_type == 'non_home_planets':
        return req_value / (config['total_planets'] / pc)
    elif req_type == 'control_resources':
        return req_value / (config['total_planets'] * 1.5 / pc)
    elif req_type == 'control_influence':
        return req_value / (config['total_planets'] * 1.5 / pc)
    elif req_type == 'total_systems':
        return req_value / (config.get('edge_systems', 18) / pc)
    else:
        return None

# Calculate individual objective errors
print("=" * 80)
print("INDIVIDUAL OBJECTIVE ERRORS (vs 6P Baseline)")
print("=" * 80)
print()

all_obj_errors = {}  # config_name -> {obj_name: normalized_error}

for config_name, config in configs.items():
    all_obj_errors[config_name] = {}

    for stage in ['stage_1', 'stage_2', 'secret']:
        for obj in objectives_data[stage]:
            if not obj.get('map_dependent', False):
                continue

            scenario_diff = calc_difficulty(obj, config)
            baseline_diff = calc_difficulty(obj, BASELINE)

            if scenario_diff is not None and baseline_diff is not None and baseline_diff > 0:
                normalized_error = ((scenario_diff - baseline_diff) / baseline_diff) ** 2
                all_obj_errors[config_name][obj['name']] = {
                    'scenario_diff': scenario_diff,
                    'baseline_diff': baseline_diff,
                    'normalized_error': normalized_error,
                    'stage': stage,
                }

# Show sample of individual errors for key scenarios
print("Sample: 'Negotiate Trade Routes' (5 TG)")
print("-" * 60)
for config_name in ['2p_base', '2p_trade_wars', '3p_base', '3p_trade_wars', '6p_base']:
    if 'Negotiate Trade Routes' in all_obj_errors[config_name]:
        err = all_obj_errors[config_name]['Negotiate Trade Routes']
        print(f"{configs[config_name]['name']:15s}: diff={err['scenario_diff']:.3f}, "
              f"baseline={err['baseline_diff']:.3f}, error={err['normalized_error']:.4f}")
print()

print("Sample: 'Hold Vast Reserves' (6i+6r+6tg)")
print("-" * 60)
for config_name in ['2p_base', '2p_trade_wars', '3p_base', '3p_trade_wars', '6p_base']:
    if 'Hold Vast Reserves' in all_obj_errors[config_name]:
        err = all_obj_errors[config_name]['Hold Vast Reserves']
        print(f"{configs[config_name]['name']:15s}: diff={err['scenario_diff']:.3f}, "
              f"baseline={err['baseline_diff']:.3f}, error={err['normalized_error']:.4f}")
print()

# Calculate SSE per stage using EXACT Excel method
print("=" * 80)
print("SSE BY STAGE (EXACT Excel: AVERAGE of individual errors)")
print("=" * 80)
print()

def calc_stage_sse(config_name, stage):
    """Calculate average normalized squared error for a stage."""
    errors = [e['normalized_error'] for name, e in all_obj_errors[config_name].items()
              if e['stage'] == stage]
    return np.mean(errors) if errors else 0.0

stage_sse = {}
for config_name in configs.keys():
    stage_sse[config_name] = {
        'stage_1': calc_stage_sse(config_name, 'stage_1'),
        'stage_2': calc_stage_sse(config_name, 'stage_2'),
        'secret': calc_stage_sse(config_name, 'secret'),
    }

# Also calculate economy error
for config_name, config in configs.items():
    econ_err = ((config['total_economy_per_round'] - BASELINE['total_economy_per_round']) /
                BASELINE['total_economy_per_round']) ** 2
    stage_sse[config_name]['economy'] = econ_err

# Display results
rows = []
for config_name, config in configs.items():
    sse = stage_sse[config_name]
    total_sse = sse['economy'] + sse['stage_1'] + sse['stage_2'] + sse['secret']
    rows.append({
        'Scenario': config['name'],
        'Econ Err': round(sse['economy'], 4),
        'Stage1 Err': round(sse['stage_1'], 4),
        'Stage2 Err': round(sse['stage_2'], 4),
        'Secret Err': round(sse['secret'], 4),
        'Total SSE': round(total_sse, 4),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print()

# Final ranking
print("=" * 80)
print("FINAL RANKING (Lower = Better)")
print("=" * 80)
print()

sorted_rows = sorted(rows, key=lambda x: x['Total SSE'])
for i, r in enumerate(sorted_rows, 1):
    print(f"  {i}. {r['Scenario']}: SSE = {r['Total SSE']:.4f}")
print()

# Comparison
print("=" * 80)
print("2P/3P COMPARISON")
print("=" * 80)
print()

base_2p = [r for r in rows if r['Scenario'] == '2P Base'][0]['Total SSE']
tw_2p = [r for r in rows if r['Scenario'] == '2P Trade Wars'][0]['Total SSE']
base_3p = [r for r in rows if r['Scenario'] == '3P Base'][0]['Total SSE']
tw_3p = [r for r in rows if r['Scenario'] == '3P Trade Wars'][0]['Total SSE']

print(f"2P: Base={base_2p:.4f}, Trade Wars={tw_2p:.4f}")
if tw_2p < base_2p:
    print(f"    -> Trade Wars IMPROVES by {(base_2p - tw_2p) / base_2p * 100:.1f}%")
else:
    print(f"    -> Trade Wars WORSENS by {(tw_2p - base_2p) / base_2p * 100:.1f}%")
print()

print(f"3P: Base={base_3p:.4f}, Trade Wars={tw_3p:.4f}")
if tw_3p < base_3p:
    print(f"    -> Trade Wars IMPROVES by {(base_3p - tw_3p) / base_3p * 100:.1f}%")
else:
    print(f"    -> Trade Wars WORSENS by {(tw_3p - base_3p) / base_3p * 100:.1f}%")

# Save
output_file = DATA_DIR / 'excel_exact_sse.json'
with open(output_file, 'w') as f:
    json.dump({'rows': rows, 'stage_sse': {k: {kk: float(vv) for kk, vv in v.items()}
                                            for k, v in stage_sse.items()}}, f, indent=2)
print(f"\nResults saved to: {output_file}")
