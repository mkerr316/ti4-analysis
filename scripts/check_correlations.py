import pandas as pd
import numpy as np

# Load merged data from notebook
df = pd.read_csv(Path(__file__).parent.parent / "data" / "combined_faction_clusters.csv")

# Check correlations among community features
community_cols = ['win_ratio', 'win_deviation', 'residual_pct_pts']
print('=== COMMUNITY FEATURE CORRELATIONS ===')
print(df[community_cols].corr().round(3))

print('\n=== HIGH CORRELATIONS IN COMBINED FEATURE SET (>0.7) ===')
combined = ['total_resources', 'total_influence', 'commodities', 'has_2c4i',
            'carriers', 'infantry', 'starting_tech_count', 'faction_tech_count',
            'home_planet_count', 'has_faction_units', 'win_ratio', 'win_deviation', 'residual_pct_pts']
corr = df[combined].corr()
for i in range(len(combined)):
    for j in range(i+1, len(combined)):
        if abs(corr.iloc[i,j]) > 0.7:
            print(f'  {combined[i]} <-> {combined[j]}: {corr.iloc[i,j]:.3f}')

print('\n=== DATA ISSUES ===')
print(f'Total features: {len(combined)}')
print(f'Starting stat features: 10')
print(f'Community features: 3')
print(f'\nPROBLEM: win_ratio and win_deviation are essentially the same metric!')
print('win_deviation = win_ratio - average_win_ratio (a linear transform)')
print('This creates artificial multicollinearity in the clustering.')
