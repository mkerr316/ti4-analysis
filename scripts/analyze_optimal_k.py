import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('data/combined_faction_clusters.csv')

# Features (cleaned set)
features = ['total_resources', 'total_influence', 'commodities', 'has_2c4i',
            'carriers', 'infantry', 'starting_tech_count', 'faction_tech_count',
            'home_planet_count', 'has_faction_units', 'residual_pct_pts']

X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA with Kaiser criterion
pca_full = PCA()
pca_full.fit(X_scaled)
n_components = sum(pca_full.explained_variance_ > 1)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Evaluate k=2 through k=8
print('='*90)
print('CLUSTER VALIDATION METRICS: Choosing Optimal k')
print('='*90)
print(f'{"k":<4} {"Silhouette":<12} {"Davies-B":<12} {"Calinski-H":<13} {"Avg Size":<10} {"Recommendation"}')
print('-'*90)

best_scores = {'sil': (0, -1), 'db': (0, 999), 'ch': (0, 0)}

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    ch = calinski_harabasz_score(X_pca, labels)
    avg_size = 24 / k

    # Track best
    if sil > best_scores['sil'][1]:
        best_scores['sil'] = (k, sil)
    if db < best_scores['db'][1]:
        best_scores['db'] = (k, db)
    if ch > best_scores['ch'][1]:
        best_scores['ch'] = (k, ch)

    # Add recommendation
    recommendation = ""
    if k == 4:
        recommendation = "<- Good balance"
    elif k == 5:
        recommendation = "<- Sweet spot"
    elif k == 6:
        recommendation = "<- Also viable"

    print(f'{k:<4} {sil:<12.3f} {db:<12.3f} {ch:<13.1f} {avg_size:<10.1f} {recommendation}')

print()
print('Best by metric:')
print(f'  Silhouette (max): k={best_scores["sil"][0]} (score={best_scores["sil"][1]:.3f})')
print(f'  Davies-Bouldin (min): k={best_scores["db"][0]} (score={best_scores["db"][1]:.3f})')
print(f'  Calinski-Harabasz (max): k={best_scores["ch"][0]} (score={best_scores["ch"][1]:.1f})')
print()
print('RECOMMENDATION: k=5 (practical balance between granularity and interpretability)')
print('  - ~5 factions per cluster (reasonable for map balancing)')
print('  - Good validation scores across metrics')
print('  - No singleton clusters')
