# TI4 Map Balance Analysis

Rigorous spatial-statistical evaluation of map balancing algorithms for *Twilight Imperium IV* (TI4), combining classical fairness metrics with local spatial association analysis to identify and penalize pathological resource distributions.

---

## Abstract

Standard TI4 map generators optimize numeric resource equality but are spatially blind: a map can satisfy Jain's Fairness Index while simultaneously clustering high-value systems around a single player's neighborhood, creating an asymmetric strategic advantage invisible to scalar metrics. This project formalizes three optimization algorithms — greedy hill-climbing (HC), simulated annealing (SA), and NSGA-II — against a composite objective that integrates balance gap, Moran's I spatial autocorrelation, and a LISA-based local cluster penalty. Under an equal 1,000-evaluation budget across 100 randomly generated maps, SA achieves a mean composite score of 75.42 versus 103.09 for NSGA-II and 105.78 for HC, while completing in 1.80 seconds versus 6.14 seconds for NSGA-II. The primary finding is that SA's lightweight Markov-chain trajectory search dedicates the full evaluation budget to map exploration, whereas NSGA-II expends a substantial share of the same budget on non-dominated sorting and crowding distance bookkeeping — a systematic disadvantage at low evaluation ceilings.

---

## Key Results

### Algorithm Comparison

| Algorithm | Mean Composite Score | Std Dev | Mean Wall-clock Time |
| --------- | -------------------- | ------- | -------------------- |
| Simulated Annealing (SA) | **75.42** | 58.54 | **1.80 s** |
| NSGA-II | 103.09 | 56.92 | 6.14 s |
| Greedy Hill-Climbing (HC) | 105.78 | 60.73 | 3.30 s |

*N = 100 random map seeds. Equal budget: 1,000 fitness evaluations per algorithm per seed. Run on the University of Georgia Sapelo2 HPC cluster (git: `2c252a6`).*

![Composite score boxplot](output/sapelo2-run-20260310/viz/fig1_composite_score_boxplot.png)

**Figure 1.** Grouped boxplot of composite score by algorithm across 100 seeds. Diamond markers indicate per-group means. Individual observations are overlaid as jittered points.

---

### Spatial Decomposition — The Bookkeeping Tax

Figure 2 decomposes the composite score advantage into its three constituent objectives.

![Per-objective boxplots](output/sapelo2-run-20260310/viz/fig2_per_objective_boxplots.png)

**Figure 2.** Per-objective distributions by algorithm. Left: balance gap (raw resource equality). Center: LISA penalty (local spatial cluster penalty). Right: |Moran's I| (global spatial autocorrelation).

The `balance_gap` panel reveals that all three algorithms achieve comparable numeric resource equality; the greedy structure of HC is sufficient for this scalar objective. The discriminating metric is `lisa_penalty`. HC and NSGA-II reduce the balance gap but fail to prevent the formation of local high-high (H-H) and low-low (L-L) spatial clusters — configurations in which similarly valued systems concentrate in adjacent neighborhoods, conferring systematic positional advantage on the player whose home system neighbors the H-H cluster. SA is the only algorithm that consistently suppresses these local outliers without sacrificing numeric balance.

This pattern is consistent with the computational structure of each algorithm. At a strict 1,000-evaluation ceiling, NSGA-II must perform non-dominated sorting and crowding distance calculation at every generation boundary. These operations do not evaluate new map configurations; they impose an overhead tax that reduces the number of distinct tile permutations the algorithm can examine. SA, as a single-trajectory Markov chain, carries zero per-step overhead and allocates the entire budget to actual map space exploration.

---

### Efficiency Frontier

![Efficiency scatter](output/sapelo2-run-20260310/viz/fig3_efficiency_scatter.png)

**Figure 3.** Composite score versus wall-clock time for all 300 observations. Diamond markers indicate per-algorithm means; ellipses enclose 95% confidence regions. SA occupies the lower-left corner of the quality-speed plane, indicating Pareto dominance over both HC and NSGA-II in both dimensions simultaneously.

---

### Reliability Across Map Seeds

![Per-seed improvement](output/sapelo2-run-20260310/viz/fig4_per_seed_improvement.png)

**Figure 4.** Percentage improvement in composite score over the HC baseline for each of the 100 seeds, sorted in ascending order of HC difficulty (easiest map left, hardest right). SA maintains a consistent improvement of approximately 20–30% across the full spectrum of map configurations. This refutes the hypothesis that SA's advantage is driven by a subset of atypically easy seeds; the gain is generalized and does not degrade on maps where the greedy baseline struggles most.

---

## Research Questions

**RQ1:** Does the choice of optimization algorithm significantly affect spatial fairness metrics beyond numeric balance?

*Yes. Balance gap is nearly indistinguishable across algorithms, but LISA penalty separates SA from HC and NSGA-II by a factor of approximately 2× in median value (Figure 2, center panel).*

**RQ2:** Under a fixed evaluation budget, does SA's lightweight overhead confer a systematic quality advantage over NSGA-II?

*Yes. The quality gap is consistent across all 100 seeds and is accompanied by a 3.4× reduction in wall-clock time, confirming that overhead — not solution quality per generation — is the binding constraint for NSGA-II at this budget level.*

**RQ3:** Can LISA penalty serve as a discriminating metric where Jain's Index and Moran's I are insufficient?

*Yes. Jain's Index is collinear with balance gap and provides no additional discriminating power in this experiment. Global Moran's I is also nearly uniform across algorithms. LISA penalty is the only metric that captures the local neighborhood structure relevant to per-player strategic advantage.*

---

## Methods

### Problem Formulation

Map optimization is framed as minimization of a weighted composite score over three spatial objectives:

```text
score = w₁ · balance_gap
      + w₂ · |morans_i|
      + w₃ · (1 - jains_index)
      + w₄ · lisa_penalty
```

Default weights: `w₁ = 1.0`, `w₂ = 0.5`, `w₃ = 0.5` (sign-flipped because Jain's Index is maximized), `w₄ = 0.3`. These are defined in `MultiObjectiveScore` in [`src/ti4_analysis/algorithms/spatial_optimizer.py`](src/ti4_analysis/algorithms/spatial_optimizer.py).

---

### Metrics

#### Balance Gap

```text
gap = max(player_values) - min(player_values)
```

where `player_value` is the distance-weighted sum of accessible system resources under the Joebrew evaluator (GREATEST_PLUS_TECH strategy: value = max(Resources, Influence) + tech specialty bonus).

#### Moran's I

Global spatial autocorrelation statistic:

```text
I = (N / W) × [Σᵢ Σⱼ wᵢⱼ (xᵢ − x̄)(xⱼ − x̄)] / Σᵢ (xᵢ − x̄)²
```

- I > 0: positive autocorrelation (similar values cluster)
- I ≈ 0: spatially random pattern
- I < 0: negative autocorrelation (checkerboard dispersion)

Spatial weights `wᵢⱼ` are binary adjacency weights, row-standardized. Implemented in [`src/ti4_analysis/spatial_stats/spatial_metrics.py`](src/ti4_analysis/spatial_stats/spatial_metrics.py).

#### LISA Penalty

Local Indicators of Spatial Association (Anselin, 1995). For each system, the local Moran statistic identifies H-H clusters (high-value systems neighbored by high-value systems) and L-L clusters (low-value systems neighbored by low-value systems). `lisa_penalty` sums the absolute local Moran values across all statistically significant cluster members, penalizing maps where resource richness or poverty is spatially concentrated regardless of whether the global Moran's I detects it.

#### Jain's Fairness Index

```text
J = (Σ xᵢ)² / (n × Σ xᵢ²)
```

Range [1/n, 1]; J = 1 indicates perfect equality. Included in the composite score but collinear with balance gap in practice; retained for completeness.

#### Getis-Ord Gi* (Hot Spot Analysis)

```text
Gi* = [Σⱼ wᵢⱼ xⱼ − X̄ Σⱼ wᵢⱼ] / [S √((n Σⱼ wᵢⱼ² − (Σⱼ wᵢⱼ)²) / (n−1))]
```

|Gi*| > 1.96 indicates a statistically significant cluster at 95% confidence. Used for exploratory analysis; not included in the benchmark composite score.

---

### Algorithms

#### Greedy Hill-Climbing (HC)

Iterative system-swap search that accepts a candidate move if and only if it strictly reduces the composite score. Carries no memory of prior states. Acts as the baseline: it serves as a lower bound on optimization quality and an upper bound on simplicity.

#### Simulated Annealing (SA)

Markov-chain search with acceptance criterion `P(accept) = exp(-Δ/T)`. Initial temperature `T₀` is calibrated by running a probe phase to achieve the specified `initial_acceptance_rate` for random uphill moves. The cooling rate is derived from the iteration budget as:

```text
eff_rate = (min_temp / T₀)^(1 / N)
```

This ensures that the temperature schedule spans exactly N steps regardless of the `min_temp` or `T₀` values, making `--sa-iter` the authoritative budget parameter (Kirkpatrick et al., 1983).

#### NSGA-II

Non-dominated sorting genetic algorithm optimizing the three-objective Pareto front (balance_gap, |morans_i|, lisa_penalty). Crossover uses a BFS-connected blob operator: a contiguous region of tiles is selected by breadth-first expansion from a random origin and swapped between two parent maps. This preserves local spatial coherence through crossover and is more likely to generate topologically valid offspring than radial wedge or uniform crossover. Population is initialized with a mix of warm starts (greedy HC solutions) and cold starts (random permutations). Non-dominated sorting and crowding distance selection follow Deb et al. (2002) exactly.

*Note:* Jain's Index is intentionally excluded from the Pareto objectives. It is nearly collinear with balance gap, and including a fourth objective in a 1,000-evaluation budget exacerbates the curse of dimensionality in many-objective settings.

---

### Experimental Protocol

- **Seeds:** 100 randomly generated 6-player TI4 maps (base seeds 0–999)
- **Budget:** 1,000 fitness evaluations per algorithm per seed
  - HC: 1,000 swap-evaluate iterations
  - SA: 1,000 iterations (cooling schedule derived as above)
  - NSGA-II: 50 generations × population of 20 = 1,000 evaluations
- **Hyperparameter tuning:** SA and NSGA-II parameters tuned separately on a disjoint seed range (9,000–9,014) using Bayesian TPE optimization via Optuna; tuned parameters are fixed for the main benchmark
- **Compute:** University of Georgia Sapelo2 HPC cluster; run configuration recorded in [`output/sapelo2-run-20260310/run_config.json`](output/sapelo2-run-20260310/run_config.json)

---

## Project Structure

```text
ti4-analysis/
├── src/ti4_analysis/
│   ├── algorithms/
│   │   ├── balance_engine.py        # Greedy HC baseline
│   │   ├── spatial_optimizer.py     # SA + MultiObjectiveScore
│   │   ├── nsga2_optimizer.py       # NSGA-II with BFS crossover
│   │   ├── map_generator.py         # Random map generation
│   │   ├── hex_grid.py              # Cube-coordinate geometry
│   │   ├── map_topology.py          # Static weight matrix (vectorized)
│   │   └── fast_map_state.py        # NumPy map state for O(1) swaps
│   ├── spatial_stats/
│   │   └── spatial_metrics.py       # Moran's I, LISA, Jain's, Gi*
│   └── evaluation/
│       └── batch_experiment.py      # Evaluator factory
├── scripts/
│   ├── benchmark_engine.py          # Monte Carlo benchmark (CLI)
│   ├── optimize_hyperparameters.py  # Bayesian hyperparameter tuning (Optuna)
│   └── plot_benchmark.py            # Publication figures from results.csv
├── output/
│   └── sapelo2-run-20260310/
│       ├── results.csv              # 300-row benchmark results
│       ├── run_config.json          # Reproducibility metadata + git hash
│       └── viz/                     # Generated figures (PNG + SVG)
├── tests/                           # pytest suite (property-based + unit)
├── docs/
│   └── lit_review/                  # Literature synthesis (.md files)
├── pyproject.toml
└── README.md
```

---

## Installation

**Requirements:** Python 3.9 or higher.

```bash
# Clone and enter the repository
cd ti4-analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify
python -c "import ti4_analysis; print('Installation successful')"
```

`optuna` is an optional dependency, required only for hyperparameter tuning:

```bash
pip install optuna
```

---

## Reproducing the Benchmark

```bash
# 1. Run the full benchmark (approximately 10–15 minutes on a modern laptop)
python scripts/benchmark_engine.py --seeds 100 --output-dir output/my_run/

# 2. Generate publication figures
python scripts/plot_benchmark.py --csv output/my_run/results.csv

# 3. (Optional) Tune SA hyperparameters with Bayesian optimization
python scripts/optimize_hyperparameters.py --algo sa --trials 50 --eval-seeds 15
```

The benchmark script streams results to CSV as each seed completes, so a partial run is not lost on interruption. Submit to an HPC cluster by wrapping the command in a SLURM batch script with `--output-dir` pointing to a shared filesystem path.

---

## Running Tests

```bash
# Full test suite
pytest

# With coverage report
pytest --cov=ti4_analysis --cov-report=html

# Specific module
pytest tests/test_nsga2_optimizer.py -v
```

---

## References

Anselin, L. (1995). Local indicators of spatial association — LISA. *Geographical Analysis*, 27(2), 93–115.

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics. *Geographical Analysis*, 24(3), 189–206.

Jain, R., Chiu, D. M., & Hawe, W. R. (1984). A quantitative measure of fairness and discrimination for resource allocation in shared computer systems. *DEC Research Report TR-301*.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671–680.

Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17–23.

---

## License

Same as the parent repository. See the root `LICENSE` file.
