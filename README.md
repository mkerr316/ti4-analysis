# TI4 Map Balance Analysis

Rigorous spatial-statistical evaluation of map balancing algorithms for *Twilight Imperium IV* (TI4), combining classical fairness metrics with local spatial association analysis to identify and penalize pathological resource distributions.

---

## Abstract

Standard TI4 map generators optimize numeric resource equality but are spatially blind: a map can satisfy Jain's Fairness Index while simultaneously clustering high-value systems around a single player's neighborhood, creating an asymmetric strategic advantage invisible to scalar metrics. This project formalizes four optimization algorithms — greedy hill-climbing (HC), simulated annealing (SA), NSGA-II, and Tabu Search (TS) — against a composite objective that integrates Multi-Jain Fairness (bottleneck JFI across resource dimensions), Moran's I spatial autocorrelation, and a LISA-based local cluster penalty. The benchmark includes a multi-budget saturation study (1k–100k evaluations) to identify whether population-based (NSGA-II) or memory-based (TS) methods eventually surpass SA when given sufficient evaluation budget, or whether all algorithms converge to the same near-optimal plateau.

*(The benchmark experiment reported in Key Results pre-dates this formulation and used balance gap as the distributional equity metric; see the version note in Methods.)*

---

## Key Results

> [!CAUTION]
> **Preliminary Benchmark (Pre-Multi-Jain).** The results below were generated using a scalar `balance_gap` metric and a 1,000-evaluation ceiling with 3 algorithms. These are being superseded by the **Sapelo2 Saturation Study**, which utilizes the Multi-Jain Bottleneck, Tabu Search as a fourth algorithm, and an expanded evaluation budget range (1k–100k). These preliminary results are retained as a reference point for the HC/SA/NSGA-II comparison under the original methodology.

### Preliminary Algorithm Comparison

| Algorithm | Mean Composite Score | Std Dev | Mean Wall-clock Time |
| --------- | -------------------- | ------- | -------------------- |
| Simulated Annealing (SA) | **75.42** | 58.54 | **1.80 s** |
| NSGA-II | 103.09 | 56.92 | 6.14 s |
| Greedy Hill-Climbing (HC) | 105.78 | 60.73 | 3.30 s |

*N = 100 random map seeds. Equal budget: 1,000 fitness evaluations per algorithm per seed. Run on the University of Georgia Sapelo2 HPC cluster (git: `2c252a6`). Composite score uses balance gap, not Multi-Jain JFI.*

![Composite score boxplot](output/sapelo2-run-20260310/viz/fig1_composite_score_boxplot.png)

**Figure 1.** Grouped boxplot of composite score by algorithm across 100 seeds. Diamond markers indicate per-group means. Individual observations are overlaid as jittered points.

---

### Spatial Decomposition — The Bookkeeping Tax

Figure 2 decomposes the composite score advantage into its three constituent objectives.

![Per-objective boxplots](output/sapelo2-run-20260310/viz/fig2_per_objective_boxplots.png)

**Figure 2.** Per-objective distributions by algorithm. Left: balance gap *(distributional equity proxy used in benchmark run at git `2c252a6`; superseded by JFI in current code)*. Center: LISA penalty (local spatial cluster penalty). Right: |Moran's I| (global spatial autocorrelation).

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

**RQ1:** Does the choice of optimization algorithm significantly affect spatial equity metrics beyond numeric balance?

*Yes. Balance gap is nearly indistinguishable across algorithms, but LISA penalty separates SA from HC and NSGA-II by a factor of approximately 2× in median value (Figure 2, center panel).*

**RQ2:** Under a fixed evaluation budget, does SA's lightweight overhead confer a systematic quality advantage over NSGA-II?

*Yes. The quality gap is consistent across all 100 seeds and is accompanied by a 3.4× reduction in wall-clock time, confirming that overhead — not solution quality per generation — is the binding constraint for NSGA-II at this budget level.*

**RQ3:** Can LISA penalty serve as a discriminating metric where balance gap and Moran's I are insufficient?

*Yes. Balance gap and global Moran's I are nearly uniform across algorithms. LISA penalty is the only metric that captures the local neighborhood structure relevant to per-player strategic advantage. Jain's Fairness Index (JFI) has since replaced balance gap as the distributional equity objective in the current codebase: JFI is dimensionless, bounded [1/n, 1], and axiomatic under peer-reviewed fairness criteria, whereas balance gap is a brittle range measure sensitive to single outliers.*

---

## Methods

### Problem Formulation

Map optimization is framed as minimization of a weighted composite score over three objectives:

$$S = w_1 \cdot (1 - J_{\min}) + w_2 \cdot |I| + w_3 \cdot \frac{\text{LISA}_+}{n(n-1)}$$

where $J_{\min} = \min(J_R, J_I)$ is the **Multi-Jain bottleneck** — Jain's Fairness Index computed independently on distance-weighted raw Resources and raw Influence, with the minimum (bottleneck) dimension determining the fairness term. This is inspired by Dominant Resource Fairness (Ghodsi et al., 2011): map fairness is limited by the *least fair* resource dimension.

Default weights: `w₁ = 5/13 ≈ 0.385`, `w₂ = 5/13 ≈ 0.385`, `w₃ = 3/13 ≈ 0.231` (ratios 5:5:3, sum = 1.0). These are defined in `MultiObjectiveScore` in [`src/ti4_analysis/algorithms/spatial_optimizer.py`](src/ti4_analysis/algorithms/spatial_optimizer.py). A weight sensitivity analysis (`--sensitivity` flag in `analyze_benchmark.py`) demonstrates whether algorithm rankings are robust to alternative weight configurations (equal, JFI-dominant, spatial-dominant, LISA-dominant).

All three terms are normalized to [0, 1] before weighting:

- `1 − J_min ∈ [0, 1]` — multi-dimensional distributive equity (bottleneck JFI across resource dimensions); minimizing this maximizes the fairness of the least-fair dimension
- `|morans_i| ∈ [0, 1]` — global spatial clustering (theoretical bound for row-standardized W)
- `lisa_penalty / [n × (n − 1)] ∈ [0, 1]` — local H-H / L-L cluster penalty; the divisor `n × (n − 1)` is the theoretical maximum of the sum of positive variance-normalized local Moran's I values (one extreme-value location surrounded by identical-deviation neighbours, replicated across all n positions)

`balance_gap` (max − min player value) is retained as a stored attribute on `MultiObjectiveScore` for display and reporting, but is excluded from the composite score and all Pareto dominance calculations.

> **Version note:** The benchmark results in Key Results were produced at git `2c252a6` with an earlier formula that used `balance_gap` (weight 1.0) as the distributional equity term in place of `1 − jains_index`. The methodology sections below describe the current code.

---

### Distance-Weighting Model (Step-Function Decay)

Per-player home values are computed as the weighted sum of all reachable system tile values within a hex-distance range gate of 5. Unlike continuous inverse-distance power laws ($w = 1/d^\gamma$) commonly used in gravity models in spatial analysis, we employ a **discrete step-function decay** based on terrain-cost pathfinding. This reflects the quantized nature of board game movement: a ship with 2 movement reaches hexes at distance 1 or 2 with equal ease — it cannot "partially" reach a system 3 hexes away. A continuous decay would impose artificial differentiation between strategically identical positions, while introducing a free parameter ($\gamma$) with no ground-truth calibration source in the game domain.

For each (home $p$, system $s$) pair:

1. **Routing distance.** Compute the shortest "modded distance" $d_m(p, s)$ via BFS pathfinding, where each hop along the path accumulates a terrain cost from `System.get_distance_modifier()`. Supernovae block the path entirely (system unreachable).

2. **Step-function weight lookup.** Map the modded distance to a weight via a discrete multiplier table $\mathbf{w}$:

$$v_p = \sum_{s \in \mathcal{R}(p)} \mathbf{w}\bigl[\lfloor d_m(p, s) \rfloor\bigr] \cdot \text{val}(s)$$

The default `DISTANCE_MULTIPLIER` table (community-calibrated by the TI4 "Joebrew" evaluator to match experienced players' intuitions about tile accessibility):

| $\lfloor d_m \rfloor$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7+ |
| ---------------------- | - | - | - | - | - | - | - | -- |
| **Weight** | 6 | 6 | 6 | 4 | 4 | 2 | 1 | 0 |

The "shelf" at weight 6 for $d_m \in [0, 2]$ is the defining feature: it captures the game-mechanical reality that nearby systems are equally accessible within a single round's movement, while systems behind anomaly barriers are progressively harder to exploit.

Terrain costs per hop under the Joebrew evaluator:

| Terrain | Cost per hop |
| ------- | ------------ |
| Blue / planet system | 0.0 |
| Empty space | 0.0 |
| Asteroid field | 1.0 |
| Nebula | 1.0 |
| Gravity rift | 0.0 |
| Supernova | path blocked |

With these defaults, most paths traverse only blue/planet and empty-space hexes, yielding $d_m = 0$ and uniform weight 6 for all reachable systems. Anomaly-traversing paths accumulate positive distance and receive reduced weight from the lookup table. This models the strategic reality that asteroid fields and nebulae degrade tile accessibility in TI4, without imposing an artificial continuous decay on the inherently discrete hex grid.

For Multi-Jain per-dimension JFI, the same weight matrix is applied to raw planet resources and raw planet influence separately (before any evaluator scalarization), producing vectors $\mathbf{r}$ and $\mathbf{i}$ for the bottleneck JFI computation.

> **Planned: distance-weight sensitivity analysis.** The `DISTANCE_MULTIPLIER` table is community-calibrated, not theoretically derived. A post-hoc sensitivity analysis will perturb the table entries (e.g., `[6,6,5,3,3,2,1]`, `[6,6,6,6,4,2,1]`) and verify that algorithm rankings remain stable under alternative distance decay assumptions, confirming that the benchmark conclusions are not brittle to the specific weight values.

---

### Metrics

#### Balance Gap

$$\text{gap} = \max(\mathbf{v}) - \min(\mathbf{v})$$

where `player_value` is the distance-weighted sum of accessible system resources under the Joebrew evaluator (GREATEST_PLUS_TECH strategy: value = max(Resources, Influence) + tech specialty bonus).

`balance_gap` is retained as a display attribute on `MultiObjectiveScore` for human-readable reporting (e.g., "Player A has 5 more resources than Player B"), but is excluded from the composite score and all Pareto dominance calculations in the current version. Jain's Fairness Index serves as the axiomatic replacement on the distributional equity axis.

#### Moran's I

Global spatial autocorrelation statistic:

$$I = \frac{N}{W} \cdot \frac{\sum_i \sum_j w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}$$

- $I > 0$: positive autocorrelation (similar values cluster)
- $I \approx 0$: spatially random pattern
- $I < 0$: negative autocorrelation (checkerboard dispersion)

Spatial weights $w_{ij}$ are binary adjacency weights, row-standardized so that each row of $\mathbf{W}$ sums to 1.0 (implemented in [`src/ti4_analysis/algorithms/map_topology.py`](src/ti4_analysis/algorithms/map_topology.py), lines 208–212). Row-standardization mitigates boundary effects on the bounded TI4 hex grid: outer-ring systems have fewer neighbours (typically 2–3) than inner systems (up to 6), which would otherwise systematically deflate their spatial lag values. After standardization, the spatial lag $(\mathbf{W}\mathbf{z})_i$ at each position is a proper weighted average of its neighbours regardless of neighbour count, ensuring that edge systems are not artificially penalized or advantaged in the Moran's I and LISA computations. Zero-sum rows (fully isolated hexes, if any) are guarded against by substituting a denominator of 1.0 to avoid division by zero.

#### LISA Penalty

Local Indicators of Spatial Association (Anselin, 1995). For each system $i$, the variance-normalised local Moran statistic is:

$$I_i = \frac{(x_i - \bar{x}) \sum_j w_{ij}(x_j - \bar{x})}{m_2}, \quad \text{where } m_2 = \frac{\sum(x_i - \bar{x})^2}{n}$$

Positive $I_i$ identifies H-H clusters (high-value systems neighbored by high-value systems) and L-L clusters (low-value systems neighbored by low-value systems). `lisa_penalty` sums only the positive local values, penalizing maps where resource richness or poverty is spatially concentrated regardless of whether the global Moran's I detects it. Dividing by $m_2$ makes the values dimensionless, ensuring $\sum I_i \approx n \times I_{\text{global}}$ and proper scaling relative to the other composite-score terms.

**LISA normalization bound.** The composite score normalizes the LISA penalty by $n(n-1)$. This bound holds specifically because $\mathbf{W}$ is row-standardized: each row sums to 1, so the spatial lag $(\mathbf{W}\mathbf{z})_i$ is a weighted *average* of neighbours, bounded by the range of $\mathbf{z}$. Without row-standardization, the spatial lag would be an unweighted sum whose magnitude depends on each node's degree, making the bound graph-dependent rather than universal. Under row-standardization, the variance-normalized local Moran's statistic at a single position is maximized when that position has extreme deviation and all its neighbours share the same sign: $(\mathbf{W}\mathbf{z})_i = z_i$ in this fully-correlated limit, giving $I_i = z_i^2 / m_2 = n - 1$ for the single-outlier configuration where $m_2 = z_i^2 / n$. At most $n$ positions can contribute positively, so $\sum_{I_i > 0} I_i \leq n(n-1)$, and dividing by this value maps the LISA penalty to $[0, 1]$.

> **Methodological note — continuous heuristic proxy.** In classical spatial statistics, LISA significance is determined via conditional permutation tests (typically 999 permutations per location) to distinguish genuine clusters from chance variation (Anselin, 1995). In an optimization loop evaluating thousands of candidate maps, embedding a permutation test inside each fitness evaluation would multiply computational cost by ~1,000×, making it infeasible under real-time budget constraints. The LISA penalty used here therefore sums *all* positive $I_i$ values without significance filtering, serving as a continuous, smooth fitness signal that provides a gradient for the optimizer rather than a binary significance classification. This design choice preserves a differentiable fitness landscape for SA's Metropolis criterion while still penalizing the spatial clustering patterns that significance-filtered LISA would flag at larger sample sizes.
>
> **Post-hoc validation.** To verify that minimising the continuous proxy successfully eliminates *statistically significant* clusters, `scripts/validate_lisa_proxy.py` runs a separate post-hoc analysis: for a subset of seeds, each algorithm's final map is subjected to full conditional-permutation LISA (999 permutations per location, $p < 0.05$). The number of significant H-H and L-L clusters is counted and compared across algorithms. If SA's proxy-penalized maps consistently show fewer significant clusters than HC's maps, the proxy is empirically validated.

#### Multi-Jain Fairness Index (Bottleneck JFI)

For a single resource dimension, Jain's Fairness Index on a vector $\mathbf{x}$ of per-player values is:

$$J(\mathbf{x}) = \frac{\left(\sum x_i\right)^2}{n \sum x_i^2}$$

Range $[1/n, 1]$; $J = 1$ indicates perfect equality.

The engine computes JFI independently on **raw Resources** and **raw Influence** — the distance-weighted planet totals per player *before* any evaluator scalarization. The composite score uses the **bottleneck** (minimum) across dimensions:

$$J_{\min} = \min\bigl(J(\mathbf{r}),\; J(\mathbf{i})\bigr)$$

where $\mathbf{r}$ and $\mathbf{i}$ are the per-player distance-weighted resource and influence totals respectively. This Multi-Jain formulation is inspired by Dominant Resource Fairness (Ghodsi et al., 2011): a map's fairness is limited by its *least fair* resource dimension. A map where Resources are perfectly balanced but Influence is heavily skewed will be penalized, even though a single-scalar JFI on the Joebrew value might report high fairness.

Both per-dimension values ($J_R$, $J_I$) are recorded in the benchmark CSV for disaggregated analysis. JFI replaces balance gap (max − min range) as the axiomatic, scale-invariant measure of resource disparity (Jain et al., 1984). Note: JFI has been shown to correlate with human fairness ratings in one-to-many allocation games (Grappiolo et al., 2013), but the link between spatial autocorrelation (Moran's I) and player-perceived map fairness remains an open empirical question requiring future user studies.

> **Relationship to DRF.** DRF (Ghodsi et al., 2011) guarantees envy-freeness and strategy-proofness by evaluating the dominant share across multiple resource types. The Multi-Jain bottleneck captures the same intuition — fairness constrained by the worst dimension — without claiming the formal DRF properties, which require an allocation mechanism rather than a fixed map topology. Moran's I and LISA continue to operate on the combined Joebrew scalar (`max(R, I) + tech`), since spatial clustering concerns total tile value regardless of resource dimension.

#### Getis-Ord Gi* (Hot Spot Analysis)

$$G_i^* = \frac{\sum_j w_{ij} x_j - \bar{X} \sum_j w_{ij}}{S \sqrt{\frac{n \sum_j w_{ij}^2 - \left(\sum_j w_{ij}\right)^2}{n - 1}}}$$

$|G_i^*| > 1.96$ indicates a statistically significant cluster at 95% confidence. Used for exploratory analysis; not included in the benchmark composite score.

---

### Algorithms

#### Greedy Hill-Climbing (HC)

Iterative system-swap search that accepts a candidate move if and only if it strictly reduces the composite score. Carries no memory of prior states. Acts as the baseline: it serves as a lower bound on optimization quality and an upper bound on simplicity.

#### Simulated Annealing (SA)

Markov-chain search with acceptance criterion $P(\text{accept}) = \exp(-\Delta / T)$. Initial temperature $T_0$ is calibrated by running a probe phase to achieve the specified `initial_acceptance_rate` for random uphill moves. The cooling rate is derived from the iteration budget as:

$$\alpha = \left(\frac{T_{\min}}{T_0}\right)^{1/N}$$

This ensures that the temperature schedule spans exactly $N$ steps regardless of the $T_{\min}$ or $T_0$ values, making `--sa-iter` the authoritative budget parameter (Kirkpatrick et al., 1983).

#### NSGA-II

Non-dominated sorting genetic algorithm optimizing the three-objective Pareto front (1 − jains_index, |morans_i|, lisa_penalty). Crossover uses a BFS-connected blob operator: a contiguous region of tiles is selected by breadth-first expansion from a random origin and swapped between two parent maps. This preserves local spatial coherence through crossover and is more likely to generate topologically valid offspring than radial wedge or uniform crossover. Population is initialized with a mix of warm starts (greedy HC solutions) and cold starts (random permutations). Non-dominated sorting and crowding distance selection follow Deb et al. (2002) exactly. Crowding distance normalization within each front makes NSGA-II scale-invariant across objectives.

**A posteriori scalarization.** NSGA-II returns a Pareto front of non-dominated maps, not a single solution. For scalar comparison against SA and HC, the benchmark applies the composite score weight vector (5:5:3) to every member of the final Pareto front and selects the member with the lowest composite score. This *a posteriori* scalarization is the standard method for comparing MOEAs against single-objective algorithms (see `benchmark_engine.py`, line 227). The `front_size` column in the results CSV records the Pareto front cardinality for each seed.

**Depth-vs-breadth tradeoff.** Under a fixed evaluation budget, population-based methods (NSGA-II) trade search *depth* for search *breadth*: a population of 20 over 50 generations explores only 50 sequential selection-crossover-mutation cycles, whereas SA explores 1,000 sequential steps along a single Markov chain. In a highly rugged combinatorial landscape ($\sim 3.7 \times 10^{89}$ tile permutations), trajectory depth is critical for escaping local optima. NSGA-II's disadvantage at low budgets is therefore structural, not merely an overhead artefact. The multi-budget convergence profile (`--budgets 1000,5000,10000,25000,50000,100000`) characterizes whether NSGA-II's breadth advantage recovers at larger budgets.

#### Tabu Search (TS)

Full-neighbourhood deterministic search with short-term memory. Each iteration evaluates all $\binom{S}{2}$ possible 2-swaps among $S$ swappable positions, selects the best non-tabu move, and unconditionally applies it — even if it worsens the current score. This deterministic escape from local optima distinguishes TS from SA's stochastic Metropolis criterion. The tabu list forbids recently executed swaps for a configurable *tenure* period (default: $\lceil\sqrt{S}\rceil$, per Glover 1989). An aspiration criterion overrides the tabu restriction when a move produces a score better than the global best.

**Evaluation budget.** Each candidate swap costs one evaluation. A single TS iteration costs $\binom{S}{2}$ evaluations (≈ 435 for $S = 30$), so at budget 1,000 TS completes only ≈ 2 full iterations versus SA's 1,000 sequential steps. This makes TS the evaluation-heaviest algorithm per iteration but also the most informed per decision — the classic breadth-per-step vs. depth-over-time tradeoff. At higher budgets (50k–100k), TS accumulates enough iterations for the tabu memory to meaningfully shape the search trajectory.

**Role.** TS serves as a *methodological control*: the Terra Mystica PCG literature (Sironi et al., 2019) demonstrated TS superiority over steepest-ascent HC for tabletop map generation. Including TS validates whether SA's stochastic escape mechanism is redundant with TS's deterministic memory, or whether the two approaches find qualitatively different solutions. If TS ≈ SA at convergence, SA's lower per-iteration cost makes it the strictly superior production algorithm.

#### Random Search (RS) — Baseline

Uniform random sampling over the full permutation space. For each evaluation, RS generates a uniformly random arrangement of all swappable tiles (via Fisher-Yates shuffle), scores the result, and retains the best score seen. Each sample is independent — no sequential memory, no gradient signal. RS establishes whether the metaheuristic search strategies (SA, TS, NSGA-II) add value beyond brute-force random exploration. If SA cannot statistically outperform RS at the same evaluation budget, the search intelligence is provably unnecessary.

#### Algorithm Role Summary

| Algorithm | Research Role | Production Role (Rust App) |
| --------- | ------------- | -------------------------- |
| **NSGA-II** | Ground truth: maps the theoretical limits of fairness trade-offs via multi-objective Pareto front | "Pro" generator: pre-computes gold-standard map libraries offline |
| **SA** | Production baseline: proven to reach near-optimal quality in a fraction of the time | Default live engine for "Generate New Map" clicks (<2 s) |
| **TS** | Methodological control: validates that SA's stochastic escape is not missing deterministic-memory-accessible optima | Excluded from production: high per-iteration cost for marginal gain |
| **HC** | Lower bound: establishes the baseline quality achievable without local-optima escape | Warm-start seed for NSGA-II population inoculation |
| **RS** | Null baseline: proves that intelligent search outperforms uniform random sampling | Not applicable |

#### Bayesian Optimization (BO) — Excluded

BO is excluded from the benchmark. The per-evaluation cost of TI4 map fitness (<1 ms via vectorized numpy) makes surrogate-model overhead (GP fitting at $O(n^3)$ in observed evaluations, acquisition function optimization) counterproductive: the same wall-time budget achieves orders of magnitude more evaluations via SA. Additionally, the discrete permutation search space ($S! \approx 2.65 \times 10^{32}$) is poorly suited to GP-based surrogates, which assume continuous, low-dimensional inputs. A detailed justification is provided in [`docs/bayesian_optimization_exclusion.md`](docs/bayesian_optimization_exclusion.md).

---

### Experimental Protocol

- **Seeds:** 100 randomly generated 6-player TI4 maps (base seeds 0–999)
- **Budget:** 1,000 fitness evaluations per algorithm per seed (extended to 100k for saturation study)
  - HC: 1,000 swap-evaluate iterations
  - SA: 1,000 iterations (cooling schedule derived as above)
  - NSGA-II: 50 generations × population of 20 = 1,000 evaluations
  - TS: full-neighbourhood iterations until budget exhausted (≈ 2 iterations at 1,000 budget)
- **Hyperparameter tuning:** SA, NSGA-II, and TS parameters tuned separately on a disjoint seed range (9,000–9,014) using Bayesian TPE optimization via Optuna (50 trials each); tuned parameters are fixed for the main benchmark. TS tunes `tabu_tenure` ∈ [3, 20] to ensure symmetric treatment across all stochastic algorithms
- **Ablation:** Multi-Jain bottleneck JFI vs optimistic (max-dimension) JFI comparison to demonstrate that the bottleneck formulation catches maps with hidden dimensional imbalance
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
│   │   ├── tabu_search_optimizer.py # Tabu Search (full-neighbourhood)
│   │   ├── map_generator.py         # Random map generation
│   │   ├── hex_grid.py              # Cube-coordinate geometry
│   │   ├── map_topology.py          # Static weight matrix (vectorized)
│   │   └── fast_map_state.py        # NumPy map state for O(1) swaps
│   ├── spatial_stats/
│   │   └── spatial_metrics.py       # Moran's I, LISA, Jain's, Gi*
│   └── evaluation/
│       └── batch_experiment.py      # Evaluator factory
├── scripts/
│   ├── benchmark_engine.py          # Monte Carlo benchmark (CLI, multi-budget)
│   ├── analyze_benchmark.py         # Non-parametric stats + weight sensitivity
│   ├── validate_lisa_proxy.py       # Post-hoc permutation-tested LISA validation
│   ├── plot_statistical_results.py  # Publication-quality figures
│   ├── optimize_hyperparameters.py  # Bayesian hyperparameter tuning (Optuna)
│   └── plot_benchmark.py            # Legacy figures from results.csv
├── output/
│   └── sapelo2-run-20260310/
│       ├── results.csv              # 300-row benchmark results
│       ├── run_config.json          # Reproducibility metadata + git hash
│       └── viz/                     # Generated figures (PNG + SVG)
├── tests/                           # pytest suite (property-based + unit)
├── docs/
│   ├── lit_review/                  # Literature synthesis (.md files)
│   ├── tabu_search_justification.md # TS design rationale
│   └── bayesian_optimization_exclusion.md  # BO exclusion argument
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
# 1. Run the full benchmark with all four algorithms
python scripts/benchmark_engine.py --seeds 100 --algorithms hc,sa,nsga2,ts --output-dir output/my_run/

# 1b. Saturation study: find where algorithms converge (recommended for Sapelo2)
python scripts/benchmark_engine.py --seeds 100 --algorithms hc,sa,nsga2,ts \
    --budgets 1000,5000,10000,25000,50000,100000

# 2. Non-parametric statistical analysis (Friedman, Wilcoxon, VDA, bootstrap)
python scripts/analyze_benchmark.py output/my_run/results.csv

# 2b. (Optional) Weight sensitivity analysis across 5 weight configs
python scripts/analyze_benchmark.py output/my_run/results.csv --sensitivity

# 3. Generate publication figures
python scripts/plot_statistical_results.py output/my_run/results.csv

# 4. (Optional) Post-hoc LISA validation — permutation-tested significance
python scripts/validate_lisa_proxy.py --seeds 30

# 5. (Optional) Tune SA hyperparameters with Bayesian optimization
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

## Future Work

- **Fitness landscape ruggedness analysis.** A formal characterization of the search space ruggedness — via autocorrelation length (Weinberger, 1990) or NK-landscape analysis (Kauffman, 1993) — would quantify the density of local optima and empirically justify the need for local-optima escape mechanisms (SA Metropolis criterion, TS tabu memory). A 2D fitness landscape slice (fixing all but two tile positions and sweeping their swap) would provide an intuitive visualization of this ruggedness to complement the convergence profiles from the saturation study.
- **Human-subject validation.** JFI has been shown to correlate with human fairness ratings in allocation games (Grappiolo et al., 2013), but the link between spatial autocorrelation metrics (Moran's I, LISA) and player-perceived map fairness remains an open empirical question. Telemetry from the companion Rust/Tauri community application could source the necessary play data for a future user study.
- **Adaptive weight tuning.** The current composite score weights (5:5:3) are fixed. A Pareto-front-guided approach could learn weight preferences from player feedback, enabling personalized map generation.

---

## References

Anselin, L. (1995). Local indicators of spatial association — LISA. *Geographical Analysis*, 27(2), 93–115.

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

Glover, F. (1989). Tabu search — Part I. *ORSA Journal on Computing*, 1(3), 190–206.

Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics. *Geographical Analysis*, 24(3), 189–206.

Ghodsi, A., Zaharia, M., Hindman, B., Konwinski, A., Shenker, S., & Stoica, I. (2011). Dominant resource fairness: Fair allocation of multiple resource types. *Proceedings of the 8th USENIX Symposium on Networked Systems Design and Implementation (NSDI)*, 24, 323–336.

Grappiolo, C., Martínez, H. P., & Yannakakis, G. N. (2013). Validating generic metrics of fairness in game-based resource allocation scenarios with crowdsourced annotations. *Transactions on Computational Collective Intelligence*, 13, 176–200.

Jain, R., Chiu, D. M., & Hawe, W. R. (1984). A quantitative measure of fairness and discrimination for resource allocation in shared computer systems. *DEC Research Report TR-301*.

Kauffman, S. A. (1993). *The Origins of Order: Self-Organization and Selection in Evolution*. Oxford University Press.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671–680.

Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17–23.

Weinberger, E. D. (1990). Correlated and uncorrelated fitness landscapes and how to tell the difference. *Biological Cybernetics*, 63(5), 325–336.

---

## License

Same as the parent repository. See the root `LICENSE` file.
