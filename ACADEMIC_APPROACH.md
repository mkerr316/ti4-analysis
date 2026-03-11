# TI4 Map Optimization: Academic Benchmarking Approach

## Overview

This repository implements a rigorous academic framework for comparing multi-objective optimization algorithms on hexagonal game map generation.

**NO GAMEPLAY VALIDATION** - This is a pure computational experiment comparing optimization algorithms.

## Research Questions

1. **RQ1**: Does NSGA-II converge faster than hill-climbing?
2. **RQ2**: Does NSGA-II produce superior Pareto fronts?
3. **RQ3**: Does unconstrained action space (red tile swapping) expand the Pareto front?
4. **RQ4**: What trade-offs exist between balance gap and spatial distribution?
5. **RQ5**: Which algorithm is most computationally efficient?

## Six Algorithms

| ID | Algorithm | Type | Objective |
|------|-----------|------|-----------|
| `rs` | Random Search | Null baseline | Scalar composite (5:5:3) |
| `hc` | Greedy Hill-Climbing | Local search (no escape) | Scalar composite (5:5:3) |
| `sa` | Simulated Annealing | Markov chain, single-trajectory | Scalar composite (5:5:3) |
| `sga` | Single-Objective GA | Population-based, scalar tournament | Scalar composite (5:5:3) |
| `nsga2` | NSGA-II | Population-based, Pareto | 3-objective: (1−JFI, \|I\|, LSAP) |
| `ts` | Tabu Search | Deterministic memory, full-neighbourhood | Scalar composite (5:5:3) |

> **Naming note:** The production benchmark pipeline (`submit_all.sh`,
> `benchmark_engine.py`) uses lowercase IDs: `rs`, `hc`, `sa`, `sga`, `nsga2`, `ts`.
> SGA was added to isolate the architecture vs objective-type comparison
> (SA vs SGA = same scalar, different architecture; SGA vs NSGA-II = same
> operators, different objective). The legacy G-prefix aliases are retired.

### Objective hierarchy (Nominal Scalarization)

**Track B (HV, IGD+, Spacing) is the primary evaluation** — the weight-independent Pareto metrics are the gold standard for the manuscript and disarm concerns about arbitrary weighting. Given the lack of consensus on optimal weighting for composite spatial indicators (Libório et al.), we use a **5:5:3** ratio (Fairness : Clustering : Local Penalty) as a **Nominal Scalarization**. It provides a fixed target for single-objective trajectories (SA, TS, HC, SGA). The 5:5:3 choice is defensible only when **weight-sensitivity analysis** (`analyze_benchmark.py --sensitivity`) shows that algorithm rankings (e.g. SA over HC) hold across equal weights, JFI-dominant, and spatial-dominant configurations; then 5:5:3 serves as a nominal anchor. HV and IGD+ (Ishibuchi et al., 2015) are computed against an **empirical reference front** — the combined set of all observed non-dominated points across seeds — which is the standard approach when the true Pareto front of the problem is unknown.

## Dependent Variables (Gold-Standard Quality Indicators)

### Primary Metrics (Track B — gold standard; implemented in `scripts/quality_indicators.py`)
- **Hypervolume (HV)**: Volume dominated by Pareto front (higher = better)
- **IGD+**: Inverted Generational Distance Plus (lower = better, Pareto-compliant)
- **Spacing**: Uniformity of solutions along front (lower = better)

### Secondary Metrics
- `evals_to_best`: evaluation count at which the incumbent was last improved (convergence tracking)
- Function evaluations to convergence
- Best composite score, best Jain's Index
- Pareto front cardinality (`front_size`)

### Efficiency Metrics
- Wall-clock time
- Function evaluations
- `evals_to_best` as fraction of budget (convergence efficiency)

## Quick Start

### 1. Install Dependencies

```bash
cd ti4-analysis
pip install -r requirements.txt
```

### 2. Run Academic Benchmark

Open `notebooks/08_academic_benchmark_study.ipynb` in Jupyter and execute all cells.

**Configuration options:**
- `n_trials=30` - Number of independent runs per algorithm (reduce to 3 for testing)
- `iterations=500` - Optimization iterations/generations
- `population_size=100` - NSGA-II population size

**Estimated runtime:** 1-2 hours for full 30 trials × 3 algorithms

### 3. Analyze Results

The notebook automatically:
- Runs 90 optimization trials (30 per algorithm)
- Calculates all quality indicators
- Performs statistical tests (Wilcoxon with Bonferroni correction)
- Generates publication-ready figures
- Exports LaTeX tables

**Output directory:** `results/academic_benchmark/`

## File Structure

```
ti4-analysis/
├── src/ti4_analysis/
│   ├── algorithms/
│   │   ├── nsga2_optimizer.py     (PyMOO NSGA-II implementation)
│   │   ├── hc_optimizer.py        (Benchmark HC: composite 5:5:3)
│   │   ├── balance_engine.py      (Gap-only HC: spatial blindness exp, warm start)
│   │   └── spatial_optimizer.py   (SA, multi-objective)
│   ├── benchmarking/              (NEW: Academic framework)
│   │   ├── quality_indicators.py  (HV, IGD+, Spacing)
│   │   ├── statistical_tests.py   (Wilcoxon, Friedman, effect sizes)
│   │   └── run_experiment.py      (Automated experiment runner)
│   ├── spatial_stats/
│   │   └── spatial_metrics.py     (Moran's I, Jain's Index)
│   └── visualization/
│       └── map_viz.py             (Rendering utilities)
├── notebooks/
│   └── 08_academic_benchmark_study.ipynb  (PRIMARY WORKFLOW)
└── tests/

```

## Key Python Packages

- **pymoo**: Multi-objective optimization (NSGA-II)
- **scipy**: Statistical tests (Wilcoxon, Friedman)
- **networkx**: Graph algorithms (betweenness centrality)
- **pysal/esda**: Spatial statistics (Moran's I)
- **matplotlib/seaborn**: Visualization

## What This Proves (Academically)

### ✅ You CAN claim:
- "We compare three optimization algorithms using gold-standard quality indicators"
- "30 independent trials provide robust statistical power"
- "Wilcoxon tests with Bonferroni correction control family-wise error"
- "Effect sizes (Cohen's d, A12) quantify practical significance"
- "NSGA-II efficiently optimizes multiple spatial objectives simultaneously"

### ❌ You CANNOT claim:
- ~~"AI agents played full games"~~
- ~~"This measures actual gameplay outcomes"~~
- ~~"Win-rate variance validates experiential balance"~~

## Publication Output

The notebook generates:
- **CSV**: `results.csv` with all trial data
- **Figures**: Box plots, violin plots, CDFs, scatter plots
- **LaTeX Tables**: Statistical comparisons for papers
- **JSON**: Summary statistics and test results

## References

- Deb et al. (2002): NSGA-II algorithm
- Blank & Deb (2020): PyMOO framework
- Ishibuchi et al. (2015): IGD+ metric
- Zitzler & Thiele (1999): Hypervolume indicator
- García et al. (2010): Non-parametric statistical tests for MOO

## License

See main repository LICENSE file.
