# TI4 Map Optimization: Academic Benchmarking Approach

## Overview

This repository implements a rigorous academic framework for **geographic/methodological** contribution: spatial metrics (Moran's I, LSAP) add meaningful constraint beyond scalar fairness (JFI) on topologically embedded discrete spaces. The main experiment is a **four-condition ablation** (JFI-only, Moran's I only, LSAP only, full composite 1:1:1) using SA as the instrument. Algorithms are **instruments**; algorithm comparison is **methods justification**, not the primary result. See [Design_Rationale.md](docs/methodology/Design_Rationale.md).

**NO GAMEPLAY VALIDATION** — This is a pure computational experiment. Human validation (causal chain from spatial clustering to competitive disadvantage in play) is future work.

## Research Questions (main experiment)

1. Do maps from the full-composite condition differ from JFI-only in their spatial profile (LSAP, Moran's I hinge)?
2. Does JFI parity hold for the full-composite condition (no fairness sacrifice)?
3. Is the spatial profile difference detectable at all budgets ≥ 10k (pre-specified)?

Algorithm-comparison RQs (which optimizer to use) support methods justification only.

## Six Algorithms

| ID | Algorithm | Type | Objective |
|------|-----------|------|-----------|
| `rs` | Random Search | Null baseline | Scalar composite (1:1:1) |
| `hc` | Greedy Hill-Climbing | Local search (no escape) | Scalar composite (1:1:1) |
| `sa` | Simulated Annealing | Markov chain, single-trajectory | Scalar composite (1:1:1) |
| `sga` | Single-Objective GA | Population-based, scalar tournament | Scalar composite (1:1:1) |
| `nsga2` | NSGA-II | Population-based, Pareto | 3-objective: (1−JFI, \|I\|, LSAP) |
| `ts` | Tabu Search | Deterministic memory, full-neighbourhood | Scalar composite (1:1:1) |

> **Naming note:** The production benchmark pipeline (`submit_all.sh`,
> `benchmark_engine.py`) uses lowercase IDs: `rs`, `hc`, `sa`, `sga`, `nsga2`, `ts`.
> SGA was added to isolate the architecture vs objective-type comparison
> (SA vs SGA = same scalar, different architecture; SGA vs NSGA-II = same
> operators, different objective). The legacy G-prefix aliases are retired.

### Objective hierarchy (Nominal Scalarization)

**Primary evaluation** for the paper is **condition comparison** (spatial profile, JFI parity); see Design_Rationale.md. **Track B (HV, IGD+)** supports methods justification. We use **equal weights 1:1:1** for the full composite. Weight-sensitivity analysis tests perturbations around 1:1:1. HV and IGD+ (Ishibuchi et al., 2015) are computed against an **empirical reference front**.

**Cross-method IGD.** To validate that single-objective trajectories approximate the Pareto manifold (and to justify the production engine choice), scalar-algorithm terminal states (SA, TS, HC, SGA) are projected into the 3-objective space $(1 - J_{\min}, |I|, \text{LSAP})$ and their **Inverted Generational Distance Plus (IGD+)** to the per-budget empirical reference front is computed. This provides a geometric distance metric from each scalar terminal to the NSGA-II front at the same evaluation budget. Implementation: `scripts/cross_method_igd.py` (Phase 6b in the pipeline); output: `cross_method_igd.csv`.

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
- **N=100 seeds** (seeds 0–99) — primary benchmark protocol
- `iterations` / evaluation budget and `population_size` as set in the pipeline (e.g. `benchmark_engine.py`, `submit_all.sh`)

### 3. Analyze Results

The notebook automatically:
- Runs the benchmark over N=100 seeds per algorithm (primary protocol)
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
- "Spatial metrics (Moran's I, LSAP) add meaningful constraint beyond scalar fairness (JFI); four-condition ablation demonstrates this."
- "Maps from the full-composite condition differ detectably from JFI-only in spatial profile (raw objective vectors); JFI parity check supports interpretation."
- "Algorithms are instruments; SA is the chosen instrument for the main experiment; algorithm comparison is methods justification."
- "100 independent seeds, Wilcoxon paired by seed, pre-specified effect size (A ≥ 0.64), Holm–Bonferroni across 6 primary spatial tests."

### ❌ You CANNOT claim:
- ~~"AI agents played full games"~~
- ~~"This measures actual gameplay outcomes"~~
- ~~"Spatial clustering causes competitive disadvantage in human play"~~ (future work)

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
