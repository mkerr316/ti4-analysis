# Response to Reviewers — Template

Pre-written rebuttal paragraphs for common critiques. Fill in specific numbers, section references, or citations as needed for your submission.

---

## 1. Single- vs multi-objective comparison (NSGA-II vs TS/SA)

**Reviewer concern:** You are comparing single-objective methods (SA, TS, HC) against a multi-objective method (NSGA-II). Single-objective methods use a weighted scalar; NSGA-II produces a Pareto front. How exactly are you comparing the "best" NSGA-II map to the "best" TS map? If you squash NSGA-II's output to a scalar, you negate the point of NSGA-II.

**Response:** We use two distinct evaluation tracks. **Track A** (production-oriented, scalar): we apply the *same* 5:5:3 composite scalarization to every algorithm's final solution. For NSGA-II, we apply this scalar to every member of the final Pareto front and select the member with the *lowest* composite score (a posteriori scalarization — standard practice when comparing MOEAs to single-objective algorithms). This allows an apples-to-apples scalar comparison for production algorithm selection. **Track B** (primary for NSGA-II): we evaluate NSGA-II's *raw* Pareto archives using Hypervolume, IGD+, and Spacing — weight-independent quality indicators that do not require scalarization. Track B is the gold standard for multi-objective performance in the manuscript. We do not run "two NSGAs"; we run NSGA-II once as a three-objective MOEA and report it both via Track A (for scalar comparison) and Track B (for Pareto quality). To isolate algorithmic architecture from objective type, we include a single-objective genetic algorithm (SGA) that uses the same crossover/mutation as NSGA-II but optimizes the scalar composite directly; thus SA vs SGA tests architecture, and SGA vs NSGA-II tests scalar vs multi-objective. See Methodology §3.1 and §3.7.

---

## 2. Bounded hexagonal edge effects (Moran's I)

**Reviewer concern:** You rely on Moran's I for spatial autocorrelation, but a standard TI4 map has only 37 hexes with hard boundaries. Standard spatial statistics assume infinite planes or wrapped borders. On a small, bounded grid, edge hexes have fewer neighbors, which heavily biases global spatial statistics. How does your spatial weights matrix (W) handle edge effects? Row-standardized weights? Edge-correction penalty?

**Response:** We use a **row-standardized** spatial weights matrix $\mathbf{W}$: binary adjacency with each row normalized so that it sums to 1.0 (implementation: `src/ti4_analysis/algorithms/map_topology.py`). This is the standard approach to mitigate edge-effect bias in bounded regular tessellations (Boots and Tiefelsdorf, 2000). Standard spatial statistics do assume infinite or wrapped domains; on our small bounded hex grid ($N = 37$), edge hexes have fewer neighbours (typically 2–3) than interior ones (up to 6), which would otherwise systematically deflate their spatial lag values. After row-standardization, the spatial lag $(\mathbf{W}\mathbf{z})_i$ at each position is a proper *weighted average* of its neighbours regardless of neighbour count, so edge systems are not artificially penalized or advantaged in Moran's I and LISA computations. We do **not** apply a separate edge-correction penalty; the correction strategy is row-standardization itself. We cite Boots and Tiefelsdorf (2000) in the methodology. See Methodology §3.3 (Spatial weights matrix and edge effects).

---

## 3. Statistical significance of the algorithm comparison

**Reviewer concern:** Phase 2 reports average scores at different budgets (e.g. 1k, 50k, 500k). If Tabu Search averages 0.054 and Simulated Annealing 0.056, you cannot simply say "Tabu Search wins." Optimization landscapes are non-parametric; you must use non-parametric tests and report p-values to claim one algorithm mathematically outperforms another.

**Response:** We do not rely on mean scores for claims of superiority. At each evaluation budget we run: (1) **Friedman** omnibus test on the 100-seed distribution of composite score; (2) **Wilcoxon signed-rank** pairwise tests for all algorithm pairs, with **Holm–Bonferroni** correction for multiple comparisons; (3) **Vargha–Delaney A** effect sizes; (4) **bootstrap 95% CIs** on the median difference. We report median and IQR (not mean) as the primary descriptive statistics. We claim that one algorithm outperforms another only when the *corrected* p-value is below our significance level (e.g. 0.05). This is implemented in `scripts/analyze_benchmark.py` (Phase 3). For a given budget (e.g. 500k), we run `analyze_benchmark.py results.csv --budget 500000` and report the Friedman p-value and pairwise Wilcoxon p-values from the generated `stats/full_report.txt`. See Methodology §3.7 (Track A) and Experimental Protocol.

---

## 4. Held-out validation variance (CV ±0.0028 vs held-out ±0.0299)

**Reviewer concern:** Your cross-validation variance is tiny (±0.0028) but your held-out test set variance is much larger (±0.0299). In traditional machine learning, a reviewer would say you overfit your hyperparameters to the training data. How do you explain this?

**Response:** We are not doing predictive ML; we are doing **combinatorial optimization**. The narrative is different:

**(A) Landscape ruggedness.** The higher variance on the held-out set is not a failure of the hyperparameters. It reflects the **rugged fitness landscape** of TI4 map generation. Different random seeds (starting map configurations) place the algorithms in vastly different regions of the search space. Some starting maps are relatively close to a fair configuration; others require traversing deep, unrecoverable local minima. So the spread in held-out performance is a reflection of the *varying difficulty of the problem instances*, not overfitting of hyperparameters to the tuning seeds.

**(B) Resiliency of the mean.** Although the variance increased, the **mean** performance barely degraded (e.g. CV mean 0.0545 → held-out mean 0.0599 — insert your actual values from `best_params.json`). This shows that the optimized hyperparameters (e.g. tabu tenure, cooling schedule) **generalize successfully** to unseen starting conditions; the absolute difficulty of individual maps varies wildly, but on average the same settings remain effective.

**(C) Phase 2 as the ultimate mitigation.** This variance is exactly why Phase 2 (the Saturation Benchmark) is designed as it is. We do not judge algorithms on a single run or on tuning/held-out sets for the main comparison. We run each algorithm on **100 seeds** (disjoint from tuning and held-out) and use **non-parametric statistics** (Friedman, Wilcoxon signed-rank with Holm–Bonferroni) on the full distribution. By using repeated measures across 100 unseen seeds, we explicitly account for this starting-state variance, so no algorithm is unfairly penalized by a "bad" starting seed. See Methodology §3.6 (Hyperparameter validation and held-out variance).

---

## 5. Nominal weight choice (5:5:3)

**Reviewer concern:** Why 5:5:3? Isn't the choice of weights arbitrary?

**Response:** Given the lack of consensus on optimal weighting for composite spatial indicators (Libório et al.), we use 5:5:3 as a **nominal scalarization** — a fixed target for single-objective methods. Its defensibility rests on **weight-sensitivity analysis**: we run the same benchmark under alternative weight configurations (equal weights, JFI-dominant, spatial-dominant, LISA-dominant) and verify that algorithm rankings (e.g. SA vs HC) remain robust. When superiority holds across these configurations, the 5:5:3 choice serves as an academically defensible nominal anchor. Primary evaluation for NSGA-II uses weight-independent Pareto indicators (HV, IGD+, Spacing) in Track B. See Methodology §3.1 and `analyze_benchmark.py --sensitivity`.

---

## 6. LSAP as a proxy for LISA (continuous heuristic)

**Reviewer concern:** You use a continuous "LSAP" instead of significance-tested LISA. How do you justify this?

**Response:** Significance-tested LISA requires conditional permutation tests (e.g. 9,999 permutations per location), which would multiply the cost of each fitness evaluation by roughly 1,000× and make optimization infeasible. The LSAP is a **continuous heuristic**: it sums all positive variance-normalized local Moran values, providing a smooth fitness signal for the optimizer. We distinguish it explicitly as "LSAP" (Local Spatial Autocorrelation Penalty), not "LISA." We do **post-hoc validation**: for a subset of seeds, we run full conditional-permutation LISA on each algorithm's final map and report correlations between the continuous proxy and the count of significant H-H/L-L clusters (FDR-corrected). If minimizing the proxy consistently reduces significant clusters, the proxy is empirically validated. See Methodology §3.3 and `scripts/validate_lisa_proxy.py`.

---

## 7. Small-N spatial statistics (N = 37)

**Reviewer concern:** With only 37 swappable tiles, can you really claim statistical significance for Moran's I or LISA?

**Response:** We do not claim significance for global Moran's I or LISA using analytical variance or z-scores at N = 37; asymptotic normality is not justified. For **LISA**, significance is evaluated via **conditional permutation tests** (e.g. 9,999 permutations per location) in our validation script; we use FDR (Benjamini–Hochberg) for multiple-testing correction. For the **optimization objective**, we use the continuous LSAP only as a heuristic; the final maps are evaluated with permutation-based LISA in post-hoc validation. For **algorithm comparison** (Phase 2), we use non-parametric tests (Friedman, Wilcoxon) on the 100-seed distribution of composite score, which do not assume normality. See `docs/limitations/limitations.md` and Methodology §3.3.

---

## Data sources for filling in numbers

- **Held-out / CV means and stds:** After running `scripts/optimize_hyperparameters.py`, read the Optuna run directory (e.g. `output/optuna_YYYYMMDD_HHMMSS/best_params.json`). Fields: `cv_mean`, `cv_std`, `held_out_mean`, `held_out_std`. Use these in the Methodology §3.6 and in Response #4.
- **Friedman / Wilcoxon p-values:** Run `scripts/analyze_benchmark.py <results.csv> --budget <N>`; read `stats/full_report.txt` and optionally `stats/pairwise_tests.csv`.
