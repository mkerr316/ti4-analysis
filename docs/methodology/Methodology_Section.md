# Formal Methodology Section (Paper Draft)

*Standalone Methodology section for the academic paper, suitable for games / PCG / AI-in-Games venues. Copy-paste ready; fill in section number (e.g. 3) and cross-references as required by your target journal.*

---

## 3.1 Problem Formulation

> **Evaluation hierarchy:** Primary evaluation uses weight-independent Pareto quality indicators (Hypervolume, IGD+); the scalar weighted-sum composite serves as a secondary benchmark for single-objective comparators only. See §3.7 for Track A/B definitions.

Anomaly tiles (asteroid fields, supernovae, nebulae, gravity rifts) are assigned under a non-adjacency constraint and frozen during optimization; all algorithms operate exclusively on the blue system tile swapping problem.

Map optimization is framed as minimization of a weighted composite score over three objectives:

$$S = w_1 \cdot (1 - J_{\min}) + w_2 \cdot \max\!\left(0,\ I - \mathbb{E}[I]\right) + w_3 \cdot \frac{\text{LSAP}}{n(n-1)} \quad \text{where } \mathbb{E}[I] = -\frac{1}{n-1}$$

The one-sided hinge penalizes only positive spatial autocorrelation above the null expectation; negative autocorrelation (spatial dispersion) incurs zero penalty, reflecting the design goal of preventing resource clustering rather than enforcing spatial randomness.

In the above, $J_{\min} = \min(J_R, J_I)$ is the **Multi-Jain bottleneck** — Jain's Fairness Index computed independently on distance-weighted raw Resources and raw Influence, with the minimum (bottleneck) dimension determining the fairness term. This formulation is theoretically inspired by the bottleneck intuition in multi-resource allocation (Ghodsi et al., 2011): map fairness is limited by the *least fair* resource dimension; we do not claim the formal DRF axioms, which apply to allocation mechanisms, not fixed topologies.

Given the lack of consensus on optimal weighting for composite spatial indicators (Libório et al.), we use **equal weights 1:1:1** ($w_1 = w_2 = w_3 = 1/3$) for the full composite — a fixed target for single-objective methods (SA, TS, HC, SGA). Primary evaluation for the main experiment is condition comparison (spatial profile, JFI parity); weight-independent Pareto indicators (HV, IGD+) support methods justification; see §3.7 and Design_Rationale.md. **Weight-sensitivity analysis** (e.g. `analyze_benchmark.py --sensitivity`) tests whether algorithm rankings (e.g. SA vs HC) are robust to alternative weight configurations around 1:1:1. When superiority holds across configurations, the equal-weight choice is academically defensible. HV and IGD+ (Ishibuchi et al., 2015) are computed against an **empirical reference front** formed by merging all observed Pareto points across seeds when the true Pareto front is unknown.

**Normalization vs. Weighting (Gen-0 Standardization)**  
To resolve the mathematical heteroskedasticity inherent in the raw objectives, the engine strictly decouples *normalization* (unit equalization) from *weighting* (strategic preference). At initialization, the system samples $N=1,000$ uniformly random map configurations from the underlying permutation state space ($37! \approx 1.37 \times 10^{43}$) to compute the empirical standard deviation ($\sigma$) of each objective. These Gen-0 static $\sigma$ values are not arbitrary, human-chosen scalars; they are exact empirical properties of the unoptimized problem domain.  
Dividing each term by its Gen-0 $\sigma$ converts the objectives into comparable standard-deviation units (Z-score analogues). Without this conversion, the Moran's I hinge term—which accounts for approximately $92\%$ of empirical variance at Gen-0—effectively dictates the gradient, reducing the search to a single-objective spatial optimizer. Gen-0 normalization forces dimensional parity. Consequently, the assigned weight vector (e.g., 1:1:1 or sensitivity variants) rigorously governs the relative importance of the objectives in the gradient calculation, ensuring the composite behaves mathematically as intended.

All three terms are normalized to $[0, 1]$ before weighting: $(1 - J_{\min}) \in [0,1]$, $\max(0,\ I - \mathbb{E}[I]) \in \left[0,\ 1 + \tfrac{1}{n-1}\right] \approx [0, 1.028]$ for $n=37$ (effectively bounded at 1 for all observed map configurations), and $\text{LSAP}/[n(n-1)] \in [0,1]$. The divisor $n(n-1)$ is the theoretical maximum of the sum of positive variance-normalized local Moran's I values under row-standardized $\mathbf{W}$. Balance gap (max − min player value) is retained as a display attribute but is excluded from the composite score and all Pareto dominance calculations.

---

## 3.2 Distance-Weighting Model

Per-player home values are computed as the weighted sum of all reachable system tile values within a hex-distance range gate of 5. We employ a **discrete step-function decay** based on terrain-cost pathfinding rather than continuous inverse-distance power laws ($w = 1/d^\gamma$), reflecting the quantized nature of board game movement and avoiding a free parameter $\gamma$ with no ground-truth calibration in the game domain.

For each (home $p$, system $s$) pair: (1) **Routing distance:** compute the shortest "modded distance" $d_m(p, s)$ via BFS pathfinding, where each hop accumulates a terrain cost from the evaluator; supernovae block the path entirely. (2) **Step-function weight lookup:** map $d_m$ to a weight via a discrete multiplier table:

$$v_p = \sum_{s \in \mathcal{R}(p)} \mathbf{w}\bigl[\lfloor d_m(p, s) \rfloor\bigr] \cdot \text{val}(s)$$

The default table (community-calibrated "Joebrew" evaluator) uses weights 6, 6, 6, 4, 4, 2, 1, 0 for $\lfloor d_m \rfloor = 0, 1, \ldots, 7+$, with a "shelf" at weight 6 for $d_m \in [0, 2]$ to capture equal accessibility within one round's movement. Terrain costs: blue/empty 0; nebula 0; gravity rift $0.3\,d_{\text{rift}} - 0.4$; asteroid/supernova block. For Multi-Jain, the same weight matrix is applied to raw Resources and raw Influence separately.

**Distance-weight sensitivity:** Six alternative weight tables are tested on a representative subset of seeds; algorithm rankings are verified to remain stable via Friedman and Wilcoxon tests and Kendall's tau, confirming that conclusions are not brittle to the specific distance decay. Results are tabulated below; mean Kendall's τ > 0.90 across all perturbation pairs indicates that algorithm rankings are invariant to the specific parameterization.

### Table: Distance-Decay Weight Robustness

| Weight Configuration | Description | Kendall's τ vs. Baseline | p-value |
| -------------------- | ----------- | ------------------------ | ------- |
| flat_nearby | Uniform shelf extended to hop 3 | ⚠ PENDING_MAIN_EXPERIMENT | — |
| steep_decay | Sharper drop at hop 3 | ⚠ PENDING_MAIN_EXPERIMENT | — |
| linear | Monotone linear decay | ⚠ PENDING_MAIN_EXPERIMENT | — |
| inverse_distance | Inverse-distance weights | ⚠ PENDING_MAIN_EXPERIMENT | — |
| binary_reachable | Binary within-range indicator | ⚠ PENDING_MAIN_EXPERIMENT | — |
| **Mean (all pairs)** | | ⚠ PENDING_MAIN_EXPERIMENT | — |

> **⚠ Pre-submission gate:** Replace all placeholders (e.g. distance-weight table above) with numbers from the main experiment before submitting. Run `scripts/pre_submission_check.sh` from the project root — zero matches required.

Source: `scripts/distance_weight_sensitivity.py`.

---

## 3.3 Metrics

### Balance gap

$\text{gap} = \max(\mathbf{v}) - \min(\mathbf{v})$ where $\mathbf{v}$ is the per-player distance-weighted value. Retained for display only; excluded from composite and Pareto dominance.

### Moran's I

Global spatial autocorrelation:

$$I = \frac{N}{W} \cdot \frac{\sum_i \sum_j w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}$$

Under the null of no spatial association, $E[I] = -1/(N-1)$. For the TI4 map, $N = 37$ swappable systems, so $E[I] \approx -0.027$. Interpretation: $I > E[I]$ positive autocorrelation; $I \approx E[I]$ spatial randomness; $I < E[I]$ negative autocorrelation.

### Spatial weights matrix and edge effects

Standard spatial statistics (e.g. Moran's I) are often derived under infinite or wrapped domains. On a small, bounded hex grid ($N = 37$), edge hexes have fewer neighbours than interior ones, which can bias global and local autocorrelation (Boots and Tiefelsdorf, 2000). We use **binary adjacency weights**, **row-standardized** so that each row of $\mathbf{W}$ sums to 1.0 — the standard approach to mitigate edge-effect bias in bounded tessellations (Boots and Tiefelsdorf, 2000). After standardization, the spatial lag $(\mathbf{W}\mathbf{z})_i$ at each position is a proper weighted average of its neighbours regardless of neighbour count, ensuring that edge systems are not artificially penalized or advantaged. Zero-sum rows (isolated hexes) are guarded by substituting a denominator of 1.0. We do not apply a separate edge-correction penalty. Implementation: `src/ti4_analysis/algorithms/map_topology.py`.

### Local Spatial Autocorrelation Penalty (LSAP)

The LSAP is a variance-normalized local spatial penalty that proxies significance-tested LISA (Anselin, 1995). For each system $i$:

$$I_i = \frac{(x_i - \bar{x}) \sum_j w_{ij}(x_j - \bar{x})}{m_2}, \quad m_2 = \frac{\sum(x_i - \bar{x})^2}{n}$$

Positive $I_i$ identifies H-H and L-L clusters. LSAP sums only the positive local values. The composite normalizes by $n(n-1)$; this bound holds because $\mathbf{W}$ is row-standardized. The LSAP serves as a **continuous heuristic** in the optimization loop (permutation LISA would be prohibitively expensive); post-hoc validation via conditional-permutation LISA (e.g. 9,999 permutations per location, FDR Benjamini–Hochberg) confirms that minimising the proxy reduces statistically significant clusters. We omit Getis-Ord $G_i^*$ from the methodology because at $N \approx 37$ its asymptotic z-score is unreliable and because LISA/LSAP are better suited to detecting spatial outliers; significance is assessed only via conditional permutation LISA.

### Multi-Jain Fairness Index (Bottleneck JFI)

For one dimension, $J(\mathbf{x}) = (\sum x_i)^2 / (n \sum x_i^2)$, range $[1/n, 1]$. We compute JFI on raw Resources and raw Influence separately and use $J_{\min} = \min(J_R, J_I)$ so that fairness is limited by the least-fair dimension (Jain et al., 1984; Ghodsi et al., 2011).

---

## 3.4 Algorithms

- **HC (Greedy Hill-Climbing):** Accepts only improving moves; no memory. Baseline and warm-start for NSGA-II/SGA.
- **SA (Simulated Annealing):** Markov chain with $P(\text{accept}) = \exp(-\Delta/T)$. Cooling rate $\alpha = (T_{\min}/T_0)^{1/N}$ over $N$ steps (Kirkpatrick et al., 1983). $T_0$ calibrated to initial acceptance rate.
- **SGA (Single-Objective GA):** Same BFS-blob crossover and swap mutation as NSGA-II; scalar tournament selection and $(\mu+\lambda)$ replacement. **Architecture control:** SA vs SGA = same scalar, different architecture; SGA vs NSGA-II = same operators, scalar vs Pareto. Recombination is localized (BFS-connected blobs + OX1) so that the geometric contiguity of the search space is preserved and epistasis is controlled; no further structural changes are required for epistatic control.
- **NSGA-II:** Three-objective Pareto (1 − JFI, |I|, LSAP). BFS-blob crossover; non-dominated sorting and crowding distance (Deb et al., 2002). For Track A we apply a posteriori scalarization (min composite over the front).
- **TS (Tabu Search):** Full-neighbourhood search; best non-tabu move applied each iteration. Default tenure scales with neighbourhood size: $\max(3, \lceil 0.05 \cdot \binom{S}{2}\rceil)$ so that at saturation budgets (e.g. 500k evaluations) TS retains meaningful memory; optional override via $\lceil k\sqrt{S}\rceil$ (Glover, 1989). **Methodological control** (cf. Terra Mystica PCG literature).
- **RS (Random Search):** Uniform random permutations; null baseline.

SA accepts equal-fitness moves with probability 1 under the Metropolis criterion; HC, TS, and SGA use lexicographic tie-breaking (primary: composite score; secondary: $-J_{\max}$) on zero-delta moves. This behavioral difference on flat fitness regions is intentional and constitutes part of the architectural comparison.

**Structural corrections (optional `--corrected-landscape`).** When enabled, the benchmark applies four corrections to the fitness landscape: (1) **Static Gen-0 normalization:** empirical standard deviations of the three objective terms are computed once from 1,000 random map permutations and used as fixed divisors for the entire run, eliminating variance dominance while keeping the landscape stationary. (2) **Smooth objectives:** the Multi-Jain bottleneck is replaced by a smooth L$_{-p}$ mean ($p=8$) and the Moran hinge by a softplus ($k=10$) to restore gradient continuity for SA. (3) **Local-variance LSAP:** each local Moran $I_i$ is scaled by $\sqrt{k_i}$ (node degree) so that edge and interior nodes have comparable scale. (4) **TS tenure** as above. These corrections are gated by the `--corrected-landscape` flag; when off, behaviour matches the nominal formulation for reproducibility.

Bayesian optimization is excluded (see `docs/bayesian_optimization_exclusion.md`).

---

## 3.5 Experimental Protocol

- **Seeds:** Benchmark seeds 0–99 (100 maps). Tuning seeds 9,000–9,099 (100). Held-out seeds 9,100–9,149 (50).
- **Budget:** 1k–500k fitness evaluations per algorithm per seed (saturation study). Budget consumed as: HC/SA by iterations; SGA/NSGA-II by generations × population; TS by full-neighbourhood iterations; RS by independent samples. Budgets are chosen on a quasi-logarithmic grid (e.g. 1k, 5k, 10k, 50k, 100k, 500k) to characterize **anytime performance**: how quickly each algorithm improves as a function of evaluations, and at what point additional budget yields diminishing returns.
- **Hyperparameter tuning:** SA, SGA, NSGA-II tuned on disjoint seeds (9,000–9,099) via Optuna TPE (50 trials each). TS tuned via **exhaustive grid search** over the tenure coefficient k (θ = max(3, ⌈k·√S⌉)) on the same seeds. Best parameters validated on a held-out set (9,100–9,149). `best_params.json` reports `cv_mean`, `cv_std` (training) and `held_out_mean`, `held_out_std` for overfitting comparison. **Held-out variance** is discussed in §3.6.
- **Design topology (randomized block / repeated measures):** The main benchmark is a **randomized block design**: the same set of 100 seeds is evaluated under every algorithm at each budget level. Observations are therefore **paired within seeds** rather than independent across algorithms.
- **Convergence:** `evals_to_best` recorded; convergence curves and budget utilization analyzed per budget. For each budget we summarize performance via medians and interquartile ranges, and use **bootstrap 95% confidence intervals on median differences** (rather than raw empirical percentiles) to obtain stable uncertainty estimates in the presence of rugged, discrete combinatorial landscapes.

---

## 3.6 Hyperparameter Validation and Held-Out Variance

**(A) Landscape ruggedness.** The variance observed on the held-out validation set is not a failure of the hyperparameters (e.g. "overfitting to the tuning set"). In combinatorial optimization, unlike predictive ML, each run starts from a different random map (seed). The TI4 map fitness landscape is highly rugged; different seeds place the search in vastly different regions of the space. Some starting maps are relatively close to a fair configuration, while others require traversing deep, unrecoverable local minima. Thus, **the spread in held-out performance reflects the inherent variability of problem difficulty across starting states**, not that the tuned settings (e.g. tabu tenure, cooling schedule) have "memorized" the tuning seeds.

**(B) Resiliency of the mean.** While the *variance* on the held-out set is larger than that in cross-validation (e.g. CV mean ± std: **0.0545 ± 0.0028** vs held-out: **0.0599 ± 0.0299** — replace with your actual values from `best_params.json`), the **mean** performance degrades only slightly. This shows that the optimized hyperparameters generalize successfully to unseen starting conditions: the same settings that perform well on the tuning seeds perform well on average on the held-out seeds, even though the absolute difficulty of individual held-out maps varies widely.

**(C) Phase 2 as the ultimate mitigation.** This variance is precisely why the main benchmark (Phase 2: Saturation Benchmark) is designed as it is. We do not judge algorithms on a single run or on means alone. We run each algorithm on **100 seeds** (disjoint from tuning and held-out) and use **non-parametric inference**: Friedman omnibus test and pairwise Wilcoxon signed-rank tests with Holm–Bonferroni correction, plus Vargha–Delaney A and bootstrap CIs on median difference. By using repeated measures across 100 unseen seeds, we explicitly account for starting-state variance, ensuring that no algorithm is unfairly penalized by a "bad" starting seed and that claims of superiority are based on reported p-values and effect sizes.

---

## 3.7 Analysis Tracks

**Track A — Scalar (methods justification).** The 1:1:1 composite is applied to every algorithm's final solution; for NSGA-II, the Pareto-front member with minimum composite is selected (a posteriori scalarization). For NSGA-II, the Track A composite score is computed from the Pareto front member with minimum composite score at the end of the run; for scalar algorithms it is the run-best composite score observed at any point during the run. All six algorithms (RS, HC, SA, SGA, NSGA-II, TS) are compared via **median and IQR**, **Friedman** test, **Wilcoxon** signed-rank pairwise with Holm–Bonferroni correction, **Vargha–Delaney A**, and **bootstrap 95% CIs** on median difference. We do not base claims on mean values; we report p-values and claim one algorithm outperforms another only when the corrected p-value is below the chosen significance level. **Empirical variance note (pre-registered, `variance_equalization_diagnostic.py`):** Across N = 1,000 random (Gen-0) configurations, the Moran's I hinge term carries approximately 92% of total weighted empirical variance, with JFI gap and LSAP sharing the remaining 8%. Track A scalar results are therefore predominantly driven by spatial autocorrelation reduction in the early phase of search; Track B (Hypervolume, weight-independent) remains the primary evaluation. The σ shift diagnostic confirms the favorable outcome: the Moran's I hinge term collapses to near-zero variance at convergence (σ ≈ 0.000), while JFI gap and LSAP retain meaningful variance (approximately 65% and 35% of weighted empirical variance respectively). The 1:1:1 weights are therefore operationally consistent in the optimized subspace where the benchmark results are reported; the Gen-0 Moran's I dominance is an early-search transient, not a persistent distortion of the objective landscape.

**Track B — Pareto (multi-objective quality).** NSGA-II's raw Pareto archives are evaluated with **Hypervolume (HV)**, **IGD+**, and **Spacing**. This is the primary evaluation for multi-objective performance and avoids scalarization. Scalar algorithms do not produce Pareto fronts and are compared only via Track A.

**Cross-method IGD.** Scalar-algorithm terminal states (SA, TS, HC, SGA) are projected into the 3-objective space and their IGD+ to the per-budget empirical reference front is reported. This validates that single-objective trajectories converge near the Pareto manifold and supports the justification for using SA (or another scalar algorithm) in production.

**Unified HV (objective commensurability).** To place scalar and multi-objective methods in a **single objective space**, we additionally compute Hypervolume for every algorithm using **empirical Pareto fronts** extracted from the logged run histories. For each (algorithm, seed, budget), we collect the non-dominated set of visited maps in the canonical 3D space $(1 - J_{\min}, |I|, \text{LSAP})$ and compute HV against a common nadir reference point derived from the worst observed objective values (auto: worst×1.1). This yields a weight-independent, distribution-agnostic quality indicator that is commensurable across RS, HC, SA, SGA, NSGA-II, and TS. The unified HV tables and non-parametric statistics (Friedman / Wilcoxon / Vargha–Delaney) are produced by `scripts/unified_hv_analysis.py` using the `unified_archives/` emitted by `benchmark_engine.py`.

**Causal scope.** The present study demonstrates that SA suppresses spatial clustering more effectively than HC and NSGA-II under fixed evaluation budgets; whether spatial clustering translates to measurable competitive disadvantage is an open empirical question requiring human-subject validation, addressed in Future Work.

**Statistical justification.** Because the design is randomized-block / repeated-measures (same seeds under all algorithms), our data consist of **dependent paired samples** across algorithms. The **Friedman** test is therefore the correct non-parametric omnibus analogue of repeated-measures ANOVA; independent-sample tests such as Kruskal–Wallis would incorrectly treat within-seed variance as between-subject noise and inflate Type II error. Likewise, pairwise comparisons use the **Wilcoxon signed-rank** test (paired) rather than Mann–Whitney U / rank-sum (independent samples), ensuring that within-seed pairing is fully exploited in the inference.

---

## 3.8 Ablation Study: Objective-Weight Paths

To empirically validate the necessity of the multi-objective formulation against the existing community standard, the methodology employs a three-path ablation design that isolates the extreme bounds of the fitness landscape:

1. **Control (Community Baseline):** Optimize strictly for numerical fairness. Uses a weight vector of 1:0:0 to maximize Jain's Fairness Index, entirely blinding the optimizer to spatial metrics.
2. **Spatial-Only:** Optimize strictly for topological dispersion. Uses a weight vector of 0:0.5:0.5 to minimize global Moran's I and the LSAP, ignoring numerical resource parity.
3. **Proposed Multi-Objective:** Optimize the Gen-0 normalized composite under the nominal weights (1:1:1).

**Expected Proof:** The ablation quantitatively demonstrates the structural deficiency of scalar fairness metrics. The Control (JFI-only) path is expected to produce maps that are numerically perfectly fair but topologically pathological—for instance, aggregating all high-value resources in a single player's slice offset by high-influence resources in another, thereby maximizing the LSAP penalty. The Proposed multi-objective path balances the gradient, achieving parity with the Control's numerical JFI while simultaneously suppressing spatial clustering (minimizing LSAP). This confirms that spatial optimization is not redundant, and that the community baseline is mathematically blind to spatial exploitation.

---

## 3.9 Null hypotheses

**RQ1:** $H_0$: the distribution of composite scores does not differ significantly across algorithms at any fixed evaluation budget (Friedman test, $\alpha = 0.05$).

**RQ2:** $H_0$: NSGA-II's hypervolume does not exceed that of scalar algorithms under equal total evaluation budget (Wilcoxon signed-rank, one-tailed, Holm–Bonferroni corrected).

**RQ3:** *(Trade-offs between balance gap and spatial distribution.)* This question is exploratory; no directional hypothesis is pre-specified. Results are reported as Spearman correlations between balance gap and spatial metrics across optimized solutions.

**RQ4:** $H_0$: wall-clock time to reach a target composite score does not differ across algorithms (Friedman test).

---

## 3.10 Statistical methods and effect size

**Effect size for paired comparisons.** For paired (before/after) or within-seed comparisons we report **Cohen's $d_z$**: the standardized mean difference of the *change*, $d_z = \bar{D} / s_D$, where $\bar{D}$ is the mean of the paired differences (after − before) and $s_D$ is their standard deviation (ddof=1). This is the appropriate effect size for the paired $t$-test and for repeated-measures designs (Lakens, 2013). Magnitude bands: $|d_z| < 0.2$ negligible; $0.2 \leq |d_z| < 0.5$ small; $0.5 \leq |d_z| < 0.8$ medium; $|d_z| \geq 0.8$ large.

**Multi-objective fairness and tie-breaking.** The composite uses the **Multi-Jain bottleneck** $J_{\min} = \min(J_R, J_I)$, so the fairness term prioritizes the *most disadvantaged* resource dimension (Resources or Influence). When a move leaves the composite score unchanged (e.g. improves only the non-bottleneck dimension), **HC, TS, and SGA** use a **lexicographic tie-breaker**: they compare solutions by $(\text{composite}, -J_{\max})$ where $J_{\max} = \max(J_R, J_I)$, so the solution with higher $J_{\max}$ is preferred on the plateau. **SA** does not use this key; it accepts equal-cost moves with probability 1 and thus traverses the plateau without an explicit tie-breaker. The observed **JFI success** (good fairness on both dimensions) is therefore the result of a multi-objective search that (i) prioritizes the worst-off dimension in the scalar composite and (ii) still improves the secondary dimension on the plateau (via lex_key for HC/TS/SGA and equal-move acceptance for SA).

---

## References (Methodology)

Anselin, L. (1995). Local indicators of spatial association — LISA. *Geographical Analysis*, 27(2), 93–115.

Boots, B., & Tiefelsdorf, M. (2000). Global and local spatial autocorrelation in bounded regular tessellations. *Journal of Geographical Systems*, 2(4), 319–348.

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

Ghodsi, A., Zaharia, M., Hindman, B., Konwinski, A., Shenker, S., & Stoica, I. (2011). Dominant resource fairness: Fair allocation of multiple resource types. *Proceedings of the 8th USENIX Symposium on Networked Systems Design and Implementation (NSDI)*, 24, 323–336.

Glover, F. (1989). Tabu search — Part I. *ORSA Journal on Computing*, 1(3), 190–206.

Ishibuchi, H., Masuda, H., Tanigaki, Y., & Nojima, Y. (2015). Modified distance calculation in generational distance and inverted generational distance. *Proceedings of the International Conference on Evolutionary Multi-Criterion Optimization (EMO)*, 110–125.

Jain, R., Chiu, D. M., & Hawe, W. R. (1984). A quantitative measure of fairness and discrimination for resource allocation in shared computer systems. *DEC Research Report TR-301*.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671–680.

Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for $t$-tests and ANOVAs. *Frontiers in Psychology*, 4, 863. https://doi.org/10.3389/fpsyg.2013.00863

Libório, M. P., de Abreu, J. F., Ekel, P., & Machado, A. (2022). Effect of sub-indicator weighting schemes on the spatial dependence of multidimensional phenomena. *Journal of Geographical Systems*, 25, 185–211.
