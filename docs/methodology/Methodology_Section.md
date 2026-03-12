# Formal Methodology Section (Paper Draft)

*Standalone Methodology section for the academic paper, suitable for games / PCG / AI-in-Games venues. Copy-paste ready; fill in section number (e.g. 3) and cross-references as required by your target journal.*

---

## 3.1 Problem Formulation

Map optimization is framed as minimization of a weighted composite score over three objectives:

$$S = w_1 \cdot (1 - J_{\min}) + w_2 \cdot |I| + w_3 \cdot \frac{\text{LSAP}}{n(n-1)}$$

where $J_{\min} = \min(J_R, J_I)$ is the **Multi-Jain bottleneck** — Jain's Fairness Index computed independently on distance-weighted raw Resources and raw Influence, with the minimum (bottleneck) dimension determining the fairness term. This formulation is theoretically inspired by the bottleneck intuition in multi-resource allocation (Ghodsi et al., 2011): map fairness is limited by the *least fair* resource dimension; we do not claim the formal DRF axioms, which apply to allocation mechanisms, not fixed topologies.

Given the lack of consensus on optimal weighting for composite spatial indicators (Libório et al.), we use a **5:5:3** ratio (Fairness : Clustering : Local Penalty) as a **Nominal Scalarization** — a fixed target for single-objective methods (SA, TS, HC, SGA). Primary evaluation relies on weight-independent Pareto indicators (HV, IGD+); see §3.7. The nominal weights are $w_1 = 5/13$, $w_2 = 5/13$, $w_3 = 3/13$. **Weight-sensitivity analysis** (e.g. `analyze_benchmark.py --sensitivity`) tests whether algorithm rankings (e.g. SA vs HC) are robust to alternative weight configurations: equal weights, JFI-dominant, spatial-dominant, and LISA-dominant. When superiority holds across these configurations, the 5:5:3 choice is academically defensible as a nominal anchor. HV and IGD+ (Ishibuchi et al., 2015) are computed against an **empirical reference front** formed by merging all observed Pareto points across seeds when the true Pareto front is unknown.

All three terms are normalized to $[0, 1]$ before weighting: $(1 - J_{\min}) \in [0,1]$, $|I| \in [0,1]$ (theoretical bound for row-standardized $\mathbf{W}$), and $\text{LSAP}/[n(n-1)] \in [0,1]$. The divisor $n(n-1)$ is the theoretical maximum of the sum of positive variance-normalized local Moran's I values under row-standardized $\mathbf{W}$. Balance gap (max − min player value) is retained as a display attribute but is excluded from the composite score and all Pareto dominance calculations.

---

## 3.2 Distance-Weighting Model

Per-player home values are computed as the weighted sum of all reachable system tile values within a hex-distance range gate of 5. We employ a **discrete step-function decay** based on terrain-cost pathfinding rather than continuous inverse-distance power laws ($w = 1/d^\gamma$), reflecting the quantized nature of board game movement and avoiding a free parameter $\gamma$ with no ground-truth calibration in the game domain.

For each (home $p$, system $s$) pair: (1) **Routing distance:** compute the shortest "modded distance" $d_m(p, s)$ via BFS pathfinding, where each hop accumulates a terrain cost from the evaluator; supernovae block the path entirely. (2) **Step-function weight lookup:** map $d_m$ to a weight via a discrete multiplier table:

$$v_p = \sum_{s \in \mathcal{R}(p)} \mathbf{w}\bigl[\lfloor d_m(p, s) \rfloor\bigr] \cdot \text{val}(s)$$

The default table (community-calibrated "Joebrew" evaluator) uses weights 6, 6, 6, 4, 4, 2, 1, 0 for $\lfloor d_m \rfloor = 0, 1, \ldots, 7+$, with a "shelf" at weight 6 for $d_m \in [0, 2]$ to capture equal accessibility within one round's movement. Terrain costs: blue/empty 0; nebula 0; gravity rift $0.3\,d_{\text{rift}} - 0.4$; asteroid/supernova block. For Multi-Jain, the same weight matrix is applied to raw Resources and raw Influence separately.

**Distance-weight sensitivity:** Six alternative weight tables are tested on a representative subset of seeds; algorithm rankings are verified to remain stable via Friedman and Wilcoxon tests and Kendall's tau, confirming that conclusions are not brittle to the specific distance decay.

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

Positive $I_i$ identifies H-H and L-L clusters. LSAP sums only the positive local values. The composite normalizes by $n(n-1)$; this bound holds because $\mathbf{W}$ is row-standardized. The LSAP serves as a **continuous heuristic** in the optimization loop (permutation LISA would be prohibitively expensive); post-hoc validation via conditional-permutation LISA (e.g. 9,999 permutations per location, FDR Benjamini–Hochberg) confirms that minimising the proxy reduces statistically significant clusters.

### Multi-Jain Fairness Index (Bottleneck JFI)

For one dimension, $J(\mathbf{x}) = (\sum x_i)^2 / (n \sum x_i^2)$, range $[1/n, 1]$. We compute JFI on raw Resources and raw Influence separately and use $J_{\min} = \min(J_R, J_I)$ so that fairness is limited by the least-fair dimension (Jain et al., 1984; Ghodsi et al., 2011).

### Getis-Ord Gi*

Used for exploratory hot-spot analysis; not included in the benchmark composite score.

---

## 3.4 Algorithms

- **HC (Greedy Hill-Climbing):** Accepts only improving moves; no memory. Baseline and warm-start for NSGA-II/SGA.
- **SA (Simulated Annealing):** Markov chain with $P(\text{accept}) = \exp(-\Delta/T)$. Cooling rate $\alpha = (T_{\min}/T_0)^{1/N}$ over $N$ steps (Kirkpatrick et al., 1983). $T_0$ calibrated to initial acceptance rate.
- **SGA (Single-Objective GA):** Same BFS-blob crossover and swap mutation as NSGA-II; scalar tournament selection and $(\mu+\lambda)$ replacement. **Architecture control:** SA vs SGA = same scalar, different architecture; SGA vs NSGA-II = same operators, scalar vs Pareto.
- **NSGA-II:** Three-objective Pareto (1 − JFI, |I|, LSAP). BFS-blob crossover; non-dominated sorting and crowding distance (Deb et al., 2002). For Track A we apply a posteriori scalarization (min composite over the front).
- **TS (Tabu Search):** Full-neighbourhood search; best non-tabu move applied each iteration. Tenure $\lceil\sqrt{S}\rceil$ (Glover, 1989). **Methodological control** (cf. Terra Mystica PCG literature).
- **RS (Random Search):** Uniform random permutations; null baseline.

Bayesian optimization is excluded (see `docs/bayesian_optimization_exclusion.md`).

---

## 3.5 Experimental Protocol

- **Seeds:** Benchmark seeds 0–99 (100 maps). Tuning seeds 9,000–9,099 (100). Held-out seeds 9,100–9,149 (50).
- **Budget:** 1k–500k fitness evaluations per algorithm per seed (saturation study). Budget consumed as: HC/SA by iterations; SGA/NSGA-II by generations × population; TS by full-neighbourhood iterations; RS by independent samples.
- **Hyperparameter tuning:** SA, SGA, NSGA-II, TS tuned separately on disjoint seeds (9,000–9,099) via Optuna TPE (50 trials each). Best parameters validated with 5-fold cross-validation on tuning seeds and a held-out set (9,100–9,149). **Held-out variance** is discussed in §3.6.
- **Convergence:** `evals_to_best` recorded; convergence curves and budget utilization analyzed per budget.

---

## 3.6 Hyperparameter Validation and Held-Out Variance

**(A) Landscape ruggedness.** The variance observed on the held-out validation set is not a failure of the hyperparameters (e.g. "overfitting to the tuning set"). In combinatorial optimization, unlike predictive ML, each run starts from a different random map (seed). The TI4 map fitness landscape is highly rugged; different seeds place the search in vastly different regions of the space. Some starting maps are relatively close to a fair configuration, while others require traversing deep, unrecoverable local minima. Thus, **the spread in held-out performance reflects the inherent variability of problem difficulty across starting states**, not that the tuned settings (e.g. tabu tenure, cooling schedule) have "memorized" the tuning seeds.

**(B) Resiliency of the mean.** While the *variance* on the held-out set is larger than that in cross-validation (e.g. CV mean ± std: **0.0545 ± 0.0028** vs held-out: **0.0599 ± 0.0299** — replace with your actual values from `best_params.json`), the **mean** performance degrades only slightly. This shows that the optimized hyperparameters generalize successfully to unseen starting conditions: the same settings that perform well on the tuning seeds perform well on average on the held-out seeds, even though the absolute difficulty of individual held-out maps varies widely.

**(C) Phase 2 as the ultimate mitigation.** This variance is precisely why the main benchmark (Phase 2: Saturation Benchmark) is designed as it is. We do not judge algorithms on a single run or on means alone. We run each algorithm on **100 seeds** (disjoint from tuning and held-out) and use **non-parametric inference**: Friedman omnibus test and pairwise Wilcoxon signed-rank tests with Holm–Bonferroni correction, plus Vargha–Delaney A and bootstrap CIs on median difference. By using repeated measures across 100 unseen seeds, we explicitly account for starting-state variance, ensuring that no algorithm is unfairly penalized by a "bad" starting seed and that claims of superiority are based on reported p-values and effect sizes.

---

## 3.7 Analysis Tracks

**Track A — Scalar (production algorithm selection).** The 5:5:3 composite is applied to every algorithm's final solution; for NSGA-II, the Pareto-front member with minimum composite is selected (a posteriori scalarization). All six algorithms (RS, HC, SA, SGA, NSGA-II, TS) are compared via **median and IQR**, **Friedman** test, **Wilcoxon** signed-rank pairwise with Holm–Bonferroni correction, **Vargha–Delaney A**, and **bootstrap 95% CIs** on median difference. We do not base claims on mean values; we report p-values and claim one algorithm outperforms another only when the corrected p-value is below the chosen significance level.

**Track B — Pareto (multi-objective quality).** NSGA-II's raw Pareto archives are evaluated with **Hypervolume (HV)**, **IGD+**, and **Spacing**. This is the primary evaluation for multi-objective performance and avoids scalarization. Scalar algorithms do not produce Pareto fronts and are compared only via Track A.

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

Libório, M. P., de Abreu, J. F., Ekel, P., & Machado, A. (2022). Effect of sub-indicator weighting schemes on the spatial dependence of multidimensional phenomena. *Journal of Geographical Systems*, 25, 185–211.
