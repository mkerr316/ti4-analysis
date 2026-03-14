# Design Rationale: Geographic Paper and Experiment Protocol

This document is the single source of truth for what the paper is trying to do and how the experiment is designed. It prevents future collaborators or reviewers from steering the work back toward an algorithmic-contribution framing.

---

## 1. Paper Goal

The paper’s contribution is **geographic/methodological**, not algorithmic. We show that **spatial metrics (Moran’s I, LSAP) add meaningful constraint beyond scalar distributional fairness (JFI)** on topologically embedded discrete spaces. Algorithms (e.g. SA) are **instruments** used to generate optimized maps; we report which instrument we use for reproducibility, but the finding is about spatial metrics, not about algorithm novelty. The study is scoped to small discrete bounded topologies (e.g. n=37); human validation (causal chain from spatial clustering to competitive disadvantage in play) is explicitly **future work**.

---

## 2. Four-Condition Ablation

| Condition       | Weights ($w_1$, $w_2$, $w_3$) | Purpose |
|----------------|-------------------------------|---------|
| **C0** (jfi_only)      | 1:0:0 (JFI only)              | Baseline — current state of the art |
| **C1** (moran_only)   | 0:1:0 (Moran's I only)        | Does global clustering alone change maps? |
| **C2** (lsap_only)    | 0:0:1 (LSAP only)             | Does local clustering alone change maps? |
| **C3** (full_composite) | 1:1:1 (full composite)      | Do all three together produce what none achieves alone? |

Only these four conditions are run. We do **not** run pairwise combinations (e.g. 1:1:0, 0:1:1); the minimal ablation isolates **main effects** (does each term contribute?) and the **full combination** (does the joint model work?). All four conditions use **SA** as the instrument; same seeds, same budget per condition.

**Lex_key note:** The `--conditions` flag sets the **primary composite weights**. HC, TS, and SGA use a lexicographic tie-breaker with $J_{\max}$ as the secondary dimension regardless of condition, so under C1/C2 they would still apply secondary JFI pressure. **SA is the chosen instrument** for the main experiment because its Metropolis criterion does not impose that secondary optimization; the declared condition is then consistent with what is actually optimized.

---

## 3. Statistical Protocol

**Design:** For each of 100 seeds, run SA under each condition at the same budget. Collect final maps and **raw objective vectors** $(1-J_{\min}, I, \text{LSAP})$. **Cross-condition comparison uses raw objective vectors only**; composite score is **not** comparable across conditions (when one weight is 1.0 the composite is that term only). Composite score is only comparable within a condition.

**Primary tests (two pre-specified):**
1. Wilcoxon signed-rank on **LSAP** of final solutions, paired by seed.
2. Wilcoxon signed-rank on **Moran's I hinge** $\max(0, I - E[I])$ of final solutions, paired by seed.

α = 0.05. **Holm–Bonferroni correction is applied across all 6 primary spatial tests simultaneously (2 spatial metrics × 3 condition pairs: C0 vs C1, C0 vs C2, C0 vs C3), not within each pair independently.**

Pre-specified effect size: Vargha–Delaney A ≥ 0.64 for practical significance.

**Interpretation rule:** “Spatial metrics add constraint” is **supported if either** spatial metric (LSAP or Moran's I hinge) shows A ≥ 0.64 in the relevant comparison **and** the JFI parity check passes for that condition pair.

**JFI parity check (all three pairs C0 vs C1, C0 vs C2, C0 vs C3):**  
Wilcoxon signed-rank on $J_{\min}$, paired by seed. **H0: median $J_{\min}$ in Cx ≥ median $J_{\min}$ in C0 (one-sided, lower tail).** Rejection indicates Cx achieves lower JFI than the baseline — interpreted as **fairness sacrifice**, not spatial improvement. **Note that higher $J_{\min}$ is better (JFI = 1 is perfect equality); lower $J_{\min}$ in Cx than C0 means the condition sacrificed distributional fairness.**

**C1 and C2 interpretation (pre-specified):**  
C1 (Moran's I only) and C2 (LSAP only) are **expected to fail JFI parity** — SA will drive the single spatial objective with no JFI pressure, so JFI will likely be compromised. This is **not** a failure of the experiment; it demonstrates that single spatial objectives cannot replace the composite. **C3 is the only condition predicted to pass JFI parity while improving spatial profile.**

**Multi-budget prediction:**  
We predict the spatial profile difference between C0 and C3 will be detectable at all budgets ≥ 10k evaluations. If the difference only emerges at budgets ≥ 100k, we interpret this as evidence that spatial terms require sustained search pressure and report the minimum effective budget as a secondary finding.

---

## 4. Primary Figure

**One primary figure:** $(1-J_{\min}, I, \text{LSAP})$ for **all four conditions** (C0, C1, C2, C3) as grouped boxplots. This is the paper’s central empirical contribution — readers see C0 baseline, C1 moving Moran's I but not necessarily LSAP, C2 moving LSAP but not necessarily Moran's I, and C3 moving both while maintaining JFI parity. Not only C0 vs C3.

---

## 5. Equal Weights and Gen-0 Decision

**Weights:** The full composite (C3) uses **equal weights 1:1:1** (i.e. $w_1 = w_2 = w_3 = 1/3$). This is self-justifying and consistent with the four-condition scheme; we do not optimize or defend a particular weight vector.

**Gen-0 equalizer:** We use **Option B — nominal 1:1:1 without the Gen-0 variance equalizer.** The variance diagnostic (Moran's I dominating early search) is reported as a **finding**; it resolves at convergence. Option A (equalizer as default) is documented as a **robustness check** for future work. This keeps the geographic contribution simple and avoids NSGA-II comparability issues in methods-justification runs.

---

## 6. Human Validation Limitation

This study demonstrates computationally that spatial autocorrelation metrics detect map configurations that scalar fairness metrics cannot optimize toward. The causal chain from spatial clustering to competitive disadvantage in human play is not tested here and constitutes the **primary empirical question for subsequent work**, addressable through telemetry from the companion application or controlled play experiments. This sentence appears in the abstract and in limitations.

---

## 7. Pipeline and Minimum Viable Run

- **Phase 0:** SA hyperparameter tuning (disjoint seeds, e.g. 9000–9149).
- **Phase 1 (primary):** Four-condition SA run using **tuned SA parameters** — same seeds, same budgets (e.g. 10k, 50k, 100k, 500k). Invoke with `--conditions jfi_only,moran_only,lsap_only,full_composite` and `--algorithms sa`.
- **Phase 2 (methods justification):** Algorithm benchmarking — all six algorithms, same budgets, using all tuned parameters. This run is **not** the primary result; it justifies the choice of SA as the instrument.

**Pre-submission gate:** Run `scripts/pre_submission_check.sh` from the project root; it must report **zero matches**.

**Mutual exclusivity:** `--conditions` and `--weight-grid-step > 0` cannot both be set; the script exits with an explicit error if they are.
