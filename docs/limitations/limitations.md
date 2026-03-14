# Limitations: Small-N Problem in Spatial Statistics

## The flaw

Standard spatial statistics (e.g. Moran's I) rely on **asymptotic normality** for variance calculations and z-scores (e.g. the 1.96 threshold). A TI4 map has roughly **37 swappable system tiles**. At **N = 37**, asymptotic approximations are mathematically shaky. Top reviewers will reject papers that claim statistical significance purely from analytical variance on N=37 grids. We do not use Getis-Ord $G_i^*$; it has been removed from the codebase and methodology because at N≈37 its asymptotic z-score is unreliable and because LISA/LSAP are better suited to spatial-outlier detection (see README and Methodology §3.3).

## The fix (what we do)

- **Exact conditional permutation testing** is required for significance.
- The codebase leans on **`scripts/validate_lisa_proxy.py`** for evaluation: it re-runs optimization for a subset of seeds, then applies **conditional-permutation LISA** (9,999 permutations per location, default) to count significant H-H and L-L clusters. With 9,999 permutations, the minimum attainable p-value is 1/10,000 = 0.0001, allowing significance to be evaluated under multiple-testing correction.
- **LSAP** (the continuous LISA proxy) is framed as a **computational heuristic** optimized during search; the **final maps are strictly evaluated** using 9,999-permutation tests. No significance claims are based on analytical variance for LISA.

## What is already in place

- **LISA**: `validate_lisa_proxy.py` implements conditional permutation LISA (Anselin 1995): for each location *i*, *z*[i] is fixed, other values are permuted, and local *I*_i is recomputed to build the null. p-value = (count_extreme + 1) / (n_perms + 1) with `--n-perms 9999` (default). The optimizer uses the continuous LSAP only as a heuristic; significance is claimed only from this script.
- **Global Moran's I in the inner loop**: `FastMapState.morans_i()` returns only the scalar *I* (no z-score, no significance). It is used as a continuous objective to minimize |*I*|, not for inference.
- **Global Moran's I significance**: `validate_lisa_proxy.py` now computes a **permutation-based p-value** for global Moran's I (exact permutation of the value vector over the fixed set of locations, no bootstrapping). The script writes `global_I` and `global_I_pvalue` per map and summarizes the fraction of maps with global I significant at α = 0.05. **Manuscript claims that optimization "optimizes spatial dispersion" should cite this permutation test** (and the validation script), not analytical variance or z-scores.

## Remaining gaps and recommendations

1. **Global Moran's I**  
   Permutation-based significance for global *I* is now implemented in `validate_lisa_proxy.py`. Do not use `spatial_metrics.morans_i()` analytical variance or a 1.96 threshold for significance claims at N=37.

2. **Getis–Ord Gi\***  
   Getis-Ord $G_i^*$ is not used in this project; it has been removed from the codebase and methodology. At N≈37 the asymptotic z-score is unreliable, and LISA/LSAP with conditional permutation testing are the only defensible methods for local spatial association. Do not reintroduce Gi\* for significance claims.

3. **Multiple testing (LISA)**  
   With 37 local tests, α = 0.05 per test does not control family-wise error. Bonferroni (α/37) assumes independent tests and is paralyzingly conservative for spatially correlated hex grids, leading to Type II errors.  
   - **Recommendation**: Use **False Discovery Rate (FDR), Benjamini–Hochberg** at q < 0.05 as the committed correction for LISA. Report both “per-location” and “corrected” counts so reviewers see the distinction.

4. **Power and effect size**  
   With N=37, permutation tests are valid but have limited power against weak clustering.  
   - **Recommendation**: Report effect size and N explicitly (e.g. “number of locations with significant local clustering (permutation p < 0.05)” and “mean number of significant H-H/L-L clusters per map”). Make the “N=37, permutation-based” and “LSAP as heuristic, significance only from validation” framing explicit in the paper.

5. **Goodhart's Law boundary for LSAP**
   The continuous LSAP proxy (sum of positive variance-normalized local Moran's I_i) is
   optimized without a significance threshold. A theoretical risk exists that the optimizer
   exploits the proxy by generating many small, sub-threshold local statistics that
   individually escape significance detection under permutation testing while collectively
   lowering the objective. If this occurs, solutions with low LSAP would not map cleanly
   to solutions with few permutation-significant clusters — the proxy would be gamed.

   `validate_lisa_proxy.py` tests this boundary directly: it computes Spearman ρ and
   Pearson r between the continuous LSAP value and the permutation-tested significant
   cluster count (FDR-corrected, Benjamini–Hochberg q < 0.05) across a sample of
   optimized solutions, and reports precision at proxy thresholds τ ∈ {0.5, 1.0, 2.0,
   3.0, 5.0} (fraction of maps with LSAP < τ that have zero significant clusters). Insert
   the Spearman ρ, Pearson r, and precision-at-τ=1.0 values from `proxy_validation_summary.json`
   here before submission: [ρ = INSERT, r = INSERT, precision@τ=1.0 = INSERT%]. If ρ > 0.80
   and precision at τ = 1.0 exceeds 90%, the proxy tracks the significance-tested outcome
   closely enough to rule out systematic gaming. If ρ < 0.70, the LSAP threshold
   sensitivity test (Task 4, `lsap_threshold_sensitivity.py`) must be re-run using the
   thresholded variant `lisa_penalty_thresholded(τ=0.05)` as the primary objective, and
   Track A results must be reported under both formulations with the divergence noted as
   a limitation.

## Is this a fundamental limitation of TI4-style map optimization?

- **N ≈ 37** is fixed by the game; it cannot be increased.
- The **unreliable** part is asymptotic normality/variance at this N. The **correct** approach is permutation-based (or other exact/simulation-based) inference. That is not a workaround—it is the statistically appropriate method for small N.
- So the limitation is: **any claim of statistical significance must use permutation-based (or similar) evaluation**, not analytical variance or z-scores at N=37. The codebase already does this for LISA via `validate_lisa_proxy.py`.
- The only inherent limitation is **power**: with 37 tiles, very weak clustering may be undetectable. Transparency (reporting N, permutation, and effect sizes) is the appropriate response.

**Bottom line:** Use permutation-based evaluation for all significance claims; treat LSAP as the optimization heuristic; do not use analytical variance or z-scores for significance at N=37; use FDR (Benjamini–Hochberg) for LISA multiple-testing correction; global Moran's I significance is reported via the permutation test in `validate_lisa_proxy.py`. Then the small-N issue is addressed rather than a permanent, unfixable flaw of TI4-style map optimization.
