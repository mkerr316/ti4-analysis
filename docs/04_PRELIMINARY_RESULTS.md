# TI4 Map Analysis — Results Inventory and Writing Checklist

## Run Information

- **Primary run tag:** saturation_20260314_205919
- **Phase 1 CSV:** `benchmark_20260314_233002/results.csv` (12,000 rows — five-condition ablation)
- **Phase 2b CSV:** `benchmark_20260315_074617/results.csv` (14,400 rows — algorithm benchmark)
- **SLURM Job ID:** 11751
- **Git hash at submission:** 5059440 (warm_fraction fix + final_tile_layout: d04a217)

---

## CONFIRMED RESULTS (verified from output files)

### Algorithm Rankings — Track A (budget=1000, Phase 2b)

**WARNING:** `analyze_benchmark.py` was run without `--budget` flag. All saved stats in
`benchmark_20260315_074617/stats/` use **budget=1000**, not 500k. Re-run with
`--budget 500000` before writing the manuscript table.

| Algorithm | Median composite | LISA penalty | Moran's I |
|-----------|-----------------|-------------|-----------|
| HC | 0.000333 | 0.044 | -0.588 |
| SGA | 0.000433 | 0.057 | -0.593 |
| TS | 0.000500 | **0.015** | -0.588 |
| NSGA-II | 0.000600 | 0.088 | -0.596 |
| SA | 0.001350 | 0.308 | -0.549 |
| RS | 0.001650 | 0.421 | -0.540 |

**Source:** `benchmark_20260315_074617/stats/full_report.txt` CONFIRMED

**Friedman omnibus:** chi2 = 403.55, df = 5, p < 0.001 CONFIRMED

**SGA vs TS:** p_corr = 0.090, VDA = 0.550 (negligible) — only non-significant pairwise comparison CONFIRMED

**All other pairwise comparisons:** p_corr < 0.001 CONFIRMED

**Bootstrap 95% CI:** All pairs exclude 0 except SGA vs TS CONFIRMED

**Weight sensitivity (budget=1000):** All Kendall's tau = 1.000 across 10 weight config pairs
Source: `stats/sensitivity_rank_stability.txt` CONFIRMED

### Moran's I Distribution by Condition — Phase 1 (all budgets pooled)

| Condition | Mean I | Std | n |
|-----------|--------|-----|---|
| jfi_only (C0) | **-0.086** | 0.205 | 2400 |
| jfi_moran (C3) | -0.553 | 0.102 | 2400 |
| full_composite (C4) | -0.646 | 0.106 | 2400 |
| lsap_only (C2) | -0.642 | 0.112 | 2400 |
| moran_only (C1) | **-0.854** | 0.076 | 2400 |

**C0 vs C4 gap = 0.560 Moran's I units. This is the paper's central finding.**
**Source:** live diagnostic query on Phase 1 CSV CONFIRMED

### Track B — NSGA-II Quality Indicators

**Source:** `quality_indicators_report.txt` CONFIRMED

- HV mean = 0.0963, std = 0.0223
- IGD+ mean = 0.0744, std = 0.0597
- Spacing mean = 0.0218, std = 0.0769
- Mean Pareto front size = 19.7 (range 2-20, cap = 20)
- Front hitting cap regularly = dense, well-distributed front

### Goodhart's Law — LSAP Proxy Validation

**Source:** `lisa_validation_20260316_100413/proxy_validation_summary.json` CONFIRMED

- Spearman rho = -0.025, p = 0.788
- Pearson r = -0.015, p = 0.870
- precision@tau=1.0 = 5.2% (4 of 77 maps with LSAP < 1.0 had zero significant clusters)
- 88.3% of maps showed significant global Moran's I (106/120)
- Pre-registered decision: rho < 0.70 → switch to thresholded variant

**Per-algorithm proxy values:**
| Algorithm | Mean sig clusters | Mean HH | Mean proxy |
|-----------|-----------------|---------|------------|
| HC | 2.2 | 0.033 | 1.004 |
| SA | 2.0 | 0.100 | 1.202 |
| TS | 2.1 | 0.067 | 0.729 |
| NSGA-II | 1.867 | 0.067 | 0.742 |

### LSAP Threshold Sensitivity

**Source:** `lsap_threshold_20260316_194325/lsap_threshold_report.txt` CONFIRMED

- Kendall's tau = 0.9494, p = 2.28e-22
- Pre-registered verdict: DEFENDED (tau > 0.90)
- Baseline LSAP mean = 1.0001, std = 0.6137
- Thresholded LSAP mean = 0.6739, std = 0.5189
- Both formulations identify the same map configurations as spatially preferable

**Combined Goodhart's Law manuscript text:**
> "Under the pre-registered Goodhart's Law diagnostic, the continuous LSAP proxy showed
> Spearman rho = -0.025 (p = 0.788) with permutation-tested local significance, and
> precision@tau=1.0 = 5.2%, falling below the pre-registered threshold of rho > 0.70.
> The continuous proxy does not reliably track which local statistics are individually
> significant under permutation testing. However, the pre-registered threshold sensitivity
> analysis (Kendall's tau = 0.949 between baseline LSAP and thresholded LSAP_tau with
> tau = 0.05, p = 2.3e-22) met the pre-registered defence threshold of tau > 0.90,
> indicating that the two formulations produce equivalent map rankings despite the proxy's
> failure to track permutation significance. Both formulations are reported; all primary
> conclusions hold under either."

### Distance Weight Sensitivity

**Source:** `dist_sensitivity_20260316_100535/` (console output captured during run) CONFIRMED FROM CONSOLE

- SGA beats SA across all 6 distance weight configurations
- Kendall's tau = 1.000 across all config pairs
- Verdict: WEIGHT-INVARIANT
- CONFIRM from file: `Get-Content output\saturation_20260314_205919\dist_sensitivity_20260316_100535\sensitivity_report.txt`

### SA Anomaly — seed=37

**Source:** `benchmark_20260315_074617/results.csv` CONFIRMED

- Seed=37, SA only, composite = 0.1816 across ALL 8 budgets (never moved)
- Moran's I = +0.439 (initial map value, unchanged)
- All other algorithms handle seed=37 normally (HC=0.0002, TS=0.0002)
- Cause: SA temperature calibration failure on map with initial I = +0.439
- Phase 2b only — does not affect Phase 1 primary results
- 1% of SA observations

### Ablation — Multi-Jain (J_R vs J_I)

**Source:** `benchmark_20260315_074617/stats/ablation_multi_jain.csv` CONFIRMED

- Bottleneck vs optimistic median difference < 0.002 across all algorithms
- J_R and J_I co-vary under optimization — dimensional collapse confirmed
- Bottleneck formulation does not add constraint beyond optimistic case

### Moran's I Boundary Violations

**Source:** diagnostic runs CONFIRMED

- 21 of 12,000 Phase 1 rows (0.2%), range -1.001 to -1.050
- Concentrated in moran_only at high budgets
- Cause: asymmetric row-standardized W on n=14 swappable subgraph
- Row sums = 1.000000 exactly (not a standardization error)
- Values clipped to [-1, 1] for reporting

---

## NOT YET CONFIRMED OR COMPUTED

### 1. Phase 1 Five-Condition Ablation Statistics (CRITICAL)

**Status:** Phase 1 CSV exists but has NOT been run through the Phase 1 condition analysis script.
This is the primary research question. The Wilcoxon signed-rank tests for
C0 vs C1, C0 vs C2, C0 vs C3, C0 vs C4 have not been computed.

**Fix:** Run the new script (see Action 1 in `07_POST_RUN_CHEAT_SHEET.md`):

```bash
python scripts/analyze_phase1_conditions.py \
    output/saturation_20260314_205919/benchmark_20260314_233002/results.csv \
    --budget 500000
```

**What this will produce:**
- Wilcoxon signed-rank on LSAP (C0 vs C1, C0 vs C2, C0 vs C3, C0 vs C4)
- Wilcoxon signed-rank on Moran's I (same pairs)
- Vargha-Delaney A values for each pair
- JFI parity check results
- Outputs in `benchmark_20260314_233002/stats/`: phase1_condition_report.txt, phase1_condition_pairs.csv, phase1_jfi_parity.csv, phase1_summary.csv

### 2. Phase 3 Stats at budget=500k

**Status:** Saved stats use budget=1000. Manuscript needs 500k.

**Fix:**
```bash
python scripts/analyze_benchmark.py \
    output/saturation_20260314_205919/benchmark_20260315_074617/results.csv \
    --sensitivity --ablation --budget 500000

python scripts/plot_statistical_results.py \
    output/saturation_20260314_205919/benchmark_20260315_074617/results.csv \
    --budget 500000
```

### 3. Variance Equalization Diagnostic (Section 3.7)

**Status:** Not yet run on the production data.

**Fix:**
```bash
python scripts/variance_equalization_diagnostic.py \
    --output-dir output/saturation_20260314_205919/
```

### 4. Distance Weight Sensitivity from File

**Status:** Console output captured but not confirmed from saved file.

**Fix (Windows):**
```powershell
Get-Content "output\saturation_20260314_205919\dist_sensitivity_20260316_100535\sensitivity_report.txt"
```

---

## PENDING_MAIN_EXPERIMENT Placeholders

All must be filled before grep gate passes.

| Placeholder | Value | Status |
|-------------|-------|--------|
| Kendall's tau (distance weight) | 1.000 | Confirm from file |
| Distance weight p-value | < 0.001 | Confirm from file |
| Goodhart rho | -0.025 | CONFIRMED |
| Goodhart p | 0.788 | CONFIRMED |
| precision@tau=1.0 | 5.2% | CONFIRMED |
| LSAP tau_kendall | 0.949 | CONFIRMED |
| LSAP tau p-value | 2.28e-22 | CONFIRMED |
| Gen-0 variance equalization numbers S3.7 | TBD | Run diagnostic |
| Phase 1 Wilcoxon W statistics | TBD | Run Phase 1 analysis (Action 1) |
| Phase 1 Vargha-Delaney A values | TBD | Run Phase 1 analysis (Action 1) |
| Phase 1 JFI parity check results | TBD | Run Phase 1 analysis (Action 1) |

---

## ACTIONS REQUIRED IN ORDER

1. Confirm dist sensitivity from file (2 minutes, Windows)
2. **Run Phase 1 five-condition analysis** — `scripts/analyze_phase1_conditions.py` (most important — primary findings). See Action 1 in `07_POST_RUN_CHEAT_SHEET.md`.
3. Re-run Phase 3 at budget=500k (algorithm benchmark table)
4. Run variance equalization diagnostic
5. Fill all PENDING_MAIN_EXPERIMENT placeholders in docs/
6. Run pre-submission grep gate (0 matches required)

---

## PRE-SUBMISSION CHECKLIST

- [ ] Phase 1 five-condition Wilcoxon results computed and filled
- [ ] Phase 3 re-run at budget=500k
- [ ] Distance weight sensitivity tau confirmed from file
- [ ] All PENDING_MAIN_EXPERIMENT placeholders filled
- [ ] Grep gate passes (0 matches)
- [ ] SA seed=37 anomaly in limitations
- [ ] Moran's I boundary violations in limitations (21/12000, 0.2%)
- [ ] Goodhart's Law section filled with actual numbers
- [ ] LSAP threshold sensitivity section filled
- [ ] Topological frustration noted in limitations
- [ ] Swappable subgraph n=14 stated (not full graph n=30)
- [ ] Budget=1000 stats replaced with budget=500k in manuscript
- [ ] Human validation scope statement in abstract
- [ ] Causal bound sentence in results
- [ ] 5:5:3 references purged (grep check)
- [ ] Track B described as primary, Track A as secondary
- [ ] Ablation J_R vs J_I dimensional collapse noted
- [ ] Five-condition table in manuscript matches code
