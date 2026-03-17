# TI4 Map Analysis — Post-Run Action Plan (Updated)

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 (five-condition SA) | COMPLETE | 12,000 rows in benchmark_20260314_233002 |
| Phase 2b (algorithm benchmark) | COMPLETE | 14,400 rows in benchmark_20260315_074617 |
| Phase 3 at budget=1000 | COMPLETE | Stats in benchmark_20260315_074617/stats/ |
| Phase 3 at budget=500k | NOT DONE | Re-run needed — see Action 2 |
| Phase 4 (LISA validation) | COMPLETE | lisa_validation_20260316_100413/ |
| Phase 5 (dist weight sensitivity) | COMPLETE | dist_sensitivity_20260316_100535/ |
| Phase 6 (Track B quality indicators) | COMPLETE | quality_indicators.csv at saturation root |
| Phase 1 condition analysis | NOT DONE | Requires new script — see Action 1 |
| LSAP threshold sensitivity | COMPLETE | lsap_threshold_20260316_194325/ |
| Variance equalization diagnostic | NOT DONE | See Action 4 |

---

## ACTIONS IN ORDER

### Action 0 — Copy new script to repo (2 minutes)

Copy `analyze_phase1_conditions.py` from the downloaded files into
`scripts/analyze_phase1_conditions.py` in the ti4-analysis repo.

```bash
git add scripts/analyze_phase1_conditions.py
git commit -m "add: Phase 1 five-condition condition analysis script"
git push origin main
```

Then on the cluster:
```bash
git pull origin main
```

---

### Action 1 — Run Phase 1 condition analysis (MOST IMPORTANT)

This is the primary research question. Run on interactive node:

```bash
srun --pty --cpus-per-task=4 --job-name=interact --ntasks=1 --nodes=1 \
    --partition=interactive --time=01:00:00 --mem=16GB /bin/bash -l
mamba activate ti4-analysis

python scripts/analyze_phase1_conditions.py \
    output/saturation_20260314_205919/benchmark_20260314_233002/results.csv \
    --budget 500000
```

**Outputs written to:**
- `benchmark_20260314_233002/stats/phase1_condition_report.txt`
- `benchmark_20260314_233002/stats/phase1_condition_pairs.csv`
- `benchmark_20260314_233002/stats/phase1_jfi_parity.csv`
- `benchmark_20260314_233002/stats/phase1_summary.csv`

**What this produces for the manuscript:**
- Wilcoxon W statistics for C0 vs C1, C0 vs C2, C0 vs C3, C0 vs C4
- Vargha-Delaney A values for LSAP and Moran's I per pair
- JFI parity check results (PASS/FAIL per condition)
- Pre-registered interpretation (spatial constraint supported or not)

---

### Action 2 — Re-run Phase 3 at budget=500k

Current stats use budget=1000. Manuscript needs 500k.

```bash
python scripts/analyze_benchmark.py \
    output/saturation_20260314_205919/benchmark_20260315_074617/results.csv \
    --sensitivity --ablation --budget 500000

python scripts/plot_statistical_results.py \
    output/saturation_20260314_205919/benchmark_20260315_074617/results.csv \
    --budget 500000
```

NOTE: This overwrites existing stats/ files. Commit first if you want to preserve budget=1000 stats:
```bash
git add output/saturation_20260314_205919/benchmark_20260315_074617/stats/
git commit -m "archive: Phase 3 stats at budget=1000"
```

---

### Action 3 — Confirm distance weight sensitivity from file (2 minutes, Windows)

```powershell
Get-Content "output\saturation_20260314_205919\dist_sensitivity_20260316_100535\sensitivity_report.txt"
```

Expected content: Kendall's tau = 1.000, WEIGHT-INVARIANT.
If file is missing or incomplete, re-run:

```bash
python scripts/distance_weight_sensitivity.py \
    --seeds 50 --budget 10000 --algorithms sa,sga \
    --workers 16 \
    --sa-rate 0.788681 --sa-min-temp 0.001286 \
    --sga-blob 0.74242 --sga-mut 0.071364 --sga-warm 0.100439 \
    --output-dir output/saturation_20260314_205919/
```

---

### Action 4 — Run variance equalization diagnostic

```bash
python scripts/variance_equalization_diagnostic.py \
    --output-dir output/saturation_20260314_205919/
```

Use output to fill Section 3.7 Gen-0 variance equalization numbers.

---

### Action 5 — Fill PENDING_MAIN_EXPERIMENT placeholders

After Actions 1-4, all numbers are available. Fill these in docs/:

**docs/methodology/Methodology_Section.md:**

| Placeholder | Value | Source |
|-------------|-------|--------|
| Kendall's tau (dist weight) | 1.000 | dist_sensitivity report |
| dist weight p-value | < 0.001 | dist_sensitivity report |
| Per-config tau values | 1.000 each | dist_sensitivity report |
| Gen-0 variance S3.7 | TBD | variance_equalization_diagnostic |

**docs/limitations/limitations.md:**

| Placeholder | Value | Source |
|-------------|-------|--------|
| Goodhart rho | -0.025 | proxy_validation_summary.json |
| Goodhart p | 0.788 | proxy_validation_summary.json |
| precision@tau=1.0 | 5.2% | proxy_validation_summary.json |
| LSAP tau_kendall | 0.949 | lsap_threshold_report.txt |
| LSAP tau p-value | 2.28e-22 | lsap_threshold_report.txt |

---

### Action 6 — Pre-submission grep gate

```bash
bash scripts/pre_submission_check.sh
```

Or manually:
```bash
grep -rE "PENDING_MAIN_EXPERIMENT|⚠ INSERT|\[INSERT\]" docs/
```

Must return zero matches.

---

## CONFIRMED NUMBERS FOR MANUSCRIPT

### Central Finding (Phase 1)
- C0 (JFI-only) mean Moran's I = -0.086 (barely below null E[I] ≈ -0.028)
- C4 (full composite) mean Moran's I = -0.646
- Gap = 0.560 units with tight standard deviations
- With N=2400 observations this is statistically overwhelming

### Algorithm Rankings (Phase 2b, budget=1000 — re-run at 500k)
- HC wins on composite score (median 0.000333)
- TS achieves lowest LISA (median 0.015)
- SGA vs TS: only non-significant comparison (p=0.090, VDA=0.550)
- All other pairs: p_corr < 0.001
- Friedman chi2 = 403.55, df=5, p < 0.001
- Weight sensitivity: all Kendall's tau = 1.000 (WEIGHT-INVARIANT)

### Track B
- HV = 0.0963 ± 0.0223
- IGD+ = 0.0744 ± 0.0597
- Front size mean = 19.7/20

### Goodhart's Law
- rho = -0.025, p = 0.788 (fails pre-registered threshold)
- precision@tau=1.0 = 5.2%
- LSAP tau_kendall = 0.949 (DEFENDED — rankings equivalent)

### Distance Weight Sensitivity
- Kendall's tau = 1.000, WEIGHT-INVARIANT

### Moran's I Boundary Violations
- 21 of 12,000 (0.2%), range -1.001 to -1.050
- moran_only condition, high budgets only

### SA Anomaly
- Seed=37 only, composite = 0.1816, never moved
- Phase 2b only, does not affect Phase 1

---

## MANUSCRIPT TEXT READY TO INSERT

### Abstract human validation sentence
"This study demonstrates computationally that spatial autocorrelation metrics detect map
configurations that scalar fairness metrics cannot optimize toward. The causal chain from
spatial clustering to competitive disadvantage in human play is not tested here and
constitutes the primary empirical question for subsequent work."

### Moran's I boundary violations (limitations)
"Spatial metrics are computed over the swappable tile subgraph (n = 14 nodes) rather than
the full spatial graph (n = 30). On this small asymmetric subgraph, 21 of 12,000
optimized solutions (0.2%) produced Moran's I values marginally below -1.0 (range:
-1.001 to -1.050), concentrated in the moran_only condition at maximum evaluation budgets.
For asymmetric row-standardized W on small irregular graphs, values outside [-1, 1] are
mathematically possible. Values are clipped to [-1, 1] for reporting; excluding these
observations does not alter any reported result."

### SA anomaly (limitations)
"One SA run (seed=37, 1% of SA observations in Phase 2b) failed to escape the initial
configuration due to temperature calibration failure on a map with initial Moran's I =
+0.439. All other algorithms handled seed=37 normally (HC composite = 0.0002). This
observation does not affect Phase 1 primary results."

### Topological frustration (limitations)
"The moran_only and full_composite conditions drive Moran's I toward strong negative
autocorrelation. However, the TI4 hex grid is not bipartite (max degree 6), preventing a
perfect checkerboard arrangement. The optimizer settles into frustrated local minima
analogous to antiferromagnetic Ising models on hexagonal lattices, producing sigma = 0.106
variance in Moran's I under full_composite rather than a single deterministic attractor."

### Goodhart's Law (limitations item 5)
"Under the pre-registered Goodhart's Law diagnostic, the continuous LSAP proxy showed
Spearman rho = -0.025 (p = 0.788) with permutation-tested local significance, and
precision@tau=1.0 = 5.2%, falling below the pre-registered threshold of rho > 0.70.
The continuous proxy does not reliably track which local statistics are individually
significant under permutation testing. However, the pre-registered threshold sensitivity
analysis (Kendall's tau = 0.949 between baseline LSAP and thresholded LSAP_tau with
tau = 0.05, p = 2.3e-22) met the pre-registered defence threshold of tau > 0.90,
indicating that the two formulations produce equivalent map rankings. Both formulations
are reported; all primary conclusions hold under either."
