# Held-Out Validation Variance (CV vs Held-Out)

## The phenomenon

After hyperparameter tuning with Optuna, the pipeline runs:

1. **5-fold cross-validation** on the tuning seeds (e.g. 9,000–9,099): mean composite score and std *across fold means*.
2. **Held-out validation** on seeds 9,100–9,149 (50 seeds): mean and std of composite score *per seed*.

In practice, CV standard deviation can be small (e.g. ±0.0028) while held-out standard deviation is much larger (e.g. ±0.0299). A reviewer trained in predictive ML might interpret this as **hyperparameter overfitting** to the tuning set.

## Why this is not overfitting

We are doing **combinatorial optimization**, not supervised learning. The narrative is:

| ML interpretation | Optimization interpretation |
|-------------------|-----------------------------|
| High held-out variance = overfitting | High held-out variance = **varying difficulty of problem instances** (different seeds → different regions of a rugged fitness landscape) |
| Model memorized training data | Same hyperparameters (e.g. tabu tenure) applied to *unseen starting maps*; some are near fair configs, others in deep local minima |

## Three points to explain in the paper

**(A) Landscape ruggedness.** The variance on the held-out set is not a failure of the hyperparameters. The TI4 map fitness landscape is highly rugged. Different random seeds place the search in vastly different areas: some starting maps are relatively close to a fair configuration, while others require traversing deep, unrecoverable local minima. The spread in held-out performance *reflects this inherent variability of problem difficulty*, not that the tuned settings have "memorized" the tuning seeds.

**(B) Resiliency of the mean.** Although the *variance* increases, the **mean** performance typically degrades only slightly (e.g. CV mean 0.0545 → held-out mean 0.0599). That shows the optimized hyperparameters **generalize** to unseen starting conditions: on average, the same settings remain effective even though the absolute difficulty of individual maps varies wildly.

**(C) Phase 2 as mitigation.** This variance is why the main benchmark (Phase 2: Saturation Benchmark) uses **100 seeds** and **non-parametric inference** (Friedman, Wilcoxon signed-rank, Holm–Bonferroni). We do not judge algorithms on a single run or on the tuning/held-out sets. Repeated measures across 100 unseen seeds explicitly account for starting-state variance, so no algorithm is unfairly penalized by a "bad" seed, and claims of superiority are based on reported p-values and effect sizes.

## Where this is written in the paper

The full narrative for the manuscript is in **Methodology §3.6 (Hyperparameter validation and held-out variance)** in `docs/methodology/Methodology_Section.md`. Use the same A/B/C structure there; this document is an internal note.

## Data source for numbers

After running:

```bash
python scripts/optimize_hyperparameters.py --algo ts --trials 50 --eval-seeds 100
```

read the output directory (e.g. `output/optuna_YYYYMMDD_HHMMSS/best_params.json`). Fields:

- `cv_mean`, `cv_std` — cross-validation summary
- `held_out_mean`, `held_out_std` — held-out validation summary

Use these values in §3.6 and in the Response to Reviewers (template entry #4). Replace the placeholders in the methodology (e.g. "0.0545 ± 0.0028" and "0.0599 ± 0.0299") with your actual numbers.

## Implementation reference

- Tuning and validation flow: `scripts/optimize_hyperparameters.py` (k-fold CV, then held-out loop; results written to `best_params.json`).
- Phase 2 (100-seed benchmark): `scripts/benchmark_engine.py`; analysis: `scripts/analyze_benchmark.py` with `--budget N`.
