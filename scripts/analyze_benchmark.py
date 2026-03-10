#!/usr/bin/env python3
"""
Non-parametric statistical analysis of benchmark results.

Pipeline:
    1. Descriptive statistics (median, IQR) per algorithm per metric
    2. Shapiro-Wilk normality tests on paired differences (justifies non-parametric)
    3. Friedman omnibus test (repeated-measures, k ≥ 3 algorithms)
    4. Wilcoxon signed-rank pairwise post-hoc with Holm-Bonferroni correction
    5. Vargha-Delaney A effect sizes (stochastic superiority)
    6. Bootstrap 95% CIs on median differences

Reads:  <run_dir>/results.csv  (produced by benchmark_engine.py)
Writes: <run_dir>/stats/
            summary_table.csv
            summary_table.tex
            full_report.txt

Usage:
    python scripts/analyze_benchmark.py output/benchmark_20260310_123456/results.csv
    python scripts/analyze_benchmark.py output/benchmark_*/results.csv --budget 1000
"""

import argparse
import sys
import textwrap
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-parametric analysis of benchmark CSV")
    p.add_argument("csv", type=Path, help="Path to results.csv")
    p.add_argument("--budget", type=int, default=None,
                   help="Filter to a single budget level (for multi-budget CSVs)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (default: 0.05)")
    p.add_argument("--n-boot", type=int, default=10_000,
                   help="Bootstrap resamples (default: 10,000)")
    p.add_argument("--sensitivity", action="store_true",
                   help="Run weight sensitivity analysis across multiple weight configs")
    p.add_argument("--ablation", action="store_true",
                   help="Run Multi-Jain vs scalar JFI ablation analysis")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def vargha_delaney_a(x: np.ndarray, y: np.ndarray) -> float:
    """
    Vargha-Delaney A statistic (stochastic superiority).

    A = P(X < Y) + 0.5 * P(X == Y)

    Interpretation (lower composite = better, so A > 0.5 means X tends to be lower):
        A ≈ 0.50  →  no difference
        A > 0.56  →  small effect   (X tends to be lower/better)
        A > 0.64  →  medium effect
        A > 0.71  →  large effect
    Symmetric thresholds apply for A < 0.44, 0.36, 0.29.
    """
    n = len(x)
    count_less = np.sum(x < y)
    count_equal = np.sum(x == y)
    return float(count_less + 0.5 * count_equal) / n


def vda_magnitude(a: float) -> str:
    """Classify VD-A magnitude per Vargha & Delaney (2000)."""
    d = abs(a - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Bootstrap CI on median difference
# ---------------------------------------------------------------------------

def bootstrap_median_diff_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    BCa-style bootstrap 95% CI on median(X) - median(Y).

    Returns (point_estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = np.median(x[idx]) - np.median(y[idx])
    point = float(np.median(x) - np.median(y))
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

METRICS = [
    "composite_score", "morans_i", "jains_index",
    "jfi_resources", "jfi_influence",
    "lisa_penalty", "balance_gap",
]
ALGO_ORDER = ["rs", "hc", "sa", "nsga2", "ts"]


def load_and_validate(csv_path: Path, budget: int = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "budget" in df.columns and budget is not None:
        df = df[df["budget"] == budget]
    elif "budget" in df.columns and df["budget"].nunique() > 1 and budget is None:
        budgets = sorted(df["budget"].unique())
        print(f"Multiple budgets detected: {budgets}. Use --budget to select one.")
        print(f"Defaulting to budget={budgets[0]}")
        df = df[df["budget"] == budgets[0]]

    algos = sorted(df["algorithm"].unique())
    seeds = sorted(df["seed"].unique())
    print(f"Loaded {len(df)} rows: {len(seeds)} seeds × {len(algos)} algorithms")

    for algo in algos:
        n = len(df[df["algorithm"] == algo])
        if n != len(seeds):
            print(f"  WARNING: {algo} has {n} rows but expected {len(seeds)}", file=sys.stderr)

    return df


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Median, IQR, min, max per algorithm per metric."""
    rows = []
    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        for metric in METRICS:
            vals = sub[metric].dropna()
            rows.append({
                "algorithm": algo,
                "metric": metric,
                "median": vals.median(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
                "iqr": vals.quantile(0.75) - vals.quantile(0.25),
                "min": vals.min(),
                "max": vals.max(),
                "n": len(vals),
            })
    return pd.DataFrame(rows)


def shapiro_wilk_on_diffs(df: pd.DataFrame, alpha: float = 0.05) -> List[Dict]:
    """
    Shapiro-Wilk on paired differences for each metric × pair.

    Formal justification for non-parametric tests: if ANY pair's differences
    are non-normal, the parametric assumption is violated.
    """
    results = []
    algos = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    for metric in METRICS:
        wide = df.pivot(index="seed", columns="algorithm", values=metric).dropna()
        for a, b in combinations(algos, 2):
            if a not in wide.columns or b not in wide.columns:
                continue
            diffs = (wide[a] - wide[b]).values
            if len(diffs) < 3:
                continue
            stat, p = stats.shapiro(diffs)
            results.append({
                "metric": metric,
                "pair": f"{a}-{b}",
                "W": stat,
                "p_shapiro": p,
                "normal": p >= alpha,
            })
    return results


def friedman_test(df: pd.DataFrame) -> Dict:
    """Friedman omnibus test across all algorithms for composite_score."""
    wide = df.pivot(index="seed", columns="algorithm", values="composite_score").dropna()
    algos = [a for a in ALGO_ORDER if a in wide.columns]
    if len(algos) < 3:
        return {"chi2": np.nan, "p_friedman": np.nan, "df": 0, "n": 0,
                "significant": False, "note": f"Need ≥3 algorithms, got {len(algos)}"}

    arrays = [wide[a].values for a in algos]
    chi2, p = stats.friedmanchisquare(*arrays)
    return {
        "chi2": chi2,
        "p_friedman": p,
        "df": len(algos) - 1,
        "n": len(wide),
        "significant": p < 0.05,
    }


def pairwise_wilcoxon(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Wilcoxon signed-rank for all pairs × all metrics, with Holm-Bonferroni correction.
    """
    algos = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    pairs = list(combinations(algos, 2))
    rows = []

    for metric in METRICS:
        wide = df.pivot(index="seed", columns="algorithm", values=metric).dropna()
        raw_pvalues = []
        pair_stats = []

        for a, b in pairs:
            if a not in wide.columns or b not in wide.columns:
                continue
            x, y = wide[a].values, wide[b].values
            try:
                stat, p = stats.wilcoxon(x, y, alternative="two-sided")
            except ValueError:
                stat, p = np.nan, 1.0
            vda = vargha_delaney_a(x, y)
            raw_pvalues.append(p)
            pair_stats.append({
                "metric": metric,
                "pair": f"{a} vs {b}",
                "algo_a": a,
                "algo_b": b,
                "W_stat": stat,
                "p_raw": p,
                "vda_A": vda,
                "vda_mag": vda_magnitude(vda),
                "median_a": np.median(x),
                "median_b": np.median(y),
            })

        if raw_pvalues:
            reject, corrected, _, _ = multipletests(raw_pvalues, method="holm")
            for i, row in enumerate(pair_stats):
                row["p_corrected"] = corrected[i]
                row["significant"] = reject[i]
            rows.extend(pair_stats)

    return pd.DataFrame(rows)


def bootstrap_cis(df: pd.DataFrame, n_boot: int = 10_000, alpha: float = 0.05) -> pd.DataFrame:
    """Bootstrap 95% CIs on median difference for composite_score."""
    algos = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    wide = df.pivot(index="seed", columns="algorithm", values="composite_score").dropna()
    rng = np.random.default_rng(42)
    rows = []
    for a, b in combinations(algos, 2):
        if a not in wide.columns or b not in wide.columns:
            continue
        x, y = wide[a].values, wide[b].values
        pt, lo, hi = bootstrap_median_diff_ci(x, y, n_boot, alpha, rng)
        rows.append({
            "pair": f"{a} vs {b}",
            "median_diff": pt,
            "ci_lower": lo,
            "ci_upper": hi,
            "excludes_zero": (lo > 0) or (hi < 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(
    desc: pd.DataFrame,
    shapiro: List[Dict],
    friedman: Dict,
    wilcoxon_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    alpha: float,
) -> str:
    lines = []
    w = lines.append
    rule = "=" * 90

    w(rule)
    w("NON-PARAMETRIC STATISTICAL ANALYSIS — BENCHMARK RESULTS")
    w(rule)
    w("")

    # 1. Descriptive
    w("1. DESCRIPTIVE STATISTICS (Median [IQR])")
    w("-" * 90)
    for metric in METRICS:
        w(f"\n  {metric}:")
        sub = desc[desc["metric"] == metric]
        for _, r in sub.iterrows():
            w(f"    {r['algorithm']:<8s}  median={r['median']:8.4f}  "
              f"IQR=[{r['q25']:.4f}, {r['q75']:.4f}]  "
              f"range=[{r['min']:.4f}, {r['max']:.4f}]  n={int(r['n'])}")

    # 2. Shapiro-Wilk
    w(f"\n{rule}")
    w("2. SHAPIRO-WILK NORMALITY TEST ON PAIRED DIFFERENCES")
    w(f"   (Justification for non-parametric tests, alpha={alpha})")
    w("-" * 90)
    any_nonnormal = False
    for r in shapiro:
        marker = "NORMAL" if r["normal"] else "NON-NORMAL ***"
        if not r["normal"]:
            any_nonnormal = True
        w(f"  {r['metric']:20s}  {r['pair']:10s}  W={r['W']:.4f}  p={r['p_shapiro']:.4f}  {marker}")
    if any_nonnormal:
        w("\n  >> Non-normal distributions detected. Non-parametric tests are required.")
    else:
        w("\n  >> All differences appear normal. Non-parametric tests are conservative but valid.")

    # 3. Friedman
    w(f"\n{rule}")
    w("3. FRIEDMAN OMNIBUS TEST (composite_score)")
    w("-" * 90)
    w(f"  chi2 = {friedman['chi2']:.4f},  df = {friedman['df']},  "
      f"p = {friedman['p_friedman']:.6f},  n = {friedman['n']}")
    if friedman["significant"]:
        w("  >> SIGNIFICANT — proceed to pairwise post-hoc tests.")
    else:
        w("  >> NOT SIGNIFICANT — no evidence of algorithm differences.")
        w("     (Pairwise results below are exploratory only.)")

    # 4. Wilcoxon + Holm-Bonferroni
    w(f"\n{rule}")
    w("4. WILCOXON SIGNED-RANK PAIRWISE TESTS (Holm-Bonferroni corrected)")
    w("-" * 90)
    for metric in METRICS:
        sub = wilcoxon_df[wilcoxon_df["metric"] == metric]
        if sub.empty:
            continue
        w(f"\n  {metric}:")
        for _, r in sub.iterrows():
            sig = "***" if r["significant"] else "   "
            w(f"    {r['pair']:14s}  W={r['W_stat']:10.1f}  "
              f"p_raw={r['p_raw']:.4f}  p_corr={r['p_corrected']:.4f} {sig}  "
              f"VDA={r['vda_A']:.3f} ({r['vda_mag']})")

    # 5. Bootstrap CIs
    w(f"\n{rule}")
    w("5. BOOTSTRAP 95% CI ON MEDIAN DIFFERENCE (composite_score)")
    w("-" * 90)
    for _, r in boot_df.iterrows():
        excl = "excludes 0" if r["excludes_zero"] else "includes 0"
        w(f"  {r['pair']:14s}  median_diff={r['median_diff']:+.4f}  "
          f"95% CI=[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]  ({excl})")

    # 6. Interpretation guide
    w(f"\n{rule}")
    w("INTERPRETATION GUIDE")
    w(rule)
    w(textwrap.dedent("""\
        Friedman test:
          - p < 0.05 → at least one algorithm differs from the others

        Wilcoxon signed-rank (Holm-Bonferroni):
          - p_corr < 0.05 → statistically significant pairwise difference

        Vargha-Delaney A (for composite_score, lower = better):
          - A > 0.50 → first algorithm tends to produce LOWER (better) scores
          - |A - 0.5| < 0.06: negligible
          - |A - 0.5| < 0.14: small
          - |A - 0.5| < 0.21: medium
          - |A - 0.5| >= 0.21: large

        Bootstrap CI:
          - CI excludes 0 → practically significant median difference
    """))

    return "\n".join(lines)


def to_latex(desc: pd.DataFrame, wilcoxon_df: pd.DataFrame, boot_df: pd.DataFrame) -> str:
    """Generate a LaTeX summary table for the composite_score metric."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Algorithm comparison on composite score (lower is better). "
        r"Median [IQR], Wilcoxon p-values Holm-corrected, Vargha-Delaney $\hat{A}$ effect sizes.}",
        r"\label{tab:algorithm_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Algorithm & Median & IQR & Time (s) \\",
        r"\midrule",
    ]
    sub = desc[desc["metric"] == "composite_score"]
    for _, r in sub.iterrows():
        lines.append(
            f"  {r['algorithm'].upper()} & {r['median']:.4f} & "
            f"[{r['q25']:.4f}, {r['q75']:.4f}] & -- \\\\"
        )
    lines.extend([
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Pairwise comparisons (composite\_score)}} \\",
        r"Pair & $p_{\text{corr}}$ & $\hat{A}$ & 95\% CI($\Delta$ median) \\",
        r"\midrule",
    ])
    ws = wilcoxon_df[wilcoxon_df["metric"] == "composite_score"]
    for _, r in ws.iterrows():
        boot_row = boot_df[boot_df["pair"] == r["pair"]]
        ci_str = ""
        if not boot_row.empty:
            br = boot_row.iloc[0]
            ci_str = f"[{br['ci_lower']:+.4f}, {br['ci_upper']:+.4f}]"
        sig = "*" if r["significant"] else ""
        lines.append(
            f"  {r['pair']} & {r['p_corrected']:.4f}{sig} & "
            f"{r['vda_A']:.3f} ({r['vda_mag']}) & {ci_str} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Weight sensitivity analysis
# ---------------------------------------------------------------------------

WEIGHT_CONFIGS = {
    "current_5_5_3": {"morans_i": 5/13, "jains_index": 5/13, "lisa_penalty": 3/13},
    "equal_1_1_1":   {"morans_i": 1/3,  "jains_index": 1/3,  "lisa_penalty": 1/3},
    "jfi_dominant":  {"morans_i": 2/10, "jains_index": 6/10, "lisa_penalty": 2/10},
    "spatial_dom":   {"morans_i": 4/10, "jains_index": 2/10, "lisa_penalty": 4/10},
    "lisa_dominant":  {"morans_i": 2/10, "jains_index": 2/10, "lisa_penalty": 6/10},
}


def recompute_composite(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Recompute composite scores from per-objective columns under new weights."""
    n_spatial = 37  # typical TI4 6-player map
    lisa_divisor = max(1, n_spatial * (n_spatial - 1))
    return (
        weights["morans_i"]     * df["morans_i"].abs() +
        weights["jains_index"]  * (1.0 - df["jains_index"]) +
        weights["lisa_penalty"] * df["lisa_penalty"] / lisa_divisor
    )


def weight_sensitivity_analysis(
    df: pd.DataFrame, alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each weight configuration, recompute composite scores and re-run
    Friedman + Wilcoxon to determine if algorithm rankings are weight-invariant.
    """
    algos = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    if len(algos) < 2:
        return pd.DataFrame()

    rows = []
    for config_name, weights in WEIGHT_CONFIGS.items():
        df_copy = df.copy()
        df_copy["composite_reweighted"] = recompute_composite(df_copy, weights)

        wide = df_copy.pivot(
            index="seed", columns="algorithm", values="composite_reweighted"
        ).dropna()

        avail = [a for a in algos if a in wide.columns]
        if len(avail) < 2:
            continue

        # Friedman (if >= 3 algorithms)
        if len(avail) >= 3:
            arrays = [wide[a].values for a in avail]
            chi2, p_friedman = stats.friedmanchisquare(*arrays)
        else:
            chi2, p_friedman = np.nan, np.nan

        # Pairwise Wilcoxon + VDA
        from itertools import combinations as combos
        pairs = list(combos(avail, 2))
        for a, b in pairs:
            x, y = wide[a].values, wide[b].values
            try:
                _, p_raw = stats.wilcoxon(x, y, alternative="two-sided")
            except ValueError:
                p_raw = 1.0
            vda = vargha_delaney_a(x, y)
            winner = a if vda > 0.5 else b if vda < 0.5 else "tie"
            rows.append({
                "weight_config": config_name,
                "pair": f"{a} vs {b}",
                "p_friedman": p_friedman,
                "p_wilcoxon": p_raw,
                "vda_A": vda,
                "vda_mag": vda_magnitude(vda),
                "winner": winner,
                "median_a": float(np.median(x)),
                "median_b": float(np.median(y)),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-Jain ablation
# ---------------------------------------------------------------------------

def multi_jain_ablation(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Compare Multi-Jain bottleneck (min) vs scalar JFI (max) to show that
    the bottleneck catches maps where one dimension is equitable but the
    other is skewed.

    For each algorithm, recomputes composite scores under two JFI strategies:
      - bottleneck: 1 - min(jfi_r, jfi_i)   (current Multi-Jain)
      - optimistic: 1 - max(jfi_r, jfi_i)   (scalar-like, hides worst dimension)

    Reports how often rankings differ and the magnitude of the gap.
    """
    if "jfi_resources" not in df.columns or "jfi_influence" not in df.columns:
        return pd.DataFrame()

    algos = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    if len(algos) < 2:
        return pd.DataFrame()

    n_spatial = 37
    lisa_divisor = max(1, n_spatial * (n_spatial - 1))
    w_jfi = 5 / 13
    w_moran = 5 / 13
    w_lisa = 3 / 13

    df_ab = df.copy()
    df_ab["jfi_bottleneck"] = df_ab[["jfi_resources", "jfi_influence"]].min(axis=1)
    df_ab["jfi_optimistic"] = df_ab[["jfi_resources", "jfi_influence"]].max(axis=1)
    df_ab["jfi_gap"] = df_ab["jfi_optimistic"] - df_ab["jfi_bottleneck"]

    df_ab["score_bottleneck"] = (
        w_moran * df_ab["morans_i"].abs() +
        w_jfi * (1.0 - df_ab["jfi_bottleneck"]) +
        w_lisa * df_ab["lisa_penalty"] / lisa_divisor
    )
    df_ab["score_optimistic"] = (
        w_moran * df_ab["morans_i"].abs() +
        w_jfi * (1.0 - df_ab["jfi_optimistic"]) +
        w_lisa * df_ab["lisa_penalty"] / lisa_divisor
    )

    rows = []
    for strategy in ("bottleneck", "optimistic"):
        col = f"score_{strategy}"
        wide = df_ab.pivot(index="seed", columns="algorithm", values=col).dropna()
        avail = [a for a in algos if a in wide.columns]
        if len(avail) < 2:
            continue
        for a in avail:
            rows.append({
                "strategy": strategy,
                "algorithm": a,
                "median": float(wide[a].median()),
                "mean": float(wide[a].mean()),
            })

    # Per-seed rank comparison
    wide_bn = df_ab.pivot(index="seed", columns="algorithm", values="score_bottleneck").dropna()
    wide_op = df_ab.pivot(index="seed", columns="algorithm", values="score_optimistic").dropna()
    avail = [a for a in algos if a in wide_bn.columns and a in wide_op.columns]
    if len(avail) >= 2:
        ranks_bn = wide_bn[avail].rank(axis=1, method="min")
        ranks_op = wide_op[avail].rank(axis=1, method="min")
        rank_changes = (ranks_bn != ranks_op).any(axis=1).sum()
        total_seeds = len(ranks_bn)
    else:
        rank_changes = 0
        total_seeds = 0

    gap_summary = df_ab.groupby("algorithm")["jfi_gap"].agg(["mean", "median", "max"]).round(4)

    return pd.DataFrame(rows), rank_changes, total_seeds, gap_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 1

    df = load_and_validate(args.csv, args.budget)

    out_dir = args.csv.parent / "stats"
    out_dir.mkdir(exist_ok=True)

    print("\n── Descriptive statistics ──")
    desc = descriptive_statistics(df)

    print("\n── Shapiro-Wilk normality tests ──")
    shapiro = shapiro_wilk_on_diffs(df, args.alpha)
    n_nonnormal = sum(1 for r in shapiro if not r["normal"])
    print(f"   {n_nonnormal}/{len(shapiro)} pairs show non-normal differences")

    print("\n── Friedman omnibus test ──")
    friedman = friedman_test(df)
    print(f"   chi2={friedman['chi2']:.4f}, p={friedman['p_friedman']:.6f}")

    print("\n── Wilcoxon pairwise + Holm-Bonferroni ──")
    wilcoxon_df = pairwise_wilcoxon(df, args.alpha)
    n_sig = wilcoxon_df["significant"].sum() if not wilcoxon_df.empty else 0
    print(f"   {n_sig}/{len(wilcoxon_df)} comparisons significant after correction")

    print(f"\n── Bootstrap CIs ({args.n_boot} resamples) ──")
    boot_df = bootstrap_cis(df, args.n_boot, args.alpha)

    # Write outputs
    report = format_report(desc, shapiro, friedman, wilcoxon_df, boot_df, args.alpha)

    report_path = out_dir / "full_report.txt"
    report_path.write_text(report)
    print(f"\nFull report  : {report_path}")

    desc.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"Summary CSV  : {out_dir / 'summary_table.csv'}")

    latex = to_latex(desc, wilcoxon_df, boot_df)
    (out_dir / "summary_table.tex").write_text(latex)
    print(f"LaTeX table  : {out_dir / 'summary_table.tex'}")

    wilcoxon_df.to_csv(out_dir / "pairwise_tests.csv", index=False)
    boot_df.to_csv(out_dir / "bootstrap_cis.csv", index=False)

    # Print the report to stdout as well
    print("\n")
    print(report)

    # Weight sensitivity analysis
    if args.sensitivity:
        print("\n── Weight sensitivity analysis ──")
        sens_df = weight_sensitivity_analysis(df, args.alpha)
        if sens_df.empty:
            print("   Skipped (need ≥ 2 algorithms)")
        else:
            sens_path = out_dir / "sensitivity_analysis.csv"
            sens_df.to_csv(sens_path, index=False)
            print(f"   Saved to {sens_path}")

            configs_tested = sens_df["weight_config"].nunique()
            all_winners = sens_df.groupby("pair")["winner"].nunique()
            stable = all(v == 1 for v in all_winners)

            print(f"   Configurations tested: {configs_tested}")
            if stable:
                print("   Result: algorithm rankings are WEIGHT-INVARIANT "
                      "across all tested configurations.")
            else:
                unstable_pairs = [p for p, v in all_winners.items() if v > 1]
                print(f"   Result: rankings VARY for {unstable_pairs}")
                print("   Inspect sensitivity_analysis.csv for details.")

            print("\n   Per-config summary:")
            for config_name in WEIGHT_CONFIGS:
                sub = sens_df[sens_df["weight_config"] == config_name]
                if sub.empty:
                    continue
                summary_parts = []
                for _, r in sub.iterrows():
                    summary_parts.append(
                        f"{r['pair']}: {r['winner']} (A={r['vda_A']:.3f})"
                    )
                print(f"     {config_name}: {', '.join(summary_parts)}")

    # Multi-Jain ablation
    if args.ablation:
        print("\n── Multi-Jain ablation (bottleneck vs optimistic JFI) ──")
        result = multi_jain_ablation(df, args.alpha)
        if isinstance(result, tuple) and len(result) == 4:
            ab_df, rank_changes, total_seeds, gap_summary = result
            if not ab_df.empty:
                ab_path = out_dir / "ablation_multi_jain.csv"
                ab_df.to_csv(ab_path, index=False)
                print(f"   Saved to {ab_path}")
                print(f"   Rank changes: {rank_changes}/{total_seeds} seeds "
                      f"({100*rank_changes/max(1,total_seeds):.1f}%)")
                print(f"\n   Per-algorithm JFI dimension gap (max - min):")
                print(gap_summary.to_string(header=True))
                print(f"\n   Median composite scores by strategy:")
                for _, r in ab_df.iterrows():
                    print(f"     {r['strategy']:12s}  {r['algorithm']:<6s}  "
                          f"median={r['median']:.4f}")
            else:
                print("   Skipped (insufficient data)")
        else:
            print("   Skipped (jfi_resources/jfi_influence columns missing)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
