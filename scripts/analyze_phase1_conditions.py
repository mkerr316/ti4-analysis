#!/usr/bin/env python3
"""
analyze_phase1_conditions.py — Statistical analysis for the five-condition ablation (Phase 1).

Reads the Phase 1 results.csv (SA only, five conditions, 100 seeds, 8 budgets),
filters to a single budget, aggregates chains, and runs:
  - Friedman omnibus test across all five conditions
  - Wilcoxon signed-rank + Holm-Bonferroni + Vargha-Delaney A for condition pairs
  - JFI parity check (one-sided Wilcoxon: is JFI in Cx < JFI in C0?)
  - Bootstrap 95% CI on median differences

Outputs:
  <csv_parent>/stats/phase1_condition_report.txt
  <csv_parent>/stats/phase1_condition_pairs.csv
  <csv_parent>/stats/phase1_jfi_parity.csv
  <csv_parent>/stats/phase1_summary.csv

Usage:
  python scripts/analyze_phase1_conditions.py \
      output/saturation_20260314_205919/benchmark_20260314_233002/results.csv \
      --budget 500000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Constants ──────────────────────────────────────────────────────────────────

CONDITION_ORDER = [
    "jfi_only",
    "moran_only",
    "lsap_only",
    "jfi_moran",
    "full_composite",
]

CONDITION_LABELS = {
    "jfi_only":       "C0 (JFI only)",
    "moran_only":     "C1 (Moran only)",
    "lsap_only":      "C2 (LSAP only)",
    "jfi_moran":      "C3 (JFI+Moran)",
    "full_composite": "C4 (Full composite)",
}

# Metrics to analyze
SPATIAL_METRICS = ["morans_i", "lisa_penalty"]
ALL_METRICS     = ["morans_i", "lisa_penalty", "jains_index",
                   "jfi_resources", "jfi_influence", "composite_score"]

# Pre-registered condition pairs for primary tests (C0 vs each other)
PRIMARY_PAIRS = [
    ("jfi_only", "moran_only"),
    ("jfi_only", "lsap_only"),
    ("jfi_only", "jfi_moran"),
    ("jfi_only", "full_composite"),
]

# Pre-specified predictions (True = condition expected to show MORE negative I / lower LSAP)
SPATIAL_IMPROVEMENT_EXPECTED = {
    "moran_only":     True,
    "lsap_only":      True,
    "jfi_moran":      True,
    "full_composite": True,
}

# JFI parity: C1 and C2 expected to FAIL (lower JFI than C0 = sacrifice)
# C3 and C4 expected to PASS (JFI no worse than C0)
JFI_PARITY_EXPECTED_PASS = {"jfi_moran", "full_composite"}
JFI_PARITY_EXPECTED_FAIL = {"moran_only", "lsap_only"}

# Pre-registered effect size threshold
VDA_THRESHOLD = 0.64

# ── Loading ────────────────────────────────────────────────────────────────────

def load(csv_path: Path, budget: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Filter to requested budget
    if "budget" in df.columns:
        available = sorted(df["budget"].unique())
        if budget not in available:
            raise ValueError(
                f"Budget {budget} not in CSV. Available: {available}"
            )
        df = df[df["budget"] == budget].copy()

    # Aggregate chains if present
    if "chain_id" in df.columns and df["chain_id"].nunique() > 1:
        n_chains = df["chain_id"].nunique()
        group_cols = [c for c in ["seed", "condition", "budget", "algorithm"]
                      if c in df.columns]
        numeric_cols = [c for c in df.select_dtypes(include="number").columns
                        if c not in ("seed", "chain_id", "budget", "front_size")]
        df = df.groupby(group_cols)[numeric_cols].mean().reset_index()
        print(f"  Aggregated {n_chains} chains → {len(df)} rows")

    # Validate conditions
    if "condition" not in df.columns:
        raise ValueError("CSV has no 'condition' column. Is this the Phase 1 CSV?")

    found = set(df["condition"].unique())
    expected = set(CONDITION_ORDER)
    missing = expected - found
    if missing:
        print(f"  WARNING: Missing conditions: {missing}")

    n_seeds = df["seed"].nunique()
    print(f"  Loaded {len(df)} rows: {n_seeds} seeds × {df['condition'].nunique()} conditions")
    return df


# ── Statistical helpers ────────────────────────────────────────────────────────

def vda(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A: P(x > y) + 0.5*P(x == y). A > 0.5 means x tends to be larger."""
    n1, n2 = len(x), len(y)
    r = 0.0
    for xi in x:
        r += np.sum(xi > y) + 0.5 * np.sum(xi == y)
    return r / (n1 * n2)


def vda_magnitude(a: float) -> str:
    d = abs(a - 0.5)
    if d < 0.06:  return "negligible"
    if d < 0.14:  return "small"
    if d < 0.21:  return "medium"
    return "large"


def holm_bonferroni(p_values: list) -> list:
    """Holm-Bonferroni correction. Returns corrected p-values in same order."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        running_max = max(running_max, adjusted)
        corrected[orig_idx] = min(running_max, 1.0)
    return corrected


def bootstrap_median_diff(x: np.ndarray, y: np.ndarray,
                          n_boot: int = 5000, seed: int = 42) -> tuple:
    """Bootstrap 95% CI on median(x) - median(y)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.median(bx) - np.median(by))
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.median(x) - np.median(y)), ci_lo, ci_hi


# ── Analysis ───────────────────────────────────────────────────────────────────

def get_paired(df: pd.DataFrame, cond_a: str, cond_b: str, metric: str):
    """Return paired arrays for seeds present in both conditions."""
    a = df[df["condition"] == cond_a][["seed", metric]].set_index("seed")[metric]
    b = df[df["condition"] == cond_b][["seed", metric]].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    return a.loc[common].values, b.loc[common].values


def run_friedman(df: pd.DataFrame, metric: str) -> dict:
    """Friedman test across all conditions."""
    groups = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond][metric].dropna().values
        if len(sub) > 0:
            groups.append(sub)
    if len(groups) < 3:
        return {"chi2": None, "p": None, "df": None}
    stat, p = stats.friedmanchisquare(*groups)
    return {"chi2": float(stat), "p": float(p), "df": len(groups) - 1}


def run_pairwise(df: pd.DataFrame, pairs: list, metrics: list,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Wilcoxon signed-rank + Holm-Bonferroni + VDA for all pairs × metrics."""
    rows = []
    # Collect raw p-values across ALL tests for Holm-Bonferroni
    all_tests = []
    for metric in metrics:
        for cond_a, cond_b in pairs:
            xa, xb = get_paired(df, cond_a, cond_b, metric)
            if len(xa) < 5:
                continue
            try:
                stat, p_raw = stats.wilcoxon(xa, xb)
            except ValueError:
                stat, p_raw = None, 1.0
            a_val = vda(xa, xb)
            med_diff, ci_lo, ci_hi = bootstrap_median_diff(xa, xb)
            all_tests.append({
                "metric": metric,
                "cond_a": cond_a,
                "cond_b": cond_b,
                "W": stat,
                "p_raw": p_raw,
                "vda_A": round(a_val, 4),
                "vda_mag": vda_magnitude(a_val),
                "median_a": float(np.median(xa)),
                "median_b": float(np.median(xb)),
                "median_diff": round(med_diff, 6),
                "ci_lo": round(ci_lo, 6),
                "ci_hi": round(ci_hi, 6),
                "n": len(xa),
            })

    # Apply Holm-Bonferroni across ALL tests simultaneously
    raw_ps = [t["p_raw"] for t in all_tests]
    corrected = holm_bonferroni(raw_ps)
    for t, p_corr in zip(all_tests, corrected):
        t["p_corrected"] = round(p_corr, 6)
        t["significant"] = p_corr < alpha
        t["practical"] = abs(t["vda_A"] - 0.5) >= 0.14  # medium or large
        rows.append(t)

    return pd.DataFrame(rows)


def run_jfi_parity(df: pd.DataFrame, baseline: str = "jfi_only") -> pd.DataFrame:
    """
    JFI parity check: one-sided Wilcoxon.
    H0: median(jains_index in Cx) >= median(jains_index in C0)
    Rejection = JFI sacrifice in condition Cx.
    Note: higher jains_index is better (1.0 = perfect equality).
    """
    rows = []
    for cond in CONDITION_ORDER:
        if cond == baseline:
            continue
        xa, xb = get_paired(df, cond, baseline, "jains_index")
        if len(xa) < 5:
            continue
        # One-sided: alternative='less' tests if Cx JFI < baseline JFI
        try:
            stat, p = stats.wilcoxon(xa, xb, alternative="less")
        except ValueError:
            stat, p = None, 1.0
        median_cx = float(np.median(xa))
        median_c0 = float(np.median(xb))
        sacrifice = p < 0.05
        expected_fail = cond in JFI_PARITY_EXPECTED_FAIL
        rows.append({
            "condition": cond,
            "label": CONDITION_LABELS.get(cond, cond),
            "median_jfi_cx": round(median_cx, 6),
            "median_jfi_c0": round(median_c0, 6),
            "W": stat,
            "p_one_sided": round(float(p), 6),
            "jfi_sacrifice": sacrifice,
            "expected_sacrifice": expected_fail,
            "prediction_correct": sacrifice == expected_fail,
            "n": len(xa),
        })
    return pd.DataFrame(rows)


# ── Reporting ──────────────────────────────────────────────────────────────────

def write_report(df_raw: pd.DataFrame, pairs_df: pd.DataFrame,
                 parity_df: pd.DataFrame, budget: int,
                 out_path: Path) -> None:
    lines = []
    sep = "=" * 88

    lines += [sep,
              "PHASE 1 — FIVE-CONDITION ABLATION STATISTICAL REPORT",
              f"Budget: {budget}  |  Pre-registered VDA threshold: A >= {VDA_THRESHOLD}",
              sep, ""]

    # Descriptive statistics
    lines += ["1. DESCRIPTIVE STATISTICS (Median [IQR])", "-" * 88]
    for metric in ALL_METRICS:
        lines.append(f"\n  {metric}:")
        for cond in CONDITION_ORDER:
            sub = df_raw[df_raw["condition"] == cond][metric].dropna()
            if len(sub) == 0:
                continue
            q25, med, q75 = np.percentile(sub, [25, 50, 75])
            label = CONDITION_LABELS.get(cond, cond)
            lines.append(
                f"    {label:30s}  median={med:+.4f}  "
                f"IQR=[{q25:+.4f}, {q75:+.4f}]  n={len(sub)}"
            )
    lines.append("")

    # Friedman
    lines += ["", "2. FRIEDMAN OMNIBUS TEST", "-" * 88]
    for metric in SPATIAL_METRICS + ["jains_index"]:
        r = run_friedman(df_raw, metric)
        if r["chi2"] is not None:
            lines.append(
                f"  {metric:20s}  chi2={r['chi2']:.4f}  df={r['df']}  "
                f"p={r['p']:.6f}  {'SIGNIFICANT' if r['p'] < 0.05 else 'ns'}"
            )
    lines.append("")

    # Pairwise tests
    lines += ["", "3. PAIRWISE WILCOXON SIGNED-RANK (Holm-Bonferroni across all tests)",
              f"   Pre-registered pairs: C0 vs C1, C0 vs C2, C0 vs C3, C0 vs C4",
              f"   Total tests corrected simultaneously: {len(pairs_df)}",
              "-" * 88]

    for metric in pairs_df["metric"].unique():
        lines.append(f"\n  {metric}:")
        sub = pairs_df[pairs_df["metric"] == metric]
        for _, row in sub.iterrows():
            la = CONDITION_LABELS.get(row["cond_a"], row["cond_a"])
            lb = CONDITION_LABELS.get(row["cond_b"], row["cond_b"])
            sig = "***" if row["significant"] else "   "
            lines.append(
                f"    {la} vs {lb}"
            )
            lines.append(
                f"      W={row['W']:.1f}  p_raw={row['p_raw']:.4f}  "
                f"p_corr={row['p_corrected']:.4f} {sig}  "
                f"VDA={row['vda_A']:.3f} ({row['vda_mag']})"
            )
            lines.append(
                f"      median_diff={row['median_diff']:+.4f}  "
                f"95%CI=[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  n={row['n']}"
            )
    lines.append("")

    # JFI parity
    lines += ["", "4. JFI PARITY CHECK (one-sided Wilcoxon, H0: Cx JFI >= C0 JFI)",
              "   Rejection = fairness sacrifice. Higher JFI is better.",
              "-" * 88]
    for _, row in parity_df.iterrows():
        result = "SACRIFICE (JFI loss)" if row["jfi_sacrifice"] else "PARITY (JFI maintained)"
        correct = "CORRECT" if row["prediction_correct"] else "INCORRECT"
        lines.append(
            f"  {row['label']:30s}  "
            f"median_cx={row['median_jfi_cx']:.6f}  median_c0={row['median_jfi_c0']:.6f}  "
            f"p={row['p_one_sided']:.4f}  → {result}  [{correct} prediction]"
        )
    lines.append("")

    # Pre-registered interpretation
    lines += ["", "5. PRE-REGISTERED INTERPRETATION", "-" * 88]
    lines.append("  Spatial constraint supported if EITHER spatial metric shows A >= 0.64")
    lines.append("  AND JFI parity passes for that condition pair.")
    lines.append("")

    for cond_b in ["moran_only", "lsap_only", "jfi_moran", "full_composite"]:
        label_b = CONDITION_LABELS.get(cond_b, cond_b)
        lines.append(f"  C0 vs {label_b}:")

        # Spatial metrics
        for metric in SPATIAL_METRICS:
            row = pairs_df[
                (pairs_df["metric"] == metric) &
                (pairs_df["cond_a"] == "jfi_only") &
                (pairs_df["cond_b"] == cond_b)
            ]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            practical = abs(row["vda_A"] - 0.5) >= 0.14
            lines.append(
                f"    {metric}: A={row['vda_A']:.3f} ({row['vda_mag']})  "
                f"p_corr={row['p_corrected']:.4f}  "
                f"{'MEETS threshold' if row['vda_A'] >= VDA_THRESHOLD else 'below threshold'}"
            )

        # JFI parity
        parity_row = parity_df[parity_df["condition"] == cond_b]
        if len(parity_row) > 0:
            pr = parity_row.iloc[0]
            parity_result = "PASS (no sacrifice)" if not pr["jfi_sacrifice"] else "FAIL (sacrifice)"
            lines.append(f"    JFI parity: {parity_result}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Phase 1 results.csv path")
    parser.add_argument("--budget", type=int, default=500000,
                        help="Budget level to analyze (default: 500000)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    out_dir = args.csv.parent / "stats"
    out_dir.mkdir(exist_ok=True)

    print(f"Phase 1 Five-Condition Analysis")
    print(f"  CSV:    {args.csv}")
    print(f"  Budget: {args.budget}")
    print(f"  Output: {out_dir}")
    print()

    df = load(args.csv, args.budget)

    # Run pairwise tests across primary pairs and spatial + JFI metrics
    print("Running pairwise Wilcoxon tests...")
    pairs_df = run_pairwise(df, PRIMARY_PAIRS, SPATIAL_METRICS + ["jains_index"],
                            alpha=args.alpha)

    # JFI parity check
    print("Running JFI parity checks...")
    parity_df = run_jfi_parity(df)

    # Summary statistics
    summary_rows = []
    for metric in ALL_METRICS:
        for cond in CONDITION_ORDER:
            sub = df[df["condition"] == cond][metric].dropna()
            if len(sub) == 0:
                continue
            q25, med, q75 = np.percentile(sub, [25, 50, 75])
            summary_rows.append({
                "condition": cond,
                "label": CONDITION_LABELS.get(cond, cond),
                "metric": metric,
                "median": round(float(med), 6),
                "q25": round(float(q25), 6),
                "q75": round(float(q75), 6),
                "mean": round(float(sub.mean()), 6),
                "std": round(float(sub.std()), 6),
                "n": len(sub),
            })
    summary_df = pd.DataFrame(summary_rows)

    # Write outputs
    pairs_df.to_csv(out_dir / "phase1_condition_pairs.csv", index=False)
    parity_df.to_csv(out_dir / "phase1_jfi_parity.csv", index=False)
    summary_df.to_csv(out_dir / "phase1_summary.csv", index=False)
    print(f"CSVs → {out_dir}")

    write_report(df, pairs_df, parity_df, args.budget,
                 out_dir / "phase1_condition_report.txt")

    # Print key findings to console
    print()
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    for cond_b in ["moran_only", "lsap_only", "jfi_moran", "full_composite"]:
        label_b = CONDITION_LABELS.get(cond_b, cond_b)
        lisa_row = pairs_df[
            (pairs_df["metric"] == "lisa_penalty") &
            (pairs_df["cond_a"] == "jfi_only") &
            (pairs_df["cond_b"] == cond_b)
        ]
        mi_row = pairs_df[
            (pairs_df["metric"] == "morans_i") &
            (pairs_df["cond_a"] == "jfi_only") &
            (pairs_df["cond_b"] == cond_b)
        ]
        parity_row = parity_df[parity_df["condition"] == cond_b]
        print(f"\nC0 vs {label_b}:")
        if len(lisa_row):
            r = lisa_row.iloc[0]
            print(f"  LSAP: A={r['vda_A']:.3f} p_corr={r['p_corrected']:.4f} "
                  f"({'SIGNIFICANT' if r['significant'] else 'ns'})")
        if len(mi_row):
            r = mi_row.iloc[0]
            print(f"  Moran's I: A={r['vda_A']:.3f} p_corr={r['p_corrected']:.4f} "
                  f"({'SIGNIFICANT' if r['significant'] else 'ns'})")
        if len(parity_row):
            pr = parity_row.iloc[0]
            print(f"  JFI parity: {'PASS' if not pr['jfi_sacrifice'] else 'FAIL'} "
                  f"(p={pr['p_one_sided']:.4f})")


if __name__ == "__main__":
    main()
