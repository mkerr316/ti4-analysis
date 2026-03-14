#!/bin/bash
#SBATCH --job-name=ti4_full_suite
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

set -euo pipefail

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Script failed with exit code $exit_code at line $1" >&2
    fi
    echo "Finished at: $(date -Iseconds)  exit_code=$exit_code"
}
trap 'cleanup $LINENO' EXIT

cd "$SLURM_SUBMIT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────
ENV_PATH="/home/mpk15956/miniforge3/envs/ti4-analysis"
PYTHON_BIN="$ENV_PATH/bin/python3"
RUN_TAG="saturation_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="output/${RUN_TAG}"
WORKERS=16

SEEDS=100
BUDGETS="1000,5000,10000,25000,50000,100000,200000,500000"
ALGORITHMS="rs,hc,sa,sga,nsga2,ts"
PLAYERS=6

TUNING_TRIALS=50
TUNING_SEEDS=100
TUNING_BASE_SEED=9000

LISA_VALIDATION_SEEDS=30
LISA_VALIDATION_ITER=10000
LISA_PERMS=9999

DIST_SENSITIVITY_SEEDS=50
DIST_SENSITIVITY_BUDGET=10000

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "============================================================"
echo "TI4 Full Benchmark Suite"
echo "============================================================"
echo "Run tag       : $RUN_TAG"
echo "Output dir    : $OUTPUT_DIR"
echo "Interpreter   : $PYTHON_BIN"
echo "Algorithms    : $ALGORITHMS"
echo "Seeds         : $SEEDS"
echo "Budgets       : $BUDGETS"
echo "Tuning seeds  : $TUNING_SEEDS"
echo "Workers       : $WORKERS"
echo "SLURM Job ID  : ${SLURM_JOB_ID:-local}"
echo "Git hash      : $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Started at    : $(date -Iseconds)"
echo "------------------------------------------------------------"
$PYTHON_BIN --version
$PYTHON_BIN -c "
import numpy, scipy, matplotlib, seaborn
print(f'  numpy      {numpy.__version__}')
print(f'  scipy      {scipy.__version__}')
print(f'  matplotlib {matplotlib.__version__}')
print(f'  seaborn    {seaborn.__version__}')
try:
    import optuna; print(f'  optuna     {optuna.__version__}')
except ImportError:
    print('  optuna     NOT INSTALLED')
"
echo "============================================================"

# ── Phase 1: Hyperparameter Tuning (disjoint seeds 9000+) ────────────────────
echo ""
echo "--- Phase 1a: Tuning SA (${TUNING_TRIALS} trials, ${TUNING_SEEDS} seeds) ---"
$PYTHON_BIN scripts/optimize_hyperparameters.py \
    --algo sa \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --players "$PLAYERS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "--- Phase 1b: Tuning NSGA-II (${TUNING_TRIALS} trials, ${TUNING_SEEDS} seeds) ---"
$PYTHON_BIN scripts/optimize_hyperparameters.py \
    --algo nsga2 \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --players "$PLAYERS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "--- Phase 1c: Tuning SGA (${TUNING_TRIALS} trials, ${TUNING_SEEDS} seeds) ---"
$PYTHON_BIN scripts/optimize_hyperparameters.py \
    --algo sga \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --players "$PLAYERS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "--- Phase 1d: Tuning TS (${TUNING_TRIALS} trials, ${TUNING_SEEDS} seeds) ---"
$PYTHON_BIN scripts/optimize_hyperparameters.py \
    --algo ts \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --players "$PLAYERS" \
    --output-dir "$OUTPUT_DIR"

# ── Extract tuned hyperparameters (fail-loud, no silent defaults) ─────────────
echo ""
echo "--- Extracting tuned hyperparameters ---"
read SA_RATE SA_MIN_TEMP NSGA_BLOB NSGA_MUT NSGA_WARM SGA_BLOB SGA_MUT SGA_WARM TS_K < <(
$PYTHON_BIN -c "
import json, glob, sys

params = {}
for path in sorted(glob.glob('${OUTPUT_DIR}/optuna_*/best_params.json')):
    with open(path) as f:
        d = json.load(f)
    params[d['algorithm']] = d['best_params']

for algo in ('sa', 'nsga2', 'sga', 'ts'):
    if algo not in params:
        print(f'FATAL: No {algo.upper()} tuning results found', file=sys.stderr)
        sys.exit(1)

sa = params['sa']
ng = params['nsga2']
sg = params['sga']
ts = params['ts']

print(
    sa['initial_acceptance_rate'],
    sa['min_temp'],
    ng['blob_fraction'],
    ng['mutation_rate'],
    ng['warm_fraction'],
    sg['blob_fraction'],
    sg['mutation_rate'],
    sg['warm_fraction'],
    ts['tabu_tenure_coefficient'],
)
"
)

echo "  SA:      rate=$SA_RATE  min_temp=$SA_MIN_TEMP"
echo "  NSGA-II: blob=$NSGA_BLOB  mut=$NSGA_MUT  warm=$NSGA_WARM"
echo "  SGA:     blob=$SGA_BLOB  mut=$SGA_MUT  warm=$SGA_WARM"
echo "  TS:      k=$TS_K"

# ── Phase 2: Saturation Study (1k → 500k evaluations, parallel) ──────────────
echo ""
echo "--- Phase 2: Saturation Benchmark (${SEEDS} seeds × ${BUDGETS}, ${WORKERS} workers) ---"
$PYTHON_BIN scripts/benchmark_engine.py \
    --seeds "$SEEDS" \
    --algorithms "$ALGORITHMS" \
    --budgets "$BUDGETS" \
    --players "$PLAYERS" \
    --workers "$WORKERS" \
    --chains 3 \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --nsga-blob "$NSGA_BLOB" \
    --nsga-mut "$NSGA_MUT" \
    --nsga-warm "$NSGA_WARM" \
    --sga-blob "$SGA_BLOB" \
    --sga-mut "$SGA_MUT" \
    --sga-warm "$SGA_WARM" \
    --ts-k "$TS_K" \
    --output-dir "$OUTPUT_DIR"

# ── Locate results CSV ───────────────────────────────────────────────────────
RESULTS_CSV=$(find "$OUTPUT_DIR" -name "results.csv" -path "*/benchmark_*" | sort | tail -1)

if [ -z "$RESULTS_CSV" ]; then
    echo "ERROR: results.csv not found in $OUTPUT_DIR" >&2
    exit 1
fi
echo "Results CSV   : $RESULTS_CSV"

# ── Phase 3: Statistical Analysis ────────────────────────────────────────────
echo ""
echo "--- Phase 3a: Non-parametric analysis + sensitivity + ablation ---"
$PYTHON_BIN scripts/analyze_benchmark.py "$RESULTS_CSV" --sensitivity --ablation

echo ""
echo "--- Phase 3b: Publication figures ---"
$PYTHON_BIN scripts/plot_statistical_results.py "$RESULTS_CSV"

# ── Phase 4: Post-hoc LISA Validation (10k budget, parallel) ─────────────────
echo ""
echo "--- Phase 4: LISA proxy validation (${LISA_VALIDATION_SEEDS} seeds, ${LISA_PERMS} perms, ${WORKERS} workers) ---"
$PYTHON_BIN scripts/validate_lisa_proxy.py \
    --seeds "$LISA_VALIDATION_SEEDS" \
    --algorithms "hc,sa,sga,nsga2,ts" \
    --sa-iter "$LISA_VALIDATION_ITER" \
    --hc-iter "$LISA_VALIDATION_ITER" \
    --ts-iter "$LISA_VALIDATION_ITER" \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --n-perms "$LISA_PERMS" \
    --workers "$WORKERS" \
    --output-dir "$OUTPUT_DIR"

# ── Phase 5: Distance-Weight Sensitivity Analysis ────────────────────────────
echo ""
echo "--- Phase 5: Distance-weight sensitivity (${DIST_SENSITIVITY_SEEDS} seeds, ${WORKERS} workers) ---"
$PYTHON_BIN scripts/distance_weight_sensitivity.py \
    --seeds "$DIST_SENSITIVITY_SEEDS" \
    --budget "$DIST_SENSITIVITY_BUDGET" \
    --algorithms "sa,sga" \
    --workers "$WORKERS" \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --sga-blob "$SGA_BLOB" \
    --sga-mut "$SGA_MUT" \
    --sga-warm "$SGA_WARM" \
    --output-dir "$OUTPUT_DIR"

# ── Phase 6: Track B Quality Indicators (NSGA-II Pareto fronts) ──────────────
ARCHIVE_DIR=$(find "$OUTPUT_DIR" -type d -name "pareto_archives" | head -1)

if [ -n "$ARCHIVE_DIR" ]; then
    echo ""
    echo "--- Phase 6: Track B quality indicators (HV, IGD+, Spacing) ---"
    $PYTHON_BIN scripts/quality_indicators.py \
        --archive-dir "$ARCHIVE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --plot
else
    echo ""
    echo "--- Phase 6: SKIPPED (no pareto_archives directory found) ---"
fi

# ── Phase 6b: Cross-method IGD (scalar terminal states vs reference front) ───
if [ -n "$ARCHIVE_DIR" ]; then
    RUN_DIR=$(dirname "$ARCHIVE_DIR")
    if [ -f "$RUN_DIR/results.csv" ]; then
        echo ""
        echo "--- Phase 6b: Cross-method IGD (scalar vs Pareto reference) ---"
        $PYTHON_BIN scripts/cross_method_igd.py --run-dir "$RUN_DIR" --report
    else
        echo ""
        echo "--- Phase 6b: SKIPPED (no results.csv in run dir) ---"
    fi
else
    echo ""
    echo "--- Phase 6b: SKIPPED (no pareto_archives directory) ---"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "All phases complete."
echo "Output dir    : $OUTPUT_DIR"
echo "Results CSV   : $RESULTS_CSV"
echo "------------------------------------------------------------"
echo "Tuned SA:      rate=$SA_RATE  min_temp=$SA_MIN_TEMP"
echo "Tuned NSGA-II: blob=$NSGA_BLOB  mut=$NSGA_MUT  warm=$NSGA_WARM"
echo "Tuned SGA:     blob=$SGA_BLOB  mut=$SGA_MUT  warm=$SGA_WARM"
echo "Tuned TS:      k=$TS_K"
echo "============================================================"
