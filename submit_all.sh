#!/bin/bash
#SBATCH --job-name=ti4_full_suite
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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
BUDGETS="1000,5000,10000,25000,50000,100000"
ALGORITHMS="rs,hc,sa,nsga2,ts"
PLAYERS=6

TUNING_TRIALS=50
TUNING_SEEDS=15
TUNING_BASE_SEED=9000

LISA_VALIDATION_SEEDS=30
LISA_VALIDATION_ITER=10000
LISA_PERMS=999

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
echo "--- Phase 1c: Tuning TS (${TUNING_TRIALS} trials, ${TUNING_SEEDS} seeds) ---"
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
read SA_RATE SA_MIN_TEMP NSGA_BLOB NSGA_MUT NSGA_WARM TS_TENURE < <(
$PYTHON_BIN -c "
import json, glob, sys

params = {}
for path in sorted(glob.glob('${OUTPUT_DIR}/optuna_*/best_params.json')):
    with open(path) as f:
        d = json.load(f)
    params[d['algorithm']] = d['best_params']

for algo in ('sa', 'nsga2', 'ts'):
    if algo not in params:
        print(f'FATAL: No {algo.upper()} tuning results found', file=sys.stderr)
        sys.exit(1)

sa = params['sa']
ng = params['nsga2']
ts = params['ts']

print(
    sa['initial_acceptance_rate'],
    sa['min_temp'],
    ng['blob_fraction'],
    ng['mutation_rate'],
    ng['warm_fraction'],
    ts['tabu_tenure'],
)
"
)

echo "  SA:      rate=$SA_RATE  min_temp=$SA_MIN_TEMP"
echo "  NSGA-II: blob=$NSGA_BLOB  mut=$NSGA_MUT  warm=$NSGA_WARM"
echo "  TS:      tenure=$TS_TENURE"

# ── Phase 2: Saturation Study (1k → 100k evaluations, parallel) ──────────────
echo ""
echo "--- Phase 2: Saturation Benchmark (${SEEDS} seeds × ${BUDGETS}, ${WORKERS} workers) ---"
$PYTHON_BIN scripts/benchmark_engine.py \
    --seeds "$SEEDS" \
    --algorithms "$ALGORITHMS" \
    --budgets "$BUDGETS" \
    --players "$PLAYERS" \
    --workers "$WORKERS" \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --nsga-blob "$NSGA_BLOB" \
    --nsga-mut "$NSGA_MUT" \
    --nsga-warm "$NSGA_WARM" \
    --ts-tenure "$TS_TENURE" \
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
    --algorithms "hc,sa,nsga2,ts" \
    --sa-iter "$LISA_VALIDATION_ITER" \
    --hc-iter "$LISA_VALIDATION_ITER" \
    --ts-iter "$LISA_VALIDATION_ITER" \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --n-perms "$LISA_PERMS" \
    --workers "$WORKERS" \
    --output-dir "$OUTPUT_DIR"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "All phases complete."
echo "Output dir    : $OUTPUT_DIR"
echo "Results CSV   : $RESULTS_CSV"
echo "------------------------------------------------------------"
echo "Tuned SA:      rate=$SA_RATE  min_temp=$SA_MIN_TEMP"
echo "Tuned NSGA-II: blob=$NSGA_BLOB  mut=$NSGA_MUT  warm=$NSGA_WARM"
echo "Tuned TS:      tenure=$TS_TENURE"
echo "============================================================"
