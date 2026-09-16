#!/usr/bin/env bash
# ============================================================================
# measure_nystromcir.sh -- Circuit proving (Groth16+Lasso+CP-Link)
#
# Runs ZK-SNARK proving for Nystrom attention circuits.
# Witnesses must already exist (run measure_nystrompy.sh first).
#
# Sections:
#   1. Nystrom Freivalds    2. Nystrom Naive
#
# Usage:
#   cd circuits/
#   bash measure_nystromcir.sh [options]
#
# Options:
#   --variant V    Run only variant V (freivalds or naive; default: both)
#   --seq N        Run only sequence length N
#   --log-dir D    Log output directory (default: logs/results_sweep/tiny)
#   --force        Re-run even if results exist
#   --dry-run      Print execution plan without running
#   --help         Show this help
#
# Environment variables:
#   CIRCUIT_DIR    Circuit crate root (default: this script's directory)
#   DATA_BASE      Witness data root
#   LOG_DIR        Log output directory
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CIRCUIT_DIR="${CIRCUIT_DIR:-$SCRIPT_DIR}"
DATA_BASE="${DATA_BASE:-$(cd "$SCRIPT_DIR/../models/Nystromformer/code/dietformer" 2>/dev/null && pwd || echo "")}"

# macOS /usr/bin/time
TIME_CMD="/usr/bin/time -l"

# -- Constants -----------------------------------------------------------------
SEQ_LENS="128 256 512 1024 2048 4096"

HIDDEN=128
HEADS=2
LAYERS=2
NUM_LANDMARKS=32

# -- Parse arguments -----------------------------------------------------------
FILTER_VARIANT=""
FILTER_SEQ=""
FLAG_LOG_DIR=""
FORCE=false
DRY_RUN=false

print_help() {
    awk '/^# ===/{n++} n==1{sub(/^# ?/,""); print}' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)  FILTER_VARIANT="$2"; shift ;;
        --seq)      FILTER_SEQ="$2"; shift ;;
        --log-dir)  FLAG_LOG_DIR="$2"; shift ;;
        --force)    FORCE=true ;;
        --dry-run)  DRY_RUN=true ;;
        --help|-h)  print_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [[ -n "$FLAG_LOG_DIR" ]]; then
    LOG_DIR="$FLAG_LOG_DIR"
else
    LOG_DIR="${LOG_DIR:-logs/results_sweep/tiny}"
fi

cd "$CIRCUIT_DIR"
mkdir -p "$LOG_DIR"

# -- Helper: witness directory name --------------------------------------------
witness_dir() {
    local n="$1"
    echo "outputs_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}_nystrom"
}

# -- Helper: cell tag ----------------------------------------------------------
cell_tag() {
    local n="$1" variant="${2:-freivalds}"
    echo "nystrom_attn_${variant}_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}"
}

# -- Helper: check if cell result exists ---------------------------------------
result_exists() {
    local tag="$1"
    local latest
    latest=$(ls -t "$LOG_DIR"/prove_${tag}_*.log 2>/dev/null | head -1)
    if [[ -n "$latest" ]] && grep -q "BENCHMARK_RESULT" "$latest" 2>/dev/null; then
        return 0
    fi
    return 1
}

# -- Measurement function ------------------------------------------------------
# run_prove WITNESS_VAR DATA_DIR TEST_FN TAG
run_prove() {
    local WITNESS_VAR="$1"
    local DATA_DIR="$2"
    local TEST_FN="$3"
    local TAG="$4"

    local FULL_DATA_PATH="$DATA_BASE/$DATA_DIR"
    local TIMESTAMP
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    local LOG_FILE="$LOG_DIR/prove_${TAG}_${TIMESTAMP}.log"
    local TIME_FILE="$LOG_DIR/prove_${TAG}_${TIMESTAMP}.time"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    echo "============================================"
    echo "[$(date +%H:%M:%S)] #${TOTAL_RUNS} $TAG"
    echo "  Test:  $TEST_FN"
    echo "  Data:  $DATA_DIR"
    echo "  Log:   $LOG_FILE"
    echo "  Time:  $TIME_FILE"
    echo "============================================"

    if [ ! -d "$FULL_DATA_PATH" ]; then
        SKIP_RUNS=$((SKIP_RUNS + 1))
        echo "  [SKIP] Data dir not found: $FULL_DATA_PATH"
        echo ""
        return 0
    fi

    if [[ "$FORCE" != true ]] && result_exists "$TAG"; then
        SKIP_RUNS=$((SKIP_RUNS + 1))
        echo "  [SKIP] Result exists for $TAG"
        echo ""
        return 0
    fi

    local EXIT_CODE=0
    if env "$WITNESS_VAR=$FULL_DATA_PATH" \
        $TIME_CMD \
        cargo test --release "$TEST_FN" -- --nocapture \
        > "$LOG_FILE" 2> "$TIME_FILE"; then
        PASS_RUNS=$((PASS_RUNS + 1))
        echo "  [PASS] $TAG"
    else
        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 137 ]] || grep -qi "killed" "$TIME_FILE" 2>/dev/null; then
            OOM_RUNS=$((OOM_RUNS + 1))
        fi
        FAIL_RUNS=$((FAIL_RUNS + 1))
        echo "  [FAIL] $TAG (exit code: $EXIT_CODE)"
    fi

    if [ -f "$TIME_FILE" ]; then
        local PEAK_RSS
        PEAK_RSS=$(grep "maximum resident set size" "$TIME_FILE" | awk '{print $1}' || echo "N/A")
        local PEAK_RSS_MB
        if [ "$PEAK_RSS" != "N/A" ] && [ "$PEAK_RSS" -gt 0 ] 2>/dev/null; then
            PEAK_RSS_MB=$((PEAK_RSS / 1048576))
        else
            PEAK_RSS_MB="N/A"
        fi
        echo "  Peak RSS: ${PEAK_RSS} bytes (${PEAK_RSS_MB} MB)"
    fi

    local LAYER_COUNT
    LAYER_COUNT=$(grep -c "BENCHMARK_RESULT" "$LOG_FILE" 2>/dev/null || true)
    : "${LAYER_COUNT:=0}"
    echo "  Layers proved: $LAYER_COUNT"
    echo ""
}

# -- Helper: should run variant? -----------------------------------------------
should_run_variant() {
    local v="$1"
    [[ -z "$FILTER_VARIANT" ]] || [[ "$FILTER_VARIANT" == "$v" ]]
}

# -- Start time ----------------------------------------------------------------
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo " measure_nystromcir.sh (Circuit Proving)"
echo " Started:     $STARTED_AT"
echo " Circuit dir: $CIRCUIT_DIR"
echo " Data base:   $DATA_BASE"
echo " Log dir:     $LOG_DIR"
echo " Config:      L=$LAYERS, h=$HIDDEN, H=$HEADS"
echo " Nystrom:     m=$NUM_LANDMARKS"
echo " Variant:     ${FILTER_VARIANT:-both}"
echo "============================================"
echo ""

# -- Counters ------------------------------------------------------------------
TOTAL_RUNS=0
PASS_RUNS=0
FAIL_RUNS=0
SKIP_RUNS=0
OOM_RUNS=0

# -- Dry-run -------------------------------------------------------------------
if [[ "$DRY_RUN" == true ]]; then
    DRY_N=0
    for N in $SEQ_LENS; do
        if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
            continue
        fi
        for VARIANT in freivalds naive; do
            if ! should_run_variant "$VARIANT"; then continue; fi
            DRY_N=$((DRY_N + 1))
            TAG="$(cell_tag "$N" "$VARIANT")"
            DIR="$(witness_dir "$N")"
            echo "[dry-run] #${DRY_N} $TAG"
            echo "  data: $DATA_BASE/$DIR"
            echo "  log:  $LOG_DIR/prove_${TAG}_YYYYMMDD_HHMMSS.log"
            case "$VARIANT" in
                freivalds)
                    echo "  env:  NYSTROM_ATTN_WITNESS_DIR"
                    echo "  test: nystrom_attn_hf_single_layer_test" ;;
                naive)
                    echo "  env:  NAIVE_NYSTROM_ATTN_WITNESS_DIR"
                    echo "  test: nystrom_attn_naive_single_layer_test" ;;
            esac
            echo ""
        done
    done
    echo "Total circuit jobs: $DRY_N"
    exit 0
fi

# -- cargo build (release, once) -----------------------------------------------
echo "=== [BUILD] cargo test --release --no-run ==="
cargo test --release --no-run 2>&1 | tail -5
echo ""

# ============================================================================
# === 1. Nystrom Attention (Freivalds) =======================================
# ============================================================================
if should_run_variant "freivalds"; then
    echo ""
    echo "################################################################"
    echo "# 1. Nystrom Attention (Freivalds)"
    echo "################################################################"
    for N in $SEQ_LENS; do
        if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
            continue
        fi
        run_prove "NYSTROM_ATTN_WITNESS_DIR" \
            "$(witness_dir "$N")" \
            "nystrom_attn_hf_single_layer_test" \
            "$(cell_tag "$N")"
    done
fi

# ============================================================================
# === 2. Nystrom Attention (Naive) ===========================================
# ============================================================================
if should_run_variant "naive"; then
    echo ""
    echo "################################################################"
    echo "# 2. Nystrom Attention (Naive)"
    echo "################################################################"
    for N in $SEQ_LENS; do
        if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
            continue
        fi
        run_prove "NAIVE_NYSTROM_ATTN_WITNESS_DIR" \
            "$(witness_dir "$N")" \
            "nystrom_attn_naive_single_layer_test" \
            "$(cell_tag "$N" naive)"
    done
fi

# ============================================================================
# === 종합 결과 ==============================================================
# ============================================================================
ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================"
echo " measure_nystromcir.sh 완료"
echo " 시작: $STARTED_AT"
echo " 종료: $ENDED_AT"
echo " 총 실행: $TOTAL_RUNS"
echo " 성공:    $PASS_RUNS"
echo " 실패:    $FAIL_RUNS"
echo " 스킵:    $SKIP_RUNS"
echo " OOM:     $OOM_RUNS"
echo "============================================"
echo ""
echo "=== 결과 요약 명령 ==="
echo ""
echo "# per-layer 벤치마크 (JSON):"
echo "  grep BENCHMARK_RESULT $LOG_DIR/prove_*.log"
echo ""
echo "# peak RSS (메모리, bytes):"
echo "  grep 'maximum resident set size' $LOG_DIR/prove_*.time"
echo ""
echo "# 전체 wall time per run:"
echo "  grep 'total_wall_ms' $LOG_DIR/prove_*.log"
echo ""
echo "# 실패한 run 확인:"
echo "  grep '\\[FAIL\\]' $LOG_DIR/prove_*.log"
echo ""
