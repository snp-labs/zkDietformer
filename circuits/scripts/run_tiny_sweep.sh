#!/usr/bin/env bash
# ============================================================================
# run_tiny_sweep.sh -- Linformer/SOFT attention circuit measurement sweep
#
# Folders: outputs_n{128..4096}_{k32_,m32_}L2_h128_{linformer,soft}
# Circuits: linformer_attn_freivalds -> soft_attn_freivalds
#
# Environment: macOS (Darwin, ARM64 Mac Studio, 128 GB)
# Purpose: SSH + nohup background execution
#
# Usage:
#   cd circuits/
#   mkdir -p logs/results_sweep/tiny
#   nohup caffeinate -i bash scripts/run_tiny_sweep.sh \
#     > logs/results_sweep/tiny/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo "PID: $!"
#
# Results:
#   tail -f logs/results_sweep/tiny/master_*.log
#   grep BENCHMARK_RESULT logs/results_sweep/tiny/prove_*.log
#   grep "maximum resident set size" logs/results_sweep/tiny/prove_*.time
#
# Options:
#   --witness-only   Generate witnesses only (no circuit measurement)
#   --circuit-only   Run circuit measurement only (witnesses must exist)
#   --kernel K       Run only kernel K (linformer or soft)
#   --seq N          Run only sequence length N
#   --force          Re-run even if results exist
#   --dry-run        Print execution plan without running
#   --help           Show this help
#
# Environment variables (override defaults):
#   CIRCUIT_DIR   Circuit crate root
#   DATA_BASE     Witness data root
#   LOG_DIR       Log output directory (default: logs/results_sweep/tiny)
#   PYTHON_BIN    Python interpreter path
# ============================================================================
set -uo pipefail
# NOTE: -e not used -- continue after individual test failures

# -- Path setup ---------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CIRCUIT_DIR="${CIRCUIT_DIR:-$REPO_ROOT}"
DATA_BASE="${DATA_BASE:-$(cd "$REPO_ROOT/../models/Nystromformer/code/dietformer" 2>/dev/null && pwd || echo "")}"
LOG_DIR="${LOG_DIR:-logs/results_sweep/tiny}"

cd "$CIRCUIT_DIR"
mkdir -p "$LOG_DIR"

# macOS /usr/bin/time (absolute path, not shell builtin)
TIME_CMD="/usr/bin/time -l"

# -- Python resolution --------------------------------------------------------
resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "$PYTHON_BIN"; return
    fi
    if command -v conda &>/dev/null; then
        local conda_py
        conda_py="$(conda run -n dietformer python -c \
            'import sys; print(sys.executable)' 2>/dev/null)" || true
        if [[ -n "$conda_py" ]] && [[ -x "$conda_py" ]]; then
            echo "$conda_py"; return
        fi
    fi
    if command -v python3 &>/dev/null; then
        command -v python3; return
    fi
    echo ""
}

PYTHON="$(resolve_python)"

# -- Constants -----------------------------------------------------------------
SOFT_EPS1=8037
SOFT_EPS3=99
SOFT_EPS4=144

SEQ_LENS="128 256 512 1024 2048 4096"

INPUT_TEXT="The capital of France is [MASK]."
INPUT_SEED=42

# Tiny config
HIDDEN=128
HEADS=2
LAYERS=2
PROJ_DIM=32          # Linformer k
NUM_LANDMARKS=32     # SOFT m
N_ITER=20            # Newton-Raphson T

# -- Parse arguments -----------------------------------------------------------
DO_WITNESS=true
DO_CIRCUIT=true
FILTER_KERNEL=""
FILTER_SEQ=""
FORCE=false
DRY_RUN=false

print_help() {
    awk '/^# ===/{n++} n==1{sub(/^# ?/,""); print}' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --witness-only) DO_WITNESS=true; DO_CIRCUIT=false ;;
        --circuit-only) DO_WITNESS=false; DO_CIRCUIT=true ;;
        --kernel) FILTER_KERNEL="$2"; shift ;;
        --seq)    FILTER_SEQ="$2"; shift ;;
        --force)  FORCE=true ;;
        --dry-run) DRY_RUN=true ;;
        --help|-h) print_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# -- Start time ----------------------------------------------------------------
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo " run_tiny_sweep.sh started: $STARTED_AT"
echo " Circuit dir: $CIRCUIT_DIR"
echo " Data base:   $DATA_BASE"
echo " Log dir:     $LOG_DIR"
echo " Python:      ${PYTHON:-<not found>}"
echo " Config:      L=$LAYERS, h=$HIDDEN, H=$HEADS"
echo " Linformer:   k=$PROJ_DIM"
echo " SOFT:        m=$NUM_LANDMARKS, T=$N_ITER, eps1=$SOFT_EPS1, eps3=$SOFT_EPS3, eps4=$SOFT_EPS4"
echo "============================================"
echo ""

# -- Counters ------------------------------------------------------------------
TOTAL_RUNS=0
PASS_RUNS=0
FAIL_RUNS=0
SKIP_RUNS=0
OOM_RUNS=0
CONSEC_WIT_FAIL=0

# -- Helper: witness directory name --------------------------------------------
witness_dir() {
    local kernel="$1" n="$2"
    case "$kernel" in
        linformer) echo "outputs_n${n}_k${PROJ_DIM}_L${LAYERS}_h${HIDDEN}_linformer" ;;
        soft)      echo "outputs_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}_soft" ;;
    esac
}

# -- Helper: cell tag ----------------------------------------------------------
cell_tag() {
    local kernel="$1" n="$2"
    case "$kernel" in
        linformer) echo "linformer_attn_freivalds_n${n}_k${PROJ_DIM}_L${LAYERS}_h${HIDDEN}" ;;
        soft)      echo "soft_attn_freivalds_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}" ;;
    esac
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

# -- Preflight check -----------------------------------------------------------
preflight_check() {
    echo "=== [PREFLIGHT] Python environment ==="
    if [[ -z "$PYTHON" ]]; then
        echo "  [FATAL] No Python found. Set PYTHON_BIN or activate conda env."
        exit 1
    fi
    if ! "$PYTHON" --version &>/dev/null; then
        echo "  [FATAL] Python not executable: $PYTHON"; exit 1
    fi
    echo "  $("$PYTHON" --version 2>&1)"
    if ! "$PYTHON" -c "import torch; print(f'  torch {torch.__version__}')" 2>/dev/null; then
        echo "  [FATAL] torch not importable"; exit 1
    fi
    if ! "$PYTHON" -c "import transformers; print(f'  transformers {transformers.__version__}')" 2>/dev/null; then
        echo "  [FATAL] transformers not importable"; exit 1
    fi
    for s in run_linformer_validation.py run_soft_validation.py; do
        if [[ ! -f "$DATA_BASE/$s" ]]; then
            echo "  [FATAL] Missing: $DATA_BASE/$s"; exit 1
        fi
    done
    echo "  Scripts OK"
    echo ""
}

# -- Consecutive failure abort -------------------------------------------------
abort_consec_fail() {
    ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "============================================"
    echo " [ABORT] 2 consecutive witness failures"
    echo " Started: $STARTED_AT"
    echo " Aborted: $ENDED_AT"
    echo " Total:   $TOTAL_RUNS"
    echo " Pass:    $PASS_RUNS"
    echo " Fail:    $FAIL_RUNS"
    echo "============================================"
    exit 1
}

# -- Witness: eps override (SOFT only) ----------------------------------------
override_soft_eps() {
    local full_path="$1"
    echo "  [eps] Overriding eps: eps1=$SOFT_EPS1, eps3=$SOFT_EPS3, eps4=$SOFT_EPS4"
    local wit_files
    wit_files=$(find "$full_path" -name "*.witness.npz" \
        -not -name "*.norm.witness.npz" 2>/dev/null)
    if [[ -z "$wit_files" ]]; then
        echo "  [ERROR] No witness NPZ files found in $full_path"
        return 1
    fi
    local eps_ok=true
    while IFS= read -r wit; do
        "$PYTHON" -c "
import numpy as np, sys
w = dict(np.load('$wit', allow_pickle=True))
w['eps1'] = np.array($SOFT_EPS1, dtype=np.uint64)
w['eps3'] = np.array($SOFT_EPS3, dtype=np.uint64)
w['eps4'] = np.array($SOFT_EPS4, dtype=np.uint64)
np.savez_compressed('$wit', **w)
v = np.load('$wit', allow_pickle=True)
e1 = int(v['eps1']); e3 = int(v['eps3']); e4 = int(v['eps4'])
if e1 != $SOFT_EPS1 or e3 != $SOFT_EPS3 or e4 != $SOFT_EPS4:
    print(f'VERIFY_FAIL: eps1={e1}, eps3={e3}, eps4={e4}', file=sys.stderr)
    sys.exit(1)
print(f'  [ok] {\"$wit\".split(\"/\")[-1]}: eps1={e1}, eps3={e3}, eps4={e4}')
" 2>&1
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then eps_ok=false; fi
    done <<< "$wit_files"
    if [[ "$eps_ok" != true ]]; then
        echo "  [ERROR] Epsilon verification failed"; return 1
    fi
}

# -- Witness: token mask verification -----------------------------------------
verify_token_mask() {
    local full_path="$1" pattern="$2"
    echo "  [verify] Token mask:"
    "$PYTHON" -c "
import numpy as np, glob, os
pat = os.path.join('$full_path', '$pattern')
for f in sorted(glob.glob(pat)):
    if 'witness' not in f and 'gaussian' not in f and 'norm' not in f:
        d = np.load(f, allow_pickle=True)
        if 'mask' in d:
            mask = d['mask'][0]
            valid = int(np.sum(mask == 1))
            pad = int(np.sum(mask == 0))
            print(f'  valid={valid}, pad={pad}')
        break
" 2>&1
}

# -- Witness generation --------------------------------------------------------
gen_witness_linformer() {
    local n="$1"
    local dir_name
    dir_name="$(witness_dir linformer "$n")"
    local full_path="$DATA_BASE/$dir_name"

    echo "  [witness] Linformer n=$n -> $dir_name"
    if [[ -d "$full_path" ]] && [[ "$FORCE" != true ]]; then
        echo "  [skip] Witness exists: $dir_name"
        return 0
    fi

    "$PYTHON" "$DATA_BASE/run_linformer_validation.py" \
        --seq_len "$n" \
        --hidden_size "$HIDDEN" \
        --num_attention_heads "$HEADS" \
        --num_hidden_layers "$LAYERS" \
        --proj_dim "$PROJ_DIM" \
        --batch_size 1 --num_batch 1 \
        2>&1 | tee "$LOG_DIR/witness_linformer_n${n}.log"

    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "  [ERROR] Linformer witness gen failed (exit $rc)"
        return 1
    fi
    verify_token_mask "$full_path" "*.npz"
}

gen_witness_soft() {
    local n="$1"
    local dir_name
    dir_name="$(witness_dir soft "$n")"
    local full_path="$DATA_BASE/$dir_name"

    echo "  [witness] SOFT n=$n -> $dir_name"
    if [[ -d "$full_path" ]] && [[ "$FORCE" != true ]]; then
        echo "  [skip] Witness exists: $dir_name"
    else
        "$PYTHON" "$DATA_BASE/run_soft_validation.py" \
            --seq_len "$n" \
            --num_landmarks "$NUM_LANDMARKS" \
            --hidden_size "$HIDDEN" \
            --num_heads "$HEADS" \
            --num_layers "$LAYERS" \
            --n_iter "$N_ITER" \
            --seed "$INPUT_SEED" \
            --text "$INPUT_TEXT" \
            2>&1 | tee "$LOG_DIR/witness_soft_n${n}.log"

        local rc=${PIPESTATUS[0]}
        if [[ $rc -ne 0 ]]; then
            echo "  [ERROR] SOFT witness gen failed (exit $rc)"
            return 1
        fi
    fi

    override_soft_eps "$full_path" || return 1
    verify_token_mask "$full_path" "*batch0*.npz"
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

    # stdout -> prove log, stderr -> time log (peak RSS)
    local EXIT_CODE=0
    if env "$WITNESS_VAR=$FULL_DATA_PATH" \
        $TIME_CMD \
        cargo test --release "$TEST_FN" -- --nocapture \
        > "$LOG_FILE" 2> "$TIME_FILE"; then
        # SOFT: eps fallback check (stage 2)
        if [[ "$TAG" == *soft_attn* ]] && grep -q "\[fallback\]" "$TIME_FILE" 2>/dev/null; then
            FAIL_RUNS=$((FAIL_RUNS + 1))
            echo "  [FAIL] $TAG (eps loaded from fallback, not NPZ)"
        else
            PASS_RUNS=$((PASS_RUNS + 1))
            echo "  [PASS] $TAG"
        fi
    else
        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 137 ]] || grep -qi "killed" "$TIME_FILE" 2>/dev/null; then
            OOM_RUNS=$((OOM_RUNS + 1))
        fi
        FAIL_RUNS=$((FAIL_RUNS + 1))
        echo "  [FAIL] $TAG (exit code: $EXIT_CODE)"
    fi

    # peak RSS (macOS: bytes)
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

    # prove layer count
    local LAYER_COUNT
    LAYER_COUNT=$(grep -c "BENCHMARK_RESULT" "$LOG_FILE" 2>/dev/null || true)
    : "${LAYER_COUNT:=0}"
    echo "  Layers proved: $LAYER_COUNT"
    echo ""
}

# -- Dry-run -------------------------------------------------------------------
if [[ "$DRY_RUN" == true ]]; then
    DRY_N=0
    for KERNEL in linformer soft; do
        if [[ -n "$FILTER_KERNEL" ]] && [[ "$FILTER_KERNEL" != "$KERNEL" ]]; then
            continue
        fi
        for N in $SEQ_LENS; do
            if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
                continue
            fi
            DRY_N=$((DRY_N + 1))
            TAG="$(cell_tag "$KERNEL" "$N")"
            DIR="$(witness_dir "$KERNEL" "$N")"
            echo "[dry-run] #${DRY_N} $TAG"
            echo "  witness: $DIR"
            echo "  log:     $LOG_DIR/prove_${TAG}_YYYYMMDD_HHMMSS.log"
            case "$KERNEL" in
                linformer)
                    echo "  env:  LINFORMER_ATTN_WITNESS_DIR=$DATA_BASE/$DIR"
                    echo "  test: linformer_attn_single_layer_test"
                    ;;
                soft)
                    echo "  env:  SOFT_ATTN_WITNESS_DIR=$DATA_BASE/$DIR"
                    echo "  test: soft_attn_single_layer_test"
                    echo "  eps:  eps1=$SOFT_EPS1 eps3=$SOFT_EPS3 eps4=$SOFT_EPS4"
                    ;;
            esac
            echo ""
        done
    done
    echo "Total cells: $DRY_N"
    exit 0
fi

# -- Preflight -----------------------------------------------------------------
if [[ "$DO_WITNESS" == true ]]; then
    preflight_check
fi

# -- cargo build (release, once) -----------------------------------------------
if [[ "$DO_CIRCUIT" == true ]]; then
    echo "=== [BUILD] cargo test --release --no-run ==="
    cargo test --release --no-run 2>&1 | tail -5
    echo ""
fi

# ============================================================================
# === 1. Linformer Attention (Freivalds) =====================================
# ============================================================================
if [[ -z "$FILTER_KERNEL" ]] || [[ "$FILTER_KERNEL" == "linformer" ]]; then

    if [[ "$DO_WITNESS" == true ]]; then
        echo "=== [WITNESS] Linformer ==="
        for N in $SEQ_LENS; do
            if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
                continue
            fi
            wit_ok=true
            gen_witness_linformer "$N" || wit_ok=false
            if [[ "$wit_ok" != true ]]; then
                CONSEC_WIT_FAIL=$((CONSEC_WIT_FAIL + 1))
                echo "  [FAIL] Linformer n=$N: witness generation failed"
                if [[ $CONSEC_WIT_FAIL -ge 2 ]]; then
                    abort_consec_fail
                fi
            else
                CONSEC_WIT_FAIL=0
            fi
        done
        echo ""
    fi

    if [[ "$DO_CIRCUIT" == true ]]; then
        echo ""
        echo "################################################################"
        echo "# 1. Linformer Attention (Freivalds)"
        echo "################################################################"
        for N in $SEQ_LENS; do
            if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
                continue
            fi
            run_prove "LINFORMER_ATTN_WITNESS_DIR" \
                "$(witness_dir linformer "$N")" \
                "linformer_attn_single_layer_test" \
                "$(cell_tag linformer "$N")"
        done
    fi

fi

# ============================================================================
# === 2. SOFT Attention (Freivalds) ==========================================
# ============================================================================
if [[ -z "$FILTER_KERNEL" ]] || [[ "$FILTER_KERNEL" == "soft" ]]; then

    if [[ "$DO_WITNESS" == true ]]; then
        echo "=== [WITNESS] SOFT ==="
        for N in $SEQ_LENS; do
            if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
                continue
            fi
            wit_ok=true
            gen_witness_soft "$N" || wit_ok=false
            if [[ "$wit_ok" != true ]]; then
                CONSEC_WIT_FAIL=$((CONSEC_WIT_FAIL + 1))
                echo "  [FAIL] SOFT n=$N: witness generation failed"
                if [[ $CONSEC_WIT_FAIL -ge 2 ]]; then
                    abort_consec_fail
                fi
            else
                CONSEC_WIT_FAIL=0
            fi
        done
        echo ""
    fi

    if [[ "$DO_CIRCUIT" == true ]]; then
        echo ""
        echo "################################################################"
        echo "# 2. SOFT Attention (Freivalds)"
        echo "################################################################"
        for N in $SEQ_LENS; do
            if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
                continue
            fi
            run_prove "SOFT_ATTN_WITNESS_DIR" \
                "$(witness_dir soft "$N")" \
                "soft_attn_single_layer_test" \
                "$(cell_tag soft "$N")"
        done
    fi

fi

# ============================================================================
# === 종합 결과 ==============================================================
# ============================================================================
ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================"
echo " run_tiny_sweep.sh 완료"
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
echo "  grep '\\[FAIL\\]' $LOG_DIR/master_*.log"
