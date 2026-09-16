#!/usr/bin/env bash
# ============================================================================
# measure_nystrompy.sh -- Witness (NPZ) generation for Nystrom
#
# Generates quantized witness NPZ files for all sequence lengths.
# Run this BEFORE measure_nystromcir.sh.
#
# Usage:
#   cd circuits/
#   bash measure_nystrompy.sh [options]
#
# Options:
#   --seq N        Run only sequence length N
#   --log-dir D    Log output directory (default: logs/results_sweep/tiny)
#   --force        Re-run even if witness dir exists
#   --dry-run      Print execution plan without running
#   --help         Show this help
#
# Environment variables:
#   DATA_BASE      Witness data root (default: ../models/Nystromformer/code/dietformer)
#   LOG_DIR        Log output directory
#   PYTHON_BIN     Python interpreter path
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_BASE="${DATA_BASE:-$(cd "$SCRIPT_DIR/../models/Nystromformer/code/dietformer" 2>/dev/null && pwd || echo "")}"

# macOS /usr/bin/time
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
SEQ_LENS="128 256 512 1024 2048 4096"

HIDDEN=128
HEADS=2
LAYERS=2
NUM_LANDMARKS=32

# -- Parse arguments -----------------------------------------------------------
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
        --seq)     FILTER_SEQ="$2"; shift ;;
        --log-dir) FLAG_LOG_DIR="$2"; shift ;;
        --force)   FORCE=true ;;
        --dry-run) DRY_RUN=true ;;
        --help|-h) print_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [[ -n "$FLAG_LOG_DIR" ]]; then
    LOG_DIR="$FLAG_LOG_DIR"
else
    LOG_DIR="${LOG_DIR:-logs/results_sweep/tiny}"
fi

cd "$SCRIPT_DIR"
mkdir -p "$LOG_DIR"

# -- Helper: witness directory name --------------------------------------------
witness_dir() {
    local n="$1"
    echo "outputs_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}_nystrom"
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
    if [[ ! -f "$DATA_BASE/run_validation.py" ]]; then
        echo "  [FATAL] Missing: $DATA_BASE/run_validation.py"; exit 1
    fi
    echo "  Scripts OK"
    echo ""
}

# -- Witness generation --------------------------------------------------------
gen_witness_nystrom() {
    local n="$1"
    local dir_name
    dir_name="$(witness_dir "$n")"
    local full_path="$DATA_BASE/$dir_name"

    echo "  [witness] Nystrom n=$n -> $dir_name"
    if [[ -d "$full_path" ]] && [[ "$FORCE" != true ]]; then
        echo "  [skip] Witness exists: $dir_name"
        return 0
    fi

    "$PYTHON" "$DATA_BASE/run_validation.py" \
        --seq_len "$n" \
        --hidden_size "$HIDDEN" \
        --num_attention_heads "$HEADS" \
        --num_hidden_layers "$LAYERS" \
        --num_landmarks "$NUM_LANDMARKS" \
        --batch_size 1 --num_batch 1 \
        2>&1 | tee "$LOG_DIR/witness_nystrom_n${n}.log"

    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "  [ERROR] Nystrom witness gen failed (exit $rc)"
        return 1
    fi

    # Verify token mask
    echo "  [verify] Token mask:"
    "$PYTHON" -c "
import numpy as np, glob, os
pat = os.path.join('$full_path', '*.npz')
for f in sorted(glob.glob(pat)):
    bn = os.path.basename(f)
    skip = ('witness', 'gaussian', 'norm', 'gelu', 'softmax', 'table')
    if any(s in bn for s in skip):
        continue
    d = np.load(f, allow_pickle=True)
    if 'mask' in d:
        mask = d['mask'][0]
        valid = int(np.sum(mask == 1))
        pad = int(np.sum(mask == 0))
        print(f'  valid={valid}, pad={pad}')
    break
" 2>&1
}

# -- Start time ----------------------------------------------------------------
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo " measure_nystrompy.sh (Witness Generation)"
echo " Started:   $STARTED_AT"
echo " Data base: $DATA_BASE"
echo " Log dir:   $LOG_DIR"
echo " Python:    ${PYTHON:-<not found>}"
echo " Config:    L=$LAYERS, h=$HIDDEN, H=$HEADS"
echo " Nystrom:   m=$NUM_LANDMARKS"
echo "============================================"
echo ""

# -- Dry-run -------------------------------------------------------------------
if [[ "$DRY_RUN" == true ]]; then
    DRY_N=0
    for N in $SEQ_LENS; do
        if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
            continue
        fi
        DRY_N=$((DRY_N + 1))
        DIR="$(witness_dir "$N")"
        echo "[dry-run] #${DRY_N} nystrom n=${N}"
        echo "  output: $DATA_BASE/$DIR"
        echo "  log:    $LOG_DIR/witness_nystrom_n${N}.log"
    done
    echo "Total witness jobs: $DRY_N"
    exit 0
fi

# -- Preflight -----------------------------------------------------------------
preflight_check

# -- Counters ------------------------------------------------------------------
TOTAL=0
PASS=0
FAIL=0
CONSEC_FAIL=0

# -- Main loop -----------------------------------------------------------------
echo "=== [WITNESS] Nystrom ==="
for N in $SEQ_LENS; do
    if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
        continue
    fi
    TOTAL=$((TOTAL + 1))
    wit_ok=true
    gen_witness_nystrom "$N" || wit_ok=false
    if [[ "$wit_ok" != true ]]; then
        FAIL=$((FAIL + 1))
        CONSEC_FAIL=$((CONSEC_FAIL + 1))
        echo "  [FAIL] Nystrom n=$N"
        if [[ $CONSEC_FAIL -ge 2 ]]; then
            echo "[ABORT] 2 consecutive witness failures"; exit 1
        fi
    else
        PASS=$((PASS + 1))
        CONSEC_FAIL=0
    fi
done
echo ""

# -- Summary -------------------------------------------------------------------
ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================"
echo " measure_nystrompy.sh 완료"
echo " 시작: $STARTED_AT"
echo " 종료: $ENDED_AT"
echo " 총:   $TOTAL  |  성공: $PASS  |  실패: $FAIL"
echo "============================================"
