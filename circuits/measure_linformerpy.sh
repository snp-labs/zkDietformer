#!/usr/bin/env bash
# ============================================================================
# measure_linformerpy.sh -- Witness (NPZ) generation for Linformer/SOFT
#
# Generates quantized witness NPZ files for all sequence lengths.
# Run this BEFORE measure_linformercir.sh.
#
# Usage:
#   cd circuits/
#   bash measure_linformerpy.sh [options]
#
# Options:
#   --kernel K     Run only kernel K (linformer or soft)
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
SOFT_EPS1=8037
SOFT_EPS3=99
SOFT_EPS4=144

SEQ_LENS="128 256 512 1024 2048 4096"

INPUT_TEXT="The capital of France is [MASK]."
INPUT_SEED=42

HIDDEN=128
HEADS=2
LAYERS=2
PROJ_DIM=32
NUM_LANDMARKS=32
N_ITER=20

# -- Parse arguments -----------------------------------------------------------
FILTER_KERNEL=""
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
        --kernel)  FILTER_KERNEL="$2"; shift ;;
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
    local kernel="$1" n="$2"
    case "$kernel" in
        linformer) echo "outputs_n${n}_k${PROJ_DIM}_L${LAYERS}_h${HIDDEN}_linformer" ;;
        soft)      echo "outputs_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}_soft" ;;
    esac
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

# -- Start time ----------------------------------------------------------------
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo " measure_linformerpy.sh (Witness Generation)"
echo " Started:   $STARTED_AT"
echo " Data base: $DATA_BASE"
echo " Log dir:   $LOG_DIR"
echo " Python:    ${PYTHON:-<not found>}"
echo " Config:    L=$LAYERS, h=$HIDDEN, H=$HEADS"
echo " Linformer: k=$PROJ_DIM"
echo " SOFT:      m=$NUM_LANDMARKS, T=$N_ITER, eps=$SOFT_EPS1/$SOFT_EPS3/$SOFT_EPS4"
echo "============================================"
echo ""

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
            DIR="$(witness_dir "$KERNEL" "$N")"
            echo "[dry-run] #${DRY_N} ${KERNEL} n=${N}"
            echo "  output: $DATA_BASE/$DIR"
            echo "  log:    $LOG_DIR/witness_${KERNEL}_n${N}.log"
        done
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
SKIP=0
CONSEC_FAIL=0

# -- Main loop -----------------------------------------------------------------

# 1. Linformer witnesses
if [[ -z "$FILTER_KERNEL" ]] || [[ "$FILTER_KERNEL" == "linformer" ]]; then
    echo "=== [WITNESS] Linformer ==="
    for N in $SEQ_LENS; do
        if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
            continue
        fi
        TOTAL=$((TOTAL + 1))
        wit_ok=true
        gen_witness_linformer "$N" || wit_ok=false
        if [[ "$wit_ok" != true ]]; then
            FAIL=$((FAIL + 1))
            CONSEC_FAIL=$((CONSEC_FAIL + 1))
            echo "  [FAIL] Linformer n=$N"
            if [[ $CONSEC_FAIL -ge 2 ]]; then
                echo "[ABORT] 2 consecutive witness failures"; exit 1
            fi
        else
            PASS=$((PASS + 1))
            CONSEC_FAIL=0
        fi
    done
    echo ""
fi

# 2. SOFT witnesses
if [[ -z "$FILTER_KERNEL" ]] || [[ "$FILTER_KERNEL" == "soft" ]]; then
    echo "=== [WITNESS] SOFT ==="
    for N in $SEQ_LENS; do
        if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
            continue
        fi
        TOTAL=$((TOTAL + 1))
        wit_ok=true
        gen_witness_soft "$N" || wit_ok=false
        if [[ "$wit_ok" != true ]]; then
            FAIL=$((FAIL + 1))
            CONSEC_FAIL=$((CONSEC_FAIL + 1))
            echo "  [FAIL] SOFT n=$N"
            if [[ $CONSEC_FAIL -ge 2 ]]; then
                echo "[ABORT] 2 consecutive witness failures"; exit 1
            fi
        else
            PASS=$((PASS + 1))
            CONSEC_FAIL=0
        fi
    done
    echo ""
fi

# -- Summary -------------------------------------------------------------------
ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================"
echo " measure_linformerpy.sh 완료"
echo " 시작: $STARTED_AT"
echo " 종료: $ENDED_AT"
echo " 총:   $TOTAL  |  성공: $PASS  |  실패: $FAIL"
echo "============================================"
