#!/usr/bin/env bash
# ============================================================================
# run_tiny_sweep.sh -- Linformer/SOFT attention circuit measurement sweep
#
# Tiny config: L=2, hidden=128, heads=2
# Kernels:     Linformer (Freivalds, k=32), SOFT (Freivalds, m=32)
# Sequences:   128, 256, 512, 1024, 2048, 4096
# Total cells: 12
#
# SOFT epsilon: n-independent constants (eps1=8037, eps3=99, eps4=144)
# Input text:   SAMPLE_TEXTS[0] from evaluation_fixtures.py
#
# Usage:
#   bash scripts/run_tiny_sweep.sh [OPTIONS]
#
# Options:
#   --witness-only   Generate witnesses only (no circuit measurement)
#   --circuit-only   Run circuit measurement only (witnesses must exist)
#   --kernel K       Run only kernel K (linformer or soft)
#   --seq N          Run only sequence length N
#   --force          Re-run even if results exist
#   --dry-run        Print execution plan without running
#   --help           Show this help
# ============================================================================
set -uo pipefail

# ── Constants ────────────────────────────────────────────────────────────────
SOFT_EPS1=8037
SOFT_EPS3=99
SOFT_EPS4=144

SEQ_LENS="128 256 512 1024 2048 4096"
KERNELS="linformer soft"

INPUT_TEXT="The capital of France is [MASK]."
INPUT_SEED=42

# Tiny config
HIDDEN=128
HEADS=2
LAYERS=2
PROJ_DIM=32          # Linformer k
NUM_LANDMARKS=32     # SOFT m
N_ITER=20            # Newton-Raphson T

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CIRCUIT_DIR="$REPO_ROOT"
DATA_BASE="$(cd "$REPO_ROOT/../models/Nystromformer/code/dietformer" && pwd)"
PYTHON="/opt/miniconda3/envs/dietformer/bin/python"

TIME_CMD="/usr/bin/time -l"

# ── Parse arguments ──────────────────────────────────────────────────────────
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

# ── Logging setup ────────────────────────────────────────────────────────────
cd "$CIRCUIT_DIR"
mkdir -p logs

STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
TOTAL=0; PASS=0; FAIL=0; SKIP=0; OOM=0

# ── Helper: witness directory name ───────────────────────────────────────────
witness_dir() {
    local kernel="$1" n="$2"
    case "$kernel" in
        linformer) echo "outputs_n${n}_k${PROJ_DIM}_L${LAYERS}_h${HIDDEN}_linformer" ;;
        soft)      echo "outputs_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}_soft" ;;
    esac
}

# ── Helper: check if cell result exists ──────────────────────────────────────
result_exists() {
    local tag="$1"
    # A cell is complete if a prove log with BENCHMARK_RESULT exists
    local latest
    latest=$(ls -t logs/prove_${tag}_*.log 2>/dev/null | head -1)
    if [[ -n "$latest" ]] && grep -q "BENCHMARK_RESULT" "$latest" 2>/dev/null; then
        return 0
    fi
    return 1
}

# ── Witness generation ───────────────────────────────────────────────────────
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
        2>&1 | tee "logs/witness_linformer_n${n}.log"

    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "  [ERROR] Linformer witness gen failed (exit $rc)"
        return 1
    fi

    # Verify tokenization: log first 10 token IDs
    echo "  [verify] Token IDs from witness:"
    "$PYTHON" -c "
import numpy as np, glob, os
pat = os.path.join('$full_path', '*.npz')
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
            2>&1 | tee "logs/witness_soft_n${n}.log"

        local rc=${PIPESTATUS[0]}
        if [[ $rc -ne 0 ]]; then
            echo "  [ERROR] SOFT witness gen failed (exit $rc)"
            return 1
        fi
    fi

    # Override epsilon with n-independent constants
    echo "  [eps] Overriding eps: eps1=$SOFT_EPS1, eps3=$SOFT_EPS3, eps4=$SOFT_EPS4"
    local wit_files
    wit_files=$(find "$full_path" -name "*.witness.npz" -not -name "*.norm.witness.npz" 2>/dev/null)
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
# Verify
v = np.load('$wit', allow_pickle=True)
e1 = int(v['eps1']); e3 = int(v['eps3']); e4 = int(v['eps4'])
if e1 != $SOFT_EPS1 or e3 != $SOFT_EPS3 or e4 != $SOFT_EPS4:
    print(f'VERIFY_FAIL: eps1={e1}, eps3={e3}, eps4={e4}', file=sys.stderr)
    sys.exit(1)
print(f'  [ok] {\"$wit\".split(\"/\")[-1]}: eps1={e1}, eps3={e3}, eps4={e4}')
" 2>&1
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            eps_ok=false
        fi
    done <<< "$wit_files"

    if [[ "$eps_ok" != true ]]; then
        echo "  [ERROR] Epsilon verification failed"
        return 1
    fi

    # Verify tokenization
    echo "  [verify] Token IDs from witness:"
    "$PYTHON" -c "
import numpy as np, glob, os
pat = os.path.join('$full_path', '*batch0*.npz')
for f in sorted(glob.glob(pat)):
    if 'witness' not in f:
        d = np.load(f, allow_pickle=True)
        if 'mask' in d:
            mask = d['mask'][0]
            valid = int(np.sum(mask == 1))
            pad = int(np.sum(mask == 0))
            print(f'  valid={valid}, pad={pad}')
        break
" 2>&1
}

# ── Circuit measurement ──────────────────────────────────────────────────────
run_prove() {
    local witness_var="$1"
    local data_dir="$2"
    local test_fn="$3"
    local tag="$4"

    local full_data_path="$DATA_BASE/$data_dir"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="logs/prove_${tag}_${ts}.log"
    local time_file="logs/prove_${tag}_${ts}.time"

    TOTAL=$((TOTAL + 1))

    echo "============================================"
    echo "[$(date +%H:%M:%S)] #${TOTAL} $tag"
    echo "  Test:  $test_fn"
    echo "  Data:  $data_dir"
    echo "  Log:   $log_file"
    echo "============================================"

    if [[ ! -d "$full_data_path" ]]; then
        SKIP=$((SKIP + 1))
        echo "  [SKIP] Data dir not found: $full_data_path"
        echo ""
        return 0
    fi

    if [[ "$FORCE" != true ]] && result_exists "$tag"; then
        SKIP=$((SKIP + 1))
        echo "  [SKIP] Result exists for $tag"
        echo ""
        return 0
    fi

    local exit_code=0
    if env "$witness_var=$full_data_path" \
        $TIME_CMD \
        cargo test --release "$test_fn" -- --nocapture \
        > "$log_file" 2> "$time_file"; then
        exit_code=0
    else
        exit_code=$?
    fi

    # Check for OOM (signal 9 = SIGKILL, common OOM behavior)
    if [[ $exit_code -eq 137 ]] || grep -qi "killed" "$time_file" 2>/dev/null; then
        OOM=$((OOM + 1))
        echo "  [OOM] $tag (exit code: $exit_code)"
        # Extract peak RSS before OOM
        local peak_rss
        peak_rss=$(grep "maximum resident set size" "$time_file" 2>/dev/null | awk '{print $1}' || echo "N/A")
        echo "  Peak RSS at OOM: $peak_rss bytes"
        echo "OOM" > "logs/result_${tag}.status"
        echo ""
        return 0
    fi

    if [[ $exit_code -ne 0 ]]; then
        FAIL=$((FAIL + 1))
        echo "  [FAIL] $tag (exit code: $exit_code)"
        echo "FAIL" > "logs/result_${tag}.status"
    else
        # Verify eps source for SOFT
        if [[ "$tag" == *soft* ]]; then
            if grep -q "\[fallback\]" "$time_file" 2>/dev/null; then
                FAIL=$((FAIL + 1))
                echo "  [FAIL] $tag: eps loaded from fallback, not NPZ"
                echo "FAIL_EPS_FALLBACK" > "logs/result_${tag}.status"
                echo ""
                return 0
            fi
            echo "  [eps] Source verified: all [npz]"
        fi

        PASS=$((PASS + 1))
        echo "  [PASS] $tag"
        echo "PASS" > "logs/result_${tag}.status"
    fi

    # Extract metrics
    if [[ -f "$time_file" ]]; then
        local peak_rss peak_rss_mb
        peak_rss=$(grep "maximum resident set size" "$time_file" 2>/dev/null | awk '{print $1}' || echo "N/A")
        if [[ "$peak_rss" != "N/A" ]] && [[ "$peak_rss" -gt 0 ]] 2>/dev/null; then
            peak_rss_mb=$((peak_rss / 1048576))
        else
            peak_rss_mb="N/A"
        fi
        echo "  Peak RSS: ${peak_rss} bytes (${peak_rss_mb} MB)"
    fi

    local layer_count
    layer_count=$(grep -c "BENCHMARK_RESULT" "$log_file" 2>/dev/null || echo "0")
    echo "  Layers proved: $layer_count"

    # Extract d1/d2/aux sizes
    local d1_line
    d1_line=$(grep "^d1=" "$log_file" 2>/dev/null | tail -1)
    if [[ -n "$d1_line" ]]; then
        echo "  Sizes: $d1_line"
    fi

    # SOFT-specific: eps and MP headroom from stderr
    if [[ "$tag" == *soft* ]] && [[ -f "$time_file" ]]; then
        local eps_line
        eps_line=$(grep "\[soft\] eps" "$time_file" 2>/dev/null | head -1)
        if [[ -n "$eps_line" ]]; then
            echo "  $eps_line"
        fi
    fi

    echo ""
}

# ── Main execution ───────────────────────────────────────────────────────────

echo "============================================"
echo " run_tiny_sweep.sh"
echo " Started:    $STARTED_AT"
echo " Circuit:    $CIRCUIT_DIR"
echo " Data:       $DATA_BASE"
echo " Python:     $PYTHON"
echo " Config:     L=$LAYERS, h=$HIDDEN, H=$HEADS"
echo " Linformer:  k=$PROJ_DIM"
echo " SOFT:       m=$NUM_LANDMARKS, T=$N_ITER"
echo " SOFT eps:   eps1=$SOFT_EPS1, eps3=$SOFT_EPS3, eps4=$SOFT_EPS4"
echo " Input text: $INPUT_TEXT"
echo " Input seed: $INPUT_SEED"
echo " Witness:    $DO_WITNESS"
echo " Circuit:    $DO_CIRCUIT"
echo " Force:      $FORCE"
echo "============================================"
echo ""

# Build (once)
if [[ "$DO_CIRCUIT" == true ]] && [[ "$DRY_RUN" != true ]]; then
    echo "=== [BUILD] cargo test --release --no-run ==="
    cargo test --release --no-run 2>&1 | tail -5
    echo ""
fi

# Iterate: small n first, linformer then soft
for N in $SEQ_LENS; do
    # Filter
    if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
        continue
    fi

    for K in $KERNELS; do
        if [[ -n "$FILTER_KERNEL" ]] && [[ "$K" != "$FILTER_KERNEL" ]]; then
            continue
        fi

        local_dir="$(witness_dir "$K" "$N")"
        tag="${K}_freivalds_n${N}"

        if [[ "$DRY_RUN" == true ]]; then
            echo "[dry-run] $tag"
            echo "  witness_dir: $local_dir"
            case "$K" in
                linformer)
                    echo "  env: LINFORMER_ATTN_WITNESS_DIR=$DATA_BASE/$local_dir"
                    echo "  test: linformer_attn_single_layer_test"
                    ;;
                soft)
                    echo "  env: SOFT_ATTN_WITNESS_DIR=$DATA_BASE/$local_dir"
                    echo "  test: soft_attn_single_layer_test"
                    echo "  eps: eps1=$SOFT_EPS1 eps3=$SOFT_EPS3 eps4=$SOFT_EPS4"
                    ;;
            esac
            echo ""
            continue
        fi

        echo ""
        echo "################################################################"
        echo "# $K n=$N"
        echo "################################################################"

        # Witness generation
        if [[ "$DO_WITNESS" == true ]]; then
            wit_ok=true
            case "$K" in
                linformer) gen_witness_linformer "$N" || wit_ok=false ;;
                soft)      gen_witness_soft "$N" || wit_ok=false ;;
            esac
            if [[ "$wit_ok" != true ]]; then
                TOTAL=$((TOTAL + 1))
                FAIL=$((FAIL + 1))
                echo "  [FAIL] $tag: witness generation failed"
                echo "FAIL_WITNESS" > "logs/result_${tag}.status"
                continue
            fi
        fi

        # Circuit measurement
        if [[ "$DO_CIRCUIT" == true ]]; then
            case "$K" in
                linformer)
                    run_prove "LINFORMER_ATTN_WITNESS_DIR" \
                        "$local_dir" \
                        "linformer_attn_single_layer_test" \
                        "$tag"
                    ;;
                soft)
                    run_prove "SOFT_ATTN_WITNESS_DIR" \
                        "$local_dir" \
                        "soft_attn_single_layer_test" \
                        "$tag"
                    ;;
            esac
        fi
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" != true ]]; then
    ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "============================================"
    echo " run_tiny_sweep.sh complete"
    echo " Started: $STARTED_AT"
    echo " Ended:   $ENDED_AT"
    echo "--------------------------------------------"
    echo " Total:   $TOTAL"
    echo " Pass:    $PASS"
    echo " Fail:    $FAIL"
    echo " OOM:     $OOM"
    echo " Skip:    $SKIP"
    echo "============================================"
    echo ""
    echo "=== Result commands ==="
    echo ""
    echo "# Per-layer benchmarks (JSON):"
    echo "  grep BENCHMARK_RESULT logs/prove_*.log"
    echo ""
    echo "# Peak RSS:"
    echo "  grep 'maximum resident set size' logs/prove_*.time"
    echo ""
    echo "# d1/d2/aux sizes:"
    echo "  grep '^d1=' logs/prove_*.log"
    echo ""
    echo "# SOFT eps source:"
    echo "  grep '\\[soft\\] eps' logs/prove_*.time"
    echo ""
    echo "# Cell status:"
    echo "  cat logs/result_*.status"
    echo ""
fi
