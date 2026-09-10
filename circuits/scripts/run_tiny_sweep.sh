#!/usr/bin/env bash
# ============================================================================
# run_tiny_sweep.sh -- Linformer/SOFT attention circuit measurement sweep
#
# 폴더: outputs_n{128,256,512,1024,2048,4096}_{k32_,m32_}L2_h128_{linformer,soft}
# 회로: linformer_attn (Freivalds) → soft_attn (Freivalds)
#
# 환경: macOS (Darwin, ARM64 Mac Studio, 128 GB)
# 용도: SSH 접속 후 nohup 백그라운드 실행 → 퇴근 후 자동 완료
#
# 실행 방법:
#   cd /Users/minhae/Projects/fin_zkdietformer/Dietformer/circuits
#   chmod +x scripts/run_tiny_sweep.sh
#   nohup caffeinate -i bash scripts/run_tiny_sweep.sh > logs/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo "PID: $!"
#
# 결과 확인:
#   tail -f logs/master_*.log                                    # 진행 상황
#   grep BENCHMARK_RESULT logs/prove_*.log                       # per-layer 벤치마크
#   grep "maximum resident set size" logs/prove_*.time           # peak RSS (bytes)
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
# NOTE: -e 미사용 — 한 테스트 실패해도 다음 테스트 계속 진행

# ── 경로 설정 ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CIRCUIT_DIR="$REPO_ROOT"
DATA_BASE="$(cd "$REPO_ROOT/../models/Nystromformer/code/dietformer" && pwd)"
PYTHON="/opt/miniconda3/envs/dietformer/bin/python"

cd "$CIRCUIT_DIR"
mkdir -p logs

# macOS /usr/bin/time (절대 경로, shell builtin 아님)
TIME_CMD="/usr/bin/time -l"

# ── Constants ────────────────────────────────────────────────────────────────
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

# ── 시작 시각 ────────────────────────────────────────────────────────────────
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo " run_tiny_sweep.sh started: $STARTED_AT"
echo " Circuit dir: $CIRCUIT_DIR"
echo " Data base:   $DATA_BASE"
echo " Config:      L=$LAYERS, h=$HIDDEN, H=$HEADS"
echo " Linformer:   k=$PROJ_DIM"
echo " SOFT:        m=$NUM_LANDMARKS, T=$N_ITER, eps1=$SOFT_EPS1, eps3=$SOFT_EPS3, eps4=$SOFT_EPS4"
echo "============================================"
echo ""

# ── 카운터 ────────────────────────────────────────────────────────────────────
TOTAL_RUNS=0
PASS_RUNS=0
FAIL_RUNS=0
SKIP_RUNS=0
OOM_RUNS=0

# ── Helper: witness directory name ───────────────────────────────────────────
witness_dir() {
    local kernel="$1" n="$2"
    case "$kernel" in
        linformer) echo "outputs_n${n}_k${PROJ_DIM}_L${LAYERS}_h${HIDDEN}_linformer" ;;
        soft)      echo "outputs_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}_soft" ;;
    esac
}

# ── Helper: cell tag ─────────────────────────────────────────────────────────
cell_tag() {
    local kernel="$1" n="$2"
    case "$kernel" in
        linformer) echo "linformer_attn_n${n}_k${PROJ_DIM}_L${LAYERS}_h${HIDDEN}" ;;
        soft)      echo "soft_attn_n${n}_m${NUM_LANDMARKS}_L${LAYERS}_h${HIDDEN}" ;;
    esac
}

# ── Helper: check if cell result exists ──────────────────────────────────────
result_exists() {
    local tag="$1"
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

# ── 측정 함수 ────────────────────────────────────────────────────────────────
# run_prove WITNESS_VAR DATA_DIR TEST_FN TAG
run_prove() {
    local WITNESS_VAR="$1"
    local DATA_DIR="$2"
    local TEST_FN="$3"
    local TAG="$4"

    local FULL_DATA_PATH="$DATA_BASE/$DATA_DIR"
    local TIMESTAMP
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    local LOG_FILE="logs/prove_${TAG}_${TIMESTAMP}.log"
    local TIME_FILE="logs/prove_${TAG}_${TIMESTAMP}.time"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    echo "============================================"
    echo "[$(date +%H:%M:%S)] #${TOTAL_RUNS} $TAG"
    echo "  Test:  $TEST_FN"
    echo "  Data:  $DATA_DIR"
    echo "  Log:   $LOG_FILE"
    echo "  Time:  $TIME_FILE"
    echo "============================================"

    # 데이터 폴더 존재 확인
    if [ ! -d "$FULL_DATA_PATH" ]; then
        SKIP_RUNS=$((SKIP_RUNS + 1))
        echo "  [SKIP] Data dir not found: $FULL_DATA_PATH"
        echo ""
        return 0
    fi

    # 기존 결과 확인
    if [[ "$FORCE" != true ]] && result_exists "$TAG"; then
        SKIP_RUNS=$((SKIP_RUNS + 1))
        echo "  [SKIP] Result exists for $TAG"
        echo ""
        return 0
    fi

    # stdout → prove log, stderr → time log (peak RSS)
    local EXIT_CODE=0
    if env "$WITNESS_VAR=$FULL_DATA_PATH" \
        $TIME_CMD \
        cargo test --release "$TEST_FN" -- --nocapture \
        > "$LOG_FILE" 2> "$TIME_FILE"; then
        # SOFT: eps fallback check
        if [[ "$TAG" == *soft_attn* ]] && grep -q "\[fallback\]" "$TIME_FILE" 2>/dev/null; then
            FAIL_RUNS=$((FAIL_RUNS + 1))
            echo "  [FAIL] $TAG (eps loaded from fallback, not NPZ)"
        else
            PASS_RUNS=$((PASS_RUNS + 1))
            echo "  [PASS] $TAG"
        fi
    else
        EXIT_CODE=$?
        # OOM check (signal 9 = SIGKILL)
        if [[ $EXIT_CODE -eq 137 ]] || grep -qi "killed" "$TIME_FILE" 2>/dev/null; then
            OOM_RUNS=$((OOM_RUNS + 1))
        fi
        FAIL_RUNS=$((FAIL_RUNS + 1))
        echo "  [FAIL] $TAG (exit code: $EXIT_CODE)"
    fi

    # peak RSS 추출 (macOS: bytes)
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

    # BENCHMARK_RESULT 라인 수 (= prove 완료 layer 수)
    local LAYER_COUNT
    LAYER_COUNT=$(grep -c "BENCHMARK_RESULT" "$LOG_FILE" 2>/dev/null || true)
    : "${LAYER_COUNT:=0}"
    echo "  Layers proved: $LAYER_COUNT"
    echo ""
}

# ── Dry-run ──────────────────────────────────────────────────────────────────
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

# ── cargo 빌드 (release, 1회) ────────────────────────────────────────────────
if [[ "$DO_CIRCUIT" == true ]]; then
    echo "=== [BUILD] cargo test --release --no-run ==="
    cargo test --release --no-run 2>&1 | tail -5
    echo ""
fi

# ============================================================================
# === 1. Linformer Attention (Freivalds) =====================================
# ============================================================================
if [[ -z "$FILTER_KERNEL" ]] || [[ "$FILTER_KERNEL" == "linformer" ]]; then

    # ── Witness generation ──────────────────────────────────────────────────
    if [[ "$DO_WITNESS" == true ]]; then
        echo "=== [WITNESS] Linformer ==="
        for N in $SEQ_LENS; do
            if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
                continue
            fi
            wit_ok=true
            gen_witness_linformer "$N" || wit_ok=false
            if [[ "$wit_ok" != true ]]; then
                echo "  [FAIL] Linformer n=$N: witness generation failed"
            fi
        done
        echo ""
    fi

    # ── Circuit measurement ─────────────────────────────────────────────────
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

    # ── Witness generation ──────────────────────────────────────────────────
    if [[ "$DO_WITNESS" == true ]]; then
        echo "=== [WITNESS] SOFT ==="
        for N in $SEQ_LENS; do
            if [[ -n "$FILTER_SEQ" ]] && [[ "$N" != "$FILTER_SEQ" ]]; then
                continue
            fi
            wit_ok=true
            gen_witness_soft "$N" || wit_ok=false
            if [[ "$wit_ok" != true ]]; then
                echo "  [FAIL] SOFT n=$N: witness generation failed"
            fi
        done
        echo ""
    fi

    # ── Circuit measurement ─────────────────────────────────────────────────
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
echo "  grep BENCHMARK_RESULT logs/prove_*.log"
echo ""
echo "# peak RSS (메모리, bytes):"
echo "  grep 'maximum resident set size' logs/prove_*.time"
echo ""
echo "# 전체 wall time per run:"
echo "  grep 'total_wall_ms' logs/prove_*.log"
echo ""
echo "# 실패한 run 확인:"
echo "  grep '\\[FAIL\\]' logs/master_*.log"
