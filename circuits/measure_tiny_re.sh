#!/usr/bin/env bash
# ============================================================================
# measure_all_resume.sh — 중단된 지점부터 이어서 실행 (남은 실험 재개)
#
# 대상:
#   2. Nystrom Attention Naive -> n=4096 만 실행
#   5. Nystrom MLP             -> 전체 시퀀스 실행 (128~4096)
#   6. BERT MLP                -> 전체 시퀀스 실행 (128~4096)
# ============================================================================
set -uo pipefail

# ── 경로 설정 ────────────────────────────────────────────────────────────────
DATA_BASE="/Users/zkrypto12/zkdietformer/Dietformer/models/Nystromformer/code/dietformer"
CIRCUIT_DIR="/Users/zkrypto12/zkdietformer/Dietformer/circuits"

cd "$CIRCUIT_DIR"
mkdir -p logs

# macOS /usr/bin/time (절대 경로, shell builtin 아님)
TIME_CMD="/usr/bin/time -l"

# ── 시작 시각 ────────────────────────────────────────────────────────────────
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo " measure_all_resume.sh started: $STARTED_AT"
echo " Circuit dir: $CIRCUIT_DIR"
echo " Data base:   $DATA_BASE"
echo "============================================"
echo ""

# ── cargo 빌드 (release, 1회) ────────────────────────────────────────────────
echo "=== [BUILD] cargo test --release --no-run ==="
cargo test --release --no-run 2>&1 | tail -5
echo ""

# ── 카운터 ────────────────────────────────────────────────────────────────────
TOTAL_RUNS=0
PASS_RUNS=0
FAIL_RUNS=0
SKIP_RUNS=0

# ── 측정 함수 (시간/메모리 출력 개선 버전) ───────────────────────────────────
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

    # 테스트 실행 및 로그 분리 저장
    local EXIT_CODE=0
    if env "$WITNESS_VAR=$FULL_DATA_PATH" \
        $TIME_CMD \
        cargo test --release "$TEST_FN" -- --nocapture \
        > "$LOG_FILE" 2> "$TIME_FILE"; then
        PASS_RUNS=$((PASS_RUNS + 1))
        echo "  [PASS] $TAG"
    else
        EXIT_CODE=$?
        FAIL_RUNS=$((FAIL_RUNS + 1))
        echo "  [FAIL] $TAG (exit code: $EXIT_CODE)"
    fi

    # /usr/bin/time -l 결과 실시간 파싱 및 화면 출력
    if [ -f "$TIME_FILE" ]; then
        local REAL_TIME; REAL_TIME=$(awk '/real/ {print $1 "s"}' "$TIME_FILE" || echo "N/A")
        local USER_TIME; USER_TIME=$(awk '/user/ {print $1 "s"}' "$TIME_FILE" || echo "N/A")
        local SYS_TIME;  SYS_TIME=$(awk '/sys/ {print $1 "s"}' "$TIME_FILE" || echo "N/A")

        local PEAK_RSS;  PEAK_RSS=$(grep "maximum resident set size" "$TIME_FILE" | awk '{print $1}' || echo "0")
        local PEAK_RSS_MB="N/A"
        
        if [ "$PEAK_RSS" -gt 0 ] 2>/dev/null; then
            PEAK_RSS_MB=$((PEAK_RSS / 1048576))
        fi

        echo "  ------------------------------------------"
        echo "  [TIME] Real: $REAL_TIME | User: $USER_TIME | Sys: $SYS_TIME"
        echo "  [MEM]  Peak RSS: $PEAK_RSS bytes ($PEAK_RSS_MB MB)"
        echo "  ------------------------------------------"
    fi

    # BENCHMARK_RESULT 라인 수 (= prove 완료 layer 수)
    local LAYER_COUNT
    LAYER_COUNT=$(grep -c "BENCHMARK_RESULT" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "  Layers proved: $LAYER_COUNT"
    echo ""
}

# ── seq_len 목록 ─────────────────────────────────────────────────────────────
SEQ_LENS="128 256 512 1024 2048 4096"

# ============================================================================
# === [이전 완료] 1. Nystrom Attention Freivalds (건너뜀) =====================
# ============================================================================

# ============================================================================
# === 2. Nystrom Attention Naive (n=4096 남은 실험 진행) =====================
# ============================================================================
echo ""
echo "################################################################"
echo "# 2. Nystrom Attention (Naive) - RESUME (n=4096)"
echo "################################################################"
run_prove "NAIVE_NYSTROM_ATTN_WITNESS_DIR" \
    "outputs_n4096_m32_L2_h128_nystrom" \
    "nystrom_attn_naive_single_layer_test" \
    "nystrom_attn_naive_n4096_m32_L2_h128"

# ============================================================================
# === [이전 완료] 3. BERT Attention Freivalds (건너뜀) =======================
# ============================================================================

# ============================================================================
# === [이전 완료] 4. BERT Attention Naive (건너뜀) ===========================
# ============================================================================

# ============================================================================
# === 5. Nystrom MLP (전체 시퀀스 진행) =======================================
# ============================================================================
echo ""
echo "################################################################"
echo "# 5. Nystrom MLP (Freivalds) - ALL SEQ_LENS"
echo "################################################################"
for N in $SEQ_LENS; do
    run_prove "LAYERNORM_WITNESS_DIR" \
        "outputs_n${N}_m32_L2_h128_nystrom" \
        "nystrom_mlp_hf_single_layer_test" \
        "nystrom_mlp_n${N}_m32_L2_h128"
done

# ============================================================================
# === 종합 결과 ==============================================================
# ============================================================================
ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================"
echo " measure_re.sh 완료"
echo " 시작: $STARTED_AT"
echo " 종료: $ENDED_AT"
echo " 총 실행: $TOTAL_RUNS"
echo " 성공:    $PASS_RUNS"
echo " 실패:    $FAIL_RUNS"
echo " 스킵:    $SKIP_RUNS"
echo "============================================"