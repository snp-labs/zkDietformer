#!/usr/bin/env bash
# ============================================================================
# measure_onlyeq/measure_mini.sh — Attention-Only (onlyeq) 벤치마크
#
# 스펙: mini (L4, h256)
# 폴더: outputs_n{128..4096}_{m32_}L4_h256_{nystrom,bert}
# 회로: nystrom_attn_onlyeq → bert_attn_onlyeq  (2종, Freivalds만)
#
# 실행 방법:
#   cd /path/to/circuits
#   chmod +x measure_onlyeq/measure_mini.sh
#   nohup caffeinate -i bash measure_onlyeq/measure_mini.sh \
#       > logs/results_onlyeq/mini/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ============================================================================
set -uo pipefail

# ── 경로 설정 ────────────────────────────────────────────────────────────────
DATA_BASE="/Users/zkrypto12/zkdietformer/Dietformer/models/Nystromformer/code/dietformer"
CIRCUIT_DIR="/Users/zkrypto12/zkdietformer/Dietformer/circuits"

cd "$CIRCUIT_DIR"
LOG_DIR="logs/results_onlyeq/mini"
mkdir -p "$LOG_DIR"

TIME_CMD="/usr/bin/time -l"

# ── 시작 시각 ────────────────────────────────────────────────────────────────
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo " measure_onlyeq/measure_mini.sh started: $STARTED_AT"
echo " Spec:        mini (L4, h256)"
echo " Circuit dir: $CIRCUIT_DIR"
echo " Data base:   $DATA_BASE"
echo " Log dir:     $LOG_DIR"
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

# ── 측정 함수 ────────────────────────────────────────────────────────────────
run_prove() {
    local WITNESS_VAR="$1"
    local DATA_DIR="$2"
    local TEST_FN="$3"
    local TAG="$4"

    local FULL_DATA_PATH="$DATA_BASE/$DATA_DIR"
    local TIMESTAMP
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    local LOG_FILE="${LOG_DIR}/prove_${TAG}_${TIMESTAMP}.log"
    local TIME_FILE="${LOG_DIR}/prove_${TAG}_${TIMESTAMP}.time"

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
    LAYER_COUNT=$(grep -c "BENCHMARK_RESULT" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "  Layers proved: $LAYER_COUNT"
    echo ""
}

# ── seq_len 목록 ─────────────────────────────────────────────────────────────
SEQ_LENS="128 256 512 1024 2048 4096"

# ============================================================================
# === 1. Nystrom Attention Onlyeq (Freivalds) ================================
# ============================================================================
echo ""
echo "################################################################"
echo "# 1. Nystrom Attention Onlyeq (Freivalds) — mini"
echo "################################################################"
for N in $SEQ_LENS; do
    run_prove "NYSTROM_ATTN_WITNESS_DIR" \
        "outputs_n${N}_m32_L4_h256_nystrom" \
        "nystrom_attn_onlyeq_single_layer_test" \
        "nystrom_attn_onlyeq_n${N}_m32_L4_h256"
done

# ============================================================================
# === 2. BERT Attention Onlyeq (Freivalds) ===================================
# ============================================================================
echo ""
echo "################################################################"
echo "# 2. BERT Attention Onlyeq (Freivalds) — mini"
echo "################################################################"
for N in $SEQ_LENS; do
    run_prove "BERT_ATTN_WITNESS_DIR" \
        "outputs_n${N}_L4_h256_bert" \
        "bert_attn_onlyeq_single_layer_test" \
        "bert_attn_onlyeq_n${N}_L4_h256"
done

# ============================================================================
# === 종합 결과 ==============================================================
# ============================================================================
ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================"
echo " measure_onlyeq/measure_mini.sh 완료"
echo " 시작: $STARTED_AT"
echo " 종료: $ENDED_AT"
echo " 총 실행: $TOTAL_RUNS"
echo " 성공:    $PASS_RUNS"
echo " 실패:    $FAIL_RUNS"
echo " 스킵:    $SKIP_RUNS"
echo "============================================"
echo ""
echo "=== 결과 요약 명령 ==="
echo ""
echo "# per-layer 벤치마크 (JSON):"
echo "  grep BENCHMARK_RESULT ${LOG_DIR}/prove_*.log"
echo ""
echo "# peak RSS (메모리, bytes):"
echo "  grep 'maximum resident set size' ${LOG_DIR}/prove_*.time"
echo ""
echo "# 전체 wall time per run:"
echo "  grep 'total_wall_ms' ${LOG_DIR}/prove_*.log"
echo ""
echo "# 실패한 run 확인:"
echo "  grep '\\[FAIL\\]' ${LOG_DIR}/../master_*.log"
echo ""
