#!/usr/bin/env bash
# ============================================================================
# measure_all.sh — 8 데이터 폴더 × 6 회로 전체 layer prove + 메모리 측정
#
# 환경: macOS (Darwin, ARM64 Mac Studio, 128 GB)
# 용도: SSH 접속 후 nohup 백그라운드 실행 → 퇴근 후 자동 완료
#
# 실행 방법:
#   cd /Users/zkrypto12/zkdietformer/Dietformer/circuits
#   chmod +x measure_all.sh
#   nohup caffeinate -i bash measure_all.sh > logs/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo "PID: $!"
#
# 결과 확인:
#   tail -f logs/master_*.log                                    # 진행 상황
#   grep BENCHMARK_RESULT logs/prove_*.log                       # per-layer 벤치마크
#   grep "maximum resident set size" logs/prove_*.time           # peak RSS (bytes)
# ============================================================================
set -euo pipefail

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
echo " measure_all.sh started: $STARTED_AT"
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

# ── 측정 함수 ────────────────────────────────────────────────────────────────
# run_prove WITNESS_VAR DATA_DIR TEST_FN TAG
#
# - WITNESS_VAR: 환경 변수 이름 (예: LAYERNORM_WITNESS_DIR)
# - DATA_DIR:    데이터 폴더명 (예: outputs_n512_m32_h768_nystrom)
# - TEST_FN:     테스트 함수명 (예: nystrom_mlp_hf_single_layer_test)
# - TAG:         로그 파일 태그 (예: nystrom_mlp_n512)
run_prove() {
    local WITNESS_VAR="$1"
    local DATA_DIR="$2"
    local TEST_FN="$3"
    local TAG="$4"

    local FULL_DATA_PATH="$DATA_BASE/$DATA_DIR"
    local LOG_FILE="logs/prove_${TAG}.log"
    local TIME_FILE="logs/prove_${TAG}.time"

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

    # *_LAYER_IDX 를 설정하지 않음 → all-layers 모드
    # stdout → prove log, stderr → time log (peak RSS)
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
    LAYER_COUNT=$(grep -c "BENCHMARK_RESULT" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "  Layers proved: $LAYER_COUNT"
    echo ""
}

# ============================================================================
# === Nystrom MLP (4 seq_len) ================================================
# ============================================================================
echo ""
echo "################################################################"
echo "# Nystrom MLP (Freivalds)"
echo "################################################################"
for N in 512 1024 2048 4096; do
    run_prove "LAYERNORM_WITNESS_DIR" \
        "outputs_n${N}_m32_L4_h512_nystrom" \
        "nystrom_mlp_hf_single_layer_test" \
        "nystrom_mlp_n${N}_m32_L4_h512"
done

# ============================================================================
# === BERT MLP (4 seq_len) ===================================================
# ============================================================================
echo ""
echo "################################################################"
echo "# BERT MLP (Freivalds)"
echo "################################################################"
for N in 512 1024 2048 4096; do
    run_prove "BERT_MLP_WITNESS_DIR" \
        "outputs_n${N}_L4_h512_bert" \
        "bert_mlp_hf_single_layer_test" \
        "bert_mlp_n${N}_L4_h512"
done

# ============================================================================
# === Nystrom Attention Freivalds (4 seq_len) ================================
# ============================================================================
echo ""
echo "################################################################"
echo "# Nystrom Attention (Freivalds)"
echo "################################################################"
for N in 512 1024 2048 4096; do
    run_prove "NYSTROM_ATTN_WITNESS_DIR" \
        "outputs_n${N}_m32_L4_h512_nystrom" \
        "nystrom_attn_hf_single_layer_test" \
        "nystrom_attn_freivalds_n${N}_m32_L4_h512"
done

# ============================================================================
# === BERT Attention Freivalds (4 seq_len, n>=2048 메모리 주의) ===============
# ============================================================================
echo ""
echo "################################################################"
echo "# BERT Attention (Freivalds)"
echo "################################################################"
for N in 512 1024 2048 4096; do
    # n>=2048 BERT Attention Freivalds 는 메모리 폭발 가능 (128GB 초과 예상).
    # 필요 시 아래 skip 조건 주석 해제.
    # if [ "$N" -ge 2048 ]; then
    #     echo "  [SKIP] BERT Attn Freivalds n=$N (메모리 부족 예상)"
    #     continue
    # fi
    run_prove "BERT_ATTN_WITNESS_DIR" \
        "outputs_n${N}_L4_h512_bert" \
        "bert_attn_hf_single_layer_test" \
        "bert_attn_freivalds_n${N}_L4_h512"
done

# ============================================================================
# === Nystrom Attention Naive (4 seq_len) ====================================
# ============================================================================
echo ""
echo "################################################################"
echo "# Nystrom Attention (Naive)"
echo "################################################################"
for N in 512 1024 2048 4096; do
    run_prove "NAIVE_NYSTROM_ATTN_WITNESS_DIR" \
        "outputs_n${N}_m32_L4_h512_nystrom" \
        "nystrom_attn_naive_single_layer_test" \
        "nystrom_attn_naive_n${N}_m32_L4_h512"
done

# ============================================================================
# === BERT Attention Naive (4 seq_len, n>=2048 메모리 주의) ===================
# ============================================================================
echo ""
echo "################################################################"
echo "# BERT Attention (Naive)"
echo "################################################################"
for N in 512 1024 2048 4096; do
    # n>=2048 BERT Attention Naive 는 메모리 폭발 가능.
    # 필요 시 아래 skip 조건 주석 해제.
    # if [ "$N" -ge 2048 ]; then
    #     echo "  [SKIP] BERT Attn Naive n=$N (메모리 부족 예상)"
    #     continue
    # fi
    run_prove "NAIVE_ROBERTA_ATTN_WITNESS_DIR" \
        "outputs_n${N}_L4_h512_bert" \
        "bert_attn_naive_single_layer_test" \
        "bert_attn_naive_n${N}_L4_h512"
done

# ============================================================================
# === 종합 결과 ==============================================================
# ============================================================================
ENDED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================"
echo " measure_all.sh 완료"
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
