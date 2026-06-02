"""
zkDietFormer 측정용 공통 fixture — BERT / Nystromformer 동일 입력.

본 모듈은 두 evaluation 스크립트 (run_bert_validation.py, run_validation.py) 가
import 하여 동일한 raw text 를 사용하도록 한다. 이로써:
- 토크나이저는 모델별 native (BERT WordPiece / Albert SP) — token sequence 는 다름
- 입력 raw text 는 동일 → 측정 조건 정렬 (논문 §7 baseline 대조)

토큰 길이 메타데이터는 fixture 작성 시점 (2026-04-28, transformers 4.57.3,
conda env 'dietformer') 에서 측정한 reference 값. 환경/버전 차이로 길이가
다르게 나오면 토크나이저 drift 가능성 — 무시하지 말고 점검할 것.

NPZ 캡처 시 batch index 가 곧 sample index 가 된다 (순서 보존). 따라서
SAMPLES 의 순서를 변경하면 npz 파일명의 batch{i} 가 다른 sample 을 가리키게
되므로 변경 전 깊이 검토 필요.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EvalSample:
    """단일 평가 입력. frozen=True 로 immutability 강제."""
    text: str
    bert_token_len: int      # BertTokenizerFast("bert-base-uncased") 측정값
    nystrom_token_len: int   # AutoTokenizer("uw-madison/nystromformer-512") 측정값


# ── Fixture ────────────────────────────────────────────────────────────────
# 절대 변경 금지: SAMPLES 의 순서/내용 변경 시 npz batch index 가 다른
# sample 을 가리키게 되어 측정 비교가 깨진다.
#
# 토큰 길이 reference (transformers==4.57.3, 2026-04-28 측정):
#   - BERT max  = 20 (#10)
#   - Nystrom max = 21 (#10)
#   - 둘 다 max_seq_len=510 의 5% 미만으로 안전.
SAMPLES: Tuple[EvalSample, ...] = (
    EvalSample(
        text="The capital of France is [MASK].",
        bert_token_len=9, nystrom_token_len=10,
    ),
    EvalSample(
        text="Machine learning is a subset of [MASK] intelligence.",
        bert_token_len=11, nystrom_token_len=11,
    ),
    EvalSample(
        text="Photosynthesis converts [MASK] dioxide into oxygen.",
        bert_token_len=11, nystrom_token_len=13,
    ),
    EvalSample(
        text="The quick brown fox jumps over the lazy [MASK].",
        bert_token_len=12, nystrom_token_len=13,
    ),
    EvalSample(
        text="She opened the book and began to [MASK] silently.",
        bert_token_len=12, nystrom_token_len=12,
    ),
    EvalSample(
        text="Quantum computers use qubits instead of classical [MASK].",
        bert_token_len=13, nystrom_token_len=14,
    ),
    EvalSample(
        text="The ancient ruins were discovered by an [MASK] team.",
        bert_token_len=12, nystrom_token_len=12,
    ),
    EvalSample(
        text="Climate change is one of the most pressing [MASK] of our time.",
        bert_token_len=15, nystrom_token_len=15,
    ),
    EvalSample(
        text="He grabbed his umbrella because the forecast predicted [MASK].",
        bert_token_len=12, nystrom_token_len=13,
    ),
    EvalSample(
        text=(
            "Newton's third law states that for every action there is an "
            "equal and opposite [MASK]."
        ),
        bert_token_len=20, nystrom_token_len=21,
    ),
)

# ── 편의 export ────────────────────────────────────────────────────────────
SAMPLE_TEXTS: Tuple[str, ...] = tuple(s.text for s in SAMPLES)
"""text 만 추출한 튜플. 두 evaluation 스크립트가 import 해서 사용."""

NUM_SAMPLES: int = len(SAMPLES)
"""고정 sample 수 (현재 10)."""

# 측정 환경에서의 통일 max_seq_len.
#
# 변경 (2026-04-28, Step 2A 후): 510 → 448.
# 사유: HF Nystromformer 의 num_landmarks=64 sweep 점이 사전학습 config 정합
# (segment_means_seq_len=64, num_landmarks=64) 의 자연스러운 후보인데, 이를
# Nyström 분기 활성화 조건 (num_landmarks ≠ seq_len) + reshape 호환 조건
# (seq_len % num_landmarks == 0) 모두 만족시키려면 seq_len 이 64 의 배수여야 함.
# 510 의 약수에 64 없음. 448 = 64 × 7 → num_landmarks=64 valid.
#
# 448 의 sweep 후보 (4:1 비율 이상): {14, 16, 28, 32, 56, 64, 112}.
# fixture 의 max 토큰 길이는 21 (사용자 측정) — 448 padding 충분.
MAX_SEQ_LEN: int = 2048
"""양 모델 공통 truncation 한계. 448 = 64 × 7 — num_landmarks ∈ {14, 16, 28, 32, 56, 64, 112} sweep 호환."""


def assert_fixture_invariants() -> None:
    """가벼운 self-check. evaluation 스크립트 import 후 한 번 호출 권장."""
    assert len(SAMPLES) == NUM_SAMPLES
    assert all(isinstance(s, EvalSample) for s in SAMPLES)
    assert all(s.bert_token_len <= MAX_SEQ_LEN for s in SAMPLES), \
        "BERT token length exceeds MAX_SEQ_LEN"
    assert all(s.nystrom_token_len <= MAX_SEQ_LEN for s in SAMPLES), \
        "Nystrom token length exceeds MAX_SEQ_LEN"
