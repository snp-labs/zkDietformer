from __future__ import annotations

import math
import types
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoConfig, NystromformerForMaskedLM
from transformers.models.nystromformer.modeling_nystromformer import (
    NystromformerSelfAttention,
)


# ── Capture registry ─────────────────────────────────────────────────────────
class NystromCaptureRegistry:
    """`layer_name → {npz key → numpy array}` 컨테이너.

    apply_monkey_patch 가 반환하며, 각 patched forward 가 capture 를 push.
    forward 직후 caller 가 registry.all() 또는 registry.get(layer_name) 으로 조회.
    """

    def __init__(self) -> None:
        self._captures: dict[str, dict[str, np.ndarray]] = {}

    def reset(self) -> None:
        """forward 직전 호출. 이전 batch 캡처 삭제."""
        self._captures.clear()

    def get(self, layer_name: str) -> dict[str, np.ndarray]:
        """특정 layer 의 capture (없으면 빈 dict)."""
        return dict(self._captures.get(layer_name, {}))

    def all(self) -> dict[str, dict[str, np.ndarray]]:
        """모든 layer 의 capture 사본."""
        return {k: dict(v) for k, v in self._captures.items()}

    def __contains__(self, layer_name: str) -> bool:
        return layer_name in self._captures

    def __len__(self) -> int:
        return len(self._captures)


# ── Patched forward ──────────────────────────────────────────────────────────
def _patched_forward(self, hidden_states, attention_mask=None, output_attentions=False):
    """Fork of NystromformerSelfAttention.forward (transformers 4.57.3).

    HF 원본의 else 분기 (Nyström landmark) 만 실행 + 모든 중간변수 capture.
    forward 출력은 HF 원본 else 분기 와 numerical 동일 (단, in-place += 만 = + 로 변경).
    """
    capture: dict[str, np.ndarray] = {}

    batch_size, seq_length, _ = hidden_states.shape

    # ── Projection 입력/가중치 캡처 (Phase 0 witness 생성용) ─────────────────
    capture["hidden_states"] = hidden_states.detach().cpu().numpy()  # [B, S, H*D]

    capture["W_q"] = self.query.weight.data.cpu().numpy()     # [H*D, H*D]
    capture["W_k"] = self.key.weight.data.cpu().numpy()
    capture["W_v"] = self.value.weight.data.cpu().numpy()
    capture["b_q"] = self.query.bias.data.cpu().numpy() if self.query.bias is not None else None
    capture["b_k"] = self.key.bias.data.cpu().numpy() if self.key.bias is not None else None
    capture["b_v"] = self.value.bias.data.cpu().numpy() if self.value.bias is not None else None

    capture["W_o"] = self._output_dense.weight.data.cpu().numpy()  # [H*D, H*D]
    capture["b_o"] = self._output_dense.bias.data.cpu().numpy() if self._output_dense.bias is not None else None

    if self.conv_kernel_size is not None:
        capture["W_conv"] = self.conv.weight.data.cpu().numpy()  # [H, 1, K_conv, 1]

    # Q/K/V projection + view + transpose (HF 원본 그대로)
    query_layer = (
        self.query(hidden_states)
        .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
    )
    key_layer = (
        self.key(hidden_states)
        .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
    )
    value_layer = (
        self.value(hidden_states)
        .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
    )

    # ── V 캡처 (mask 사전 곱 없음, Q3 결정 a) ─────────────────────────────
    capture["V"] = value_layer.detach().cpu().numpy()

    # Q/K scaling (HF 원본 그대로)
    query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
    key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))

    # ── Q_proc, K_proc 캡처 (scaling 후) ───────────────────────────────────
    capture["Q_proc"] = query_layer.detach().cpu().numpy()
    capture["K_proc"] = key_layer.detach().cpu().numpy()

    # ── Safety guards ──────────────────────────────────────────────────────
    # if-branch (full attention) 는 본 fork 의 의미상 모순 — override 누락 시 즉시 실패.
    if self.num_landmarks == self.seq_len:
        raise RuntimeError(
            f"NystromformerSelfAttention monkey-patch invoked with "
            f"num_landmarks ({self.num_landmarks}) == seq_len ({self.seq_len}). "
            f"Nyström landmark 분기를 강제하려면 두 값을 다르게 설정하세요. "
            f"권장: load_nystromformer_with_landmarks_override(num_landmarks=30, seq_len=510)."
        )

    segment_size = self.seq_len // self.num_landmarks
    expected_input_len = self.num_landmarks * segment_size
    if seq_length != expected_input_len:
        raise ValueError(
            f"input seq_length ({seq_length}) != num_landmarks × segment_size "
            f"({self.num_landmarks} × {segment_size} = {expected_input_len}). "
            f"config.segment_means_seq_len ({self.seq_len}) 가 입력 길이와 일치하도록 설정하세요."
        )

    # ── Q_landmarks, K_landmarks (segment-mean) ───────────────────────────
    q_landmarks = query_layer.reshape(
        -1,
        self.num_attention_heads,
        self.num_landmarks,
        segment_size,
        self.attention_head_size,
    ).mean(dim=-2)
    k_landmarks = key_layer.reshape(
        -1,
        self.num_attention_heads,
        self.num_landmarks,
        segment_size,
        self.attention_head_size,
    ).mean(dim=-2)
    capture["Q_landmarks"] = q_landmarks.detach().cpu().numpy()
    capture["K_landmarks"] = k_landmarks.detach().cpu().numpy()

    # ── kernel_1: softmax(Q_proc @ K_landmarks.T) ─────────────────────────
    kernel_1_scores = torch.matmul(query_layer, k_landmarks.transpose(-1, -2))
    kernel_1 = torch.nn.functional.softmax(kernel_1_scores, dim=-1)
    capture["kernel_1_scores"] = kernel_1_scores.detach().cpu().numpy()
    capture["kernel_1"] = kernel_1.detach().cpu().numpy()

    # ── kernel_2: softmax(Q_landmarks @ K_landmarks.T) ────────────────────
    kernel_2_scores = torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2))
    kernel_2 = torch.nn.functional.softmax(kernel_2_scores, dim=-1)
    capture["kernel_2_scores"] = kernel_2_scores.detach().cpu().numpy()
    capture["kernel_2"] = kernel_2.detach().cpu().numpy()

    # ── kernel_3: mask 적용 *전* score 보관, mask 적용 후 softmax ─────────
    kernel_3_scores = torch.matmul(q_landmarks, key_layer.transpose(-1, -2))
    capture["kernel_3_scores"] = kernel_3_scores.detach().cpu().numpy()

    if attention_mask is not None:
        attention_scores_masked = kernel_3_scores + attention_mask
    else:
        attention_scores_masked = kernel_3_scores
    kernel_3 = nn.functional.softmax(attention_scores_masked, dim=-1)
    capture["kernel_3"] = kernel_3.detach().cpu().numpy()

    # ── inv_kernel_2: iterative_inv(kernel_2) ─────────────────────────────
    inv_kernel_2 = self.iterative_inv(kernel_2)
    capture["inv_kernel_2"] = inv_kernel_2.detach().cpu().numpy()

    # ── mid = kernel_1 @ inv_kernel_2  (HF 원본의 attention_probs 와 동일) ─
    mid = torch.matmul(kernel_1, inv_kernel_2)
    capture["mid"] = mid.detach().cpu().numpy()

    # ── temp = kernel_3 @ V  (HF 원본의 new_value_layer 와 동일) ───────────
    temp = torch.matmul(kernel_3, value_layer)
    capture["temp"] = temp.detach().cpu().numpy()

    # ── X = mid @ temp  (conv 합산 *전* context_layer) ────────────────────
    context_layer = torch.matmul(mid, temp)
    capture["X"] = context_layer.detach().cpu().numpy()

    # ── conv 처리: in-place += 대신 = + 사용 (X / X_final 분리 캡처) ───────
    if self.conv_kernel_size is not None:
        conv_term = self.conv(value_layer)
        capture["conv_term"] = conv_term.detach().cpu().numpy()
        context_layer = context_layer + conv_term
        capture["X_final"] = context_layer.detach().cpu().numpy()

    # ── 최종 reshape (HF 원본 그대로) ─────────────────────────────────────
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    # ── Output projection 입력 캡처 (reshape 후 [B, S, H*D]) ────────────
    capture["attn_output_for_proj"] = context_layer.detach().cpu().numpy()

    # ── capture 등록 ─────────────────────────────────────────────────────
    self._capture_registry._captures[self._capture_layer_name] = capture

    # HF 원본 반환 형식. mid 가 attention_probs 역할.
    outputs = (context_layer, mid) if output_attentions else (context_layer,)
    return outputs


# ── Public API ───────────────────────────────────────────────────────────────
def valid_landmark_choices(seq_len: int, max_ratio: int = 4) -> list[int]:
    """seq_len 의 약수 중 num_landmarks 후보 (Nyström 비율 ≥ max_ratio 기준)."""
    return sorted(
        d for d in range(2, seq_len)
        if seq_len % d == 0 and d <= seq_len // max_ratio
    )


def load_nystromformer_with_landmarks_override(
    model_id: str = "uw-madison/nystromformer-512",
    num_landmarks: int = 30,
    seq_len: int = 510,
) -> NystromformerForMaskedLM:
    """
    HF Nystromformer 를 num_landmarks override 와 함께 로드.

    config override:
        config.num_landmarks       = num_landmarks
        config.segment_means_seq_len = seq_len

    constraint: seq_len % num_landmarks == 0 (reshape 호환)

    Args:
        model_id: HF Hub model ID.
        num_landmarks: 본 sweep 변수 (예: 30, 51).
        seq_len: 측정 입력 길이 (예: 510). config.segment_means_seq_len 와 일치.

    Returns:
        NystromformerForMaskedLM with overridden config.
    """
    if seq_len % num_landmarks != 0:
        suggestions = valid_landmark_choices(seq_len)
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by num_landmarks ({num_landmarks}). "
            f"가능한 num_landmarks (seq_len={seq_len}): {suggestions}"
        )

    config = AutoConfig.from_pretrained(model_id)
    config.num_landmarks = num_landmarks
    config.segment_means_seq_len = seq_len

    model = NystromformerForMaskedLM.from_pretrained(model_id, config=config)
    return model


def load_small_nystromformer(
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    num_landmarks: int = 32,
    seq_len: int = 128,
    vocab_size: int = 50265,
) -> NystromformerForMaskedLM:
    """
    Random init Nystromformer — 회로 비용 측정 전용.

    pretrained 가중치 없이 config 만으로 모델 생성.
    회로 constraint 수 / prove time 은 architecture 만 의존하므로
    가중치가 random 이어도 측정 결과 동일.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})."
        )
    if seq_len % num_landmarks != 0:
        suggestions = valid_landmark_choices(seq_len)
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by num_landmarks ({num_landmarks}). "
            f"Valid choices: {suggestions}"
        )

    config = AutoConfig.from_pretrained("uw-madison/nystromformer-512")
    config.hidden_size = hidden_size
    config.intermediate_size = hidden_size * 4
    config.num_hidden_layers = num_hidden_layers
    config.num_attention_heads = num_attention_heads
    config.num_landmarks = num_landmarks
    config.segment_means_seq_len = seq_len
    config.vocab_size = vocab_size
    config.max_position_embeddings = max(seq_len, 510)

    model = NystromformerForMaskedLM(config)
    return model


def apply_monkey_patch(
    model: nn.Module,
    num_landmarks_override: Optional[int] = None,
    seq_len_override: Optional[int] = None,
) -> NystromCaptureRegistry:
    """
    NystromformerSelfAttention.forward 를 fork 한 _patched_forward 로 교체.
    모든 layer 에 적용 + capture registry 반환.

    Args:
        model: NystromformerForMaskedLM (또는 동일 백본).
        num_landmarks_override: 옵션. self.num_landmarks 를 layer 측에서 덮어쓰기.
            load_nystromformer_with_landmarks_override 사용 시 None.
        seq_len_override: 옵션. self.seq_len 도 함께 덮어쓰기.

    Returns:
        NystromCaptureRegistry.
    """
    if not hasattr(model, "nystromformer"):
        raise ValueError(
            "model.nystromformer 가 없습니다. NystromformerForMaskedLM 인스턴스인지 확인."
        )

    layers = model.nystromformer.encoder.layer
    expected = len(layers)

    registry = NystromCaptureRegistry()
    n_patched = 0
    for name, module in model.named_modules():
        if isinstance(module, NystromformerSelfAttention):
            if num_landmarks_override is not None:
                module.num_landmarks = num_landmarks_override
            if seq_len_override is not None:
                module.seq_len = seq_len_override

            module._capture_registry = registry
            module._capture_layer_name = name

            # Output projection 가중치 참조 (parent.output.dense)
            layer_idx = int(name.split('.')[3])
            parent_attn = layers[layer_idx].attention
            module._output_dense = parent_attn.output.dense

            module.forward = types.MethodType(_patched_forward, module)
            n_patched += 1

    if n_patched != expected:
        raise RuntimeError(
            f"monkey-patched {n_patched} layers, expected {expected}."
        )

    return registry
