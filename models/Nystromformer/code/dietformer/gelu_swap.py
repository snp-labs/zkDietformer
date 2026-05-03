"""
HF 모델 (BERT, Nystromformer) 의 표준 GELU (gelu_new tanh-based 또는 erf-based)
를 본 모듈의 SigmoidGELU 로 교체

본 swap 의 영향:
- forward 출력값이 *미세하게* 달라짐 (~1% 정도, sigmoid 근사 vs 정확 GELU 차이)
- 모델은 여전히 well-defined; 

지원 model_type:
- 'bert'         → model.bert.encoder.layer[i].intermediate.intermediate_act_fn
- 'nystromformer'→ model.nystromformer.encoder.layer[i].intermediate.intermediate_act_fn
"""
from __future__ import annotations

import torch
import torch.nn as nn

class SigmoidGELU(nn.Module):
    # 회로 친화 GELU 근사: f(x) = x · σ(1.702 · x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


_SUPPORTED_MODEL_TYPES = ("bert", "nystromformer")


def replace_activation_with_sigmoid_gelu(
    model: nn.Module, model_type: str,
) -> int:
    """
    `model.{model_type}.encoder.layer[i].intermediate.intermediate_act_fn` 를
    SigmoidGELU 인스턴스로 교체.

    Returns:
        교체된 layer 수 (config.num_hidden_layers 와 같아야 함).

    Raises:
        ValueError: model_type 이 지원되지 않거나, 모델 구조가 예상과 다름.
        RuntimeError: 내부 카운트 불일치 (보통 발생 안 함).

    설계 선택:
    - 모든 layer 에 *같은* SigmoidGELU 인스턴스 부착 (stateless 이므로 안전,
      메모리 12× 절약). 인스턴스 교체로 layer 별 swap 회수도 가능.
    """
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. "
            f"Expected one of {_SUPPORTED_MODEL_TYPES}."
        )

    if not hasattr(model, model_type):
        raise ValueError(
            f"Model has no attribute {model_type!r}. "
            f"Did you pass the right model_type for this model? "
            f"For BertForMaskedLM use 'bert'; for NystromformerForMaskedLM use 'nystromformer'."
        )

    backbone = getattr(model, model_type)
    if not hasattr(backbone, "encoder") or not hasattr(backbone.encoder, "layer"):
        raise ValueError(
            f"model.{model_type}.encoder.layer 가 존재하지 않음. "
            f"예상 백본 구조: {model_type}.encoder.layer[i].intermediate.intermediate_act_fn"
        )

    layers = backbone.encoder.layer
    expected_n = len(layers)

    # 단일 인스턴스 — stateless 이므로 모든 layer 가 공유해도 무방.
    new_act = SigmoidGELU()

    swapped = 0
    for i, layer in enumerate(layers):
        if not hasattr(layer, "intermediate") or not hasattr(
            layer.intermediate, "intermediate_act_fn",
        ):
            raise ValueError(
                f"Layer {i} 에 intermediate.intermediate_act_fn 속성이 없음. "
                f"HF 모델 구조 가정과 다름."
            )
        layer.intermediate.intermediate_act_fn = new_act # 교체
        swapped += 1

    if swapped != expected_n:
        raise RuntimeError(
            f"Internal: swapped={swapped}, expected={expected_n}"
        )

    return swapped
