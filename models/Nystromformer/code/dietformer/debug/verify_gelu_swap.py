from __future__ import annotations

import os
import sys

# 모듈 디렉터리 (gelu_swap.py 등) 를 sys.path 에 추가: debug/ → 1단계 위
HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(HERE)
sys.path.insert(0, MODULE_DIR)


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    import torch
    from gelu_swap import SigmoidGELU, replace_activation_with_sigmoid_gelu

    failures: list[str] = []

    # ── [1] SigmoidGELU numerical correctness ────────────────────────
    _section("[1] SigmoidGELU numerical correctness")
    x = torch.linspace(-5, 5, 100)
    out_module = SigmoidGELU()(x)
    out_inline = x * torch.sigmoid(1.702 * x)
    max_abs_diff = (out_module - out_inline).abs().max().item()
    print(f"  input: torch.linspace(-5, 5, 100)")
    print(f"  max_abs_diff(SigmoidGELU(x), x * σ(1.702x)) = {max_abs_diff}")
    if max_abs_diff == 0.0:
        print("  ✅ PASS — exact match")
    else:
        failures.append(f"[1] SigmoidGELU mismatch: {max_abs_diff}")
        print("  ❌ FAIL")

    # ── [2] BERT swap ─────────────────────────────────────────────────
    _section("[2] BERT-base-uncased swap")
    from transformers import BertForMaskedLM
    print("  loading bert-base-uncased ...")
    model_b = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model_b.eval()
    n_layers_b = model_b.config.num_hidden_layers

    pre_type_b = type(
        model_b.bert.encoder.layer[0].intermediate.intermediate_act_fn
    ).__name__
    print(f"  pre-swap type:  {pre_type_b}")

    n_swapped_b = replace_activation_with_sigmoid_gelu(model_b, "bert")
    print(f"  swapped count:  {n_swapped_b} (expected {n_layers_b})")

    post_type_b = type(
        model_b.bert.encoder.layer[0].intermediate.intermediate_act_fn
    ).__name__
    print(f"  post-swap type: {post_type_b}")

    all_swapped_b = all(
        isinstance(model_b.bert.encoder.layer[i].intermediate.intermediate_act_fn, SigmoidGELU)
        for i in range(n_layers_b)
    )
    print(f"  all {n_layers_b} layers SigmoidGELU? {all_swapped_b}")

    if n_swapped_b == n_layers_b and post_type_b == "SigmoidGELU" and all_swapped_b:
        print("  ✅ PASS")
    else:
        failures.append("[2] BERT swap incomplete")
        print("  ❌ FAIL")

    # ── [3] Nystromformer swap ────────────────────────────────────────
    _section("[3] uw-madison/nystromformer-512 swap")
    from transformers import NystromformerForMaskedLM
    print("  loading uw-madison/nystromformer-512 ...")
    model_n = NystromformerForMaskedLM.from_pretrained("uw-madison/nystromformer-512")
    model_n.eval()
    n_layers_n = model_n.config.num_hidden_layers

    pre_type_n = type(
        model_n.nystromformer.encoder.layer[0].intermediate.intermediate_act_fn
    ).__name__
    print(f"  pre-swap type:  {pre_type_n}")

    n_swapped_n = replace_activation_with_sigmoid_gelu(model_n, "nystromformer")
    print(f"  swapped count:  {n_swapped_n} (expected {n_layers_n})")

    post_type_n = type(
        model_n.nystromformer.encoder.layer[0].intermediate.intermediate_act_fn
    ).__name__
    print(f"  post-swap type: {post_type_n}")

    all_swapped_n = all(
        isinstance(model_n.nystromformer.encoder.layer[i].intermediate.intermediate_act_fn, SigmoidGELU)
        for i in range(n_layers_n)
    )
    print(f"  all {n_layers_n} layers SigmoidGELU? {all_swapped_n}")

    if n_swapped_n == n_layers_n and post_type_n == "SigmoidGELU" and all_swapped_n:
        print("  ✅ PASS")
    else:
        failures.append("[3] Nystromformer swap incomplete")
        print("  ❌ FAIL")

    # ── [4] Wrong model_type ──────────────────────────────────────────
    _section("[4] Wrong model_type → ValueError")
    try:
        replace_activation_with_sigmoid_gelu(model_b, "invalid")
        failures.append("[4] no ValueError on wrong model_type")
        print("  ❌ FAIL — should have raised ValueError")
    except ValueError as e:
        msg = str(e)
        print(f"  caught ValueError: {msg[:90]}{'...' if len(msg) > 90 else ''}")
        if "Unsupported model_type" in msg or "invalid" in msg:
            print("  ✅ PASS")
        else:
            failures.append("[4] ValueError raised but message unclear")
            print("  ⚠️  message unexpected")

    # ── [5] dtype 보존 ────────────────────────────────────────────────
    _section("[5] dtype preservation")
    bert_dtype = next(model_b.parameters()).dtype
    nyst_dtype = next(model_n.parameters()).dtype
    print(f"  BERT model.dtype:        {bert_dtype}")
    print(f"  Nystromformer model.dtype: {nyst_dtype}")

    test_in_f32 = torch.randn(8, dtype=torch.float32)
    test_out_b = model_b.bert.encoder.layer[0].intermediate.intermediate_act_fn(test_in_f32)
    test_out_n = model_n.nystromformer.encoder.layer[0].intermediate.intermediate_act_fn(test_in_f32)
    print(f"  SigmoidGELU(f32 input) → BERT layer:    dtype={test_out_b.dtype}")
    print(f"  SigmoidGELU(f32 input) → Nystrom layer: dtype={test_out_n.dtype}")

    dtype_ok = (
        bert_dtype == torch.float32
        and nyst_dtype == torch.float32
        and test_out_b.dtype == torch.float32
        and test_out_n.dtype == torch.float32
    )
    if dtype_ok:
        print("  ✅ PASS — float32 preserved everywhere")
    else:
        failures.append("[5] dtype changed unexpectedly")
        print("  ❌ FAIL")

    # ── 종합 ─────────────────────────────────────────────────────────
    _section("Summary")
    if not failures:
        print("  ✅ All gelu_swap verification tests PASSED")
        return 0
    print("  ❌ Failures:")
    for f in failures:
        print(f"     - {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
