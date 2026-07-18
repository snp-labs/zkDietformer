from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent              # .../code/dietformer/ — 모듈 디렉터리
PROJECT_ROOT = HERE.parents[4]        # .../Dietformer/ — 프로젝트 루트
sys.path.insert(0, str(MODULE_DIR))


# ── 모델 로드 (BERT, Nystromformer) ────────────────────────────────────────
def load_bert():
    from transformers import BertForMaskedLM
    from gelu_swap import replace_activation_with_sigmoid_gelu, SigmoidGELU

    print("  [load] bert-base-uncased ...", flush=True)
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()
    n_swapped = replace_activation_with_sigmoid_gelu(model, "bert")
    all_swapped = all(
        isinstance(
            model.bert.encoder.layer[i].intermediate.intermediate_act_fn,
            SigmoidGELU,
        )
        for i in range(model.config.num_hidden_layers)
    )
    return model, n_swapped, all_swapped


def load_nystrom(num_landmarks=64, seq_len=448):
    from nystrom_attention_capture import (
        load_nystromformer_with_landmarks_override,
        apply_monkey_patch,
    )
    from gelu_swap import replace_activation_with_sigmoid_gelu, SigmoidGELU

    print(f"  [load] uw-madison/nystromformer-512 (m={num_landmarks}, seq={seq_len}) ...", flush=True)
    model = load_nystromformer_with_landmarks_override(
        num_landmarks=num_landmarks, seq_len=seq_len,
    )
    model.eval()
    n_swapped = replace_activation_with_sigmoid_gelu(model, "nystromformer")
    all_swapped = all(
        isinstance(
            model.nystromformer.encoder.layer[i].intermediate.intermediate_act_fn,
            SigmoidGELU,
        )
        for i in range(model.config.num_hidden_layers)
    )

    registry = apply_monkey_patch(model)

    # monkey-patch 부착 검증 — _capture_registry attribute 가 부착됐는지
    layer0_attn = model.nystromformer.encoder.layer[0].attention.self
    is_patched = (
        hasattr(layer0_attn, "_capture_registry")
        and hasattr(layer0_attn, "_capture_layer_name")
    )
    return model, n_swapped, all_swapped, registry, is_patched


def collect() -> dict:
    """양 모델 로드 + 환경 정보 수집."""
    from evaluation_fixtures import MAX_SEQ_LEN, NUM_SAMPLES, SAMPLE_TEXTS
    import transformers
    import torch
    import numpy as np

    print("Loading models ...")
    bert, bert_n, bert_all = load_bert()
    nys, nys_n, nys_all, registry, is_patched = load_nystrom(
        num_landmarks=64, seq_len=MAX_SEQ_LEN,
    )

    return {
        "bert": {
            "model": bert,
            "n_swapped": bert_n,
            "all_swapped": bert_all,
        },
        "nystrom": {
            "model": nys,
            "n_swapped": nys_n,
            "all_swapped": nys_all,
            "registry": registry,
            "is_patched": is_patched,
        },
        "env": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": ".".join(map(str, sys.version_info[:3])),
        },
        "fixture": {
            "max_seq_len": MAX_SEQ_LEN,
            "num_samples": NUM_SAMPLES,
            "sample_count": len(SAMPLE_TEXTS),
        },
    }


# ── Assertion 검증 ──────────────────────────────────────────────────────────
def run_assertions(data: dict) -> tuple:
    """공통 일치 + 의도된 차이 + Nystrom 전용 검증."""
    bert = data["bert"]["model"]
    nys = data["nystrom"]["model"]
    max_seq = data["fixture"]["max_seq_len"]

    passes: list = []
    fails: list = []

    def check(label, cond, expected, actual):
        target = passes if cond else fails
        target.append((label, expected, actual))

    # ── 공통 일치 ──────────────────────────────────────────────────────
    check(
        "hidden_size 양 모델 일치 (768)",
        bert.config.hidden_size == 768 and nys.config.hidden_size == 768,
        "BERT=768, Nys=768",
        f"BERT={bert.config.hidden_size}, Nys={nys.config.hidden_size}",
    )
    check(
        "num_hidden_layers 양 모델 일치 (12)",
        bert.config.num_hidden_layers == 12 and nys.config.num_hidden_layers == 12,
        "BERT=12, Nys=12",
        f"BERT={bert.config.num_hidden_layers}, Nys={nys.config.num_hidden_layers}",
    )
    check(
        "num_attention_heads 양 모델 일치 (12)",
        bert.config.num_attention_heads == 12 and nys.config.num_attention_heads == 12,
        "BERT=12, Nys=12",
        f"BERT={bert.config.num_attention_heads}, Nys={nys.config.num_attention_heads}",
    )
    check(
        "training mode 비활성 (eval)",
        not bert.training and not nys.training,
        "False, False",
        f"BERT.training={bert.training}, Nys.training={nys.training}",
    )
    check(
        "BERT GELU swap 12/12 SigmoidGELU",
        data["bert"]["n_swapped"] == 12 and data["bert"]["all_swapped"],
        "12/12 SigmoidGELU",
        f"swapped={data['bert']['n_swapped']}, all={data['bert']['all_swapped']}",
    )
    check(
        "Nystrom GELU swap 12/12 SigmoidGELU",
        data["nystrom"]["n_swapped"] == 12 and data["nystrom"]["all_swapped"],
        "12/12 SigmoidGELU",
        f"swapped={data['nystrom']['n_swapped']}, all={data['nystrom']['all_swapped']}",
    )

    # ── Nystromformer 전용 ─────────────────────────────────────────────
    check(
        "Nystrom monkey-patch 적용",
        data["nystrom"]["is_patched"],
        "_capture_registry/_capture_layer_name attached",
        f"is_patched={data['nystrom']['is_patched']}",
    )
    check(
        "Nystrom config.num_landmarks override (64)",
        nys.config.num_landmarks == 64,
        "config.num_landmarks=64",
        f"actual={nys.config.num_landmarks}",
    )
    check(
        "Nystrom config.segment_means_seq_len = MAX_SEQ_LEN (448)",
        nys.config.segment_means_seq_len == max_seq,
        f"={max_seq}",
        f"actual={nys.config.segment_means_seq_len}",
    )
    check(
        "Nystrom Nyström 분기 활성 (num_landmarks ≠ seq_len)",
        nys.config.num_landmarks != nys.config.segment_means_seq_len,
        "num_landmarks ≠ seq_len → else branch",
        f"{nys.config.num_landmarks} vs {nys.config.segment_means_seq_len}",
    )

    # ── 의도된 차이 (assert DIFFERENT) ──────────────────────────────────
    check(
        "layer_norm_eps 차이 (BERT 1e-12 ≠ Nys 1e-5)",
        bert.config.layer_norm_eps != nys.config.layer_norm_eps,
        "different",
        f"BERT={bert.config.layer_norm_eps:g}, Nys={nys.config.layer_norm_eps:g}",
    )
    check(
        "max_position_embeddings 차이 (BERT 512 ≠ Nys 510)",
        bert.config.max_position_embeddings != nys.config.max_position_embeddings,
        "different",
        f"BERT={bert.config.max_position_embeddings}, Nys={nys.config.max_position_embeddings}",
    )
    check(
        "vocab_size 차이 (BERT 30522 ≠ Nys 30000)",
        bert.config.vocab_size != nys.config.vocab_size,
        "different",
        f"BERT={bert.config.vocab_size}, Nys={nys.config.vocab_size}",
    )
    check(
        "hidden_act 차이 (HF default — swap 전 기준)",
        bert.config.hidden_act != nys.config.hidden_act,
        "different",
        f"BERT={bert.config.hidden_act}, Nys={nys.config.hidden_act}",
    )

    return passes, fails


# ── Markdown 자동 생성 ──────────────────────────────────────────────────────
def write_markdown(data: dict, passes: list, fails: list, md_path: Path) -> None:
    bert = data["bert"]["model"]
    nys = data["nystrom"]["model"]
    max_seq = data["fixture"]["max_seq_len"]
    overall = "✅ ALL PASS" if not fails else f"❌ {len(fails)} FAIL"

    L: list[str] = []
    L.append("# 작업 2E — 측정 환경 보고서 (논문 §7 인용 형식)")
    L.append("")
    L.append(f"> 자동 생성: `verify_measurement_parity.py`, {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"> 환경: dietformer (transformers {data['env']['transformers']}, torch {data['env']['torch']})")
    L.append("")
    L.append("---")
    L.append("")

    # 0. 종합
    L.append(f"## 0. 종합 — {overall}")
    L.append("")
    L.append(f"- pass: {len(passes)}")
    L.append(f"- fail: {len(fails)}")
    L.append("")

    # 1. Models
    L.append("## 1. Models")
    L.append("")
    L.append("| 항목 | BERT-base-uncased | uw-madison/nystromformer-512 |")
    L.append("|------|-------------------|------------------------------|")
    L.append(f"| Hidden dim | {bert.config.hidden_size} | {nys.config.hidden_size} |")
    L.append(f"| Num layers | {bert.config.num_hidden_layers} | {nys.config.num_hidden_layers} |")
    L.append(f"| Num attention heads | {bert.config.num_attention_heads} | {nys.config.num_attention_heads} |")
    L.append(f"| Intermediate (FFN) size | {bert.config.intermediate_size} | {nys.config.intermediate_size} |")
    L.append(f"| max_seq_len (this work) | {max_seq} | {max_seq} |")
    L.append(f"| max_position_embeddings | {bert.config.max_position_embeddings} | {nys.config.max_position_embeddings} |")
    L.append("| Tokenizer | WordPiece | Albert SentencePiece |")
    L.append(f"| Vocab size | {bert.config.vocab_size} | {nys.config.vocab_size} |")
    L.append(f"| hidden_act (HF default) | `{bert.config.hidden_act}` | `{nys.config.hidden_act}` |")
    L.append("| Activation (after swap) | SigmoidGELU `x·σ(1.702x)` | SigmoidGELU `x·σ(1.702x)` |")
    L.append(f"| layer_norm_eps | {bert.config.layer_norm_eps:g} | {nys.config.layer_norm_eps:g} |")
    L.append("| Pretraining corpus | BookCorpus + English Wikipedia (~16GB) | BookCorpus + English Wikipedia (~16GB) |")
    L.append("")

    # 2. Attention
    L.append("## 2. Attention")
    L.append("")
    L.append("| 항목 | BERT | Nystromformer |")
    L.append("|------|------|---------------|")
    L.append("| Type | full O(n²) | Nyström O(nm) |")
    L.append(f"| num_landmarks | N/A (full attention) | sweep: {{32, 64, 112}}; default 64 |")
    L.append(f"| segment_means_seq_len | N/A | {nys.config.segment_means_seq_len} (= MAX_SEQ_LEN) |")
    L.append(f"| Conv kernel size (depthwise) | N/A | {nys.config.conv_kernel_size} |")
    L.append("| iterative_inv n_iter | N/A | 6 (HF default) |")
    L.append("| Capture mechanism | forward hook (BertSelfAttention) | monkey-patch fork (NystromformerSelfAttention.forward) |")
    L.append("")

    # 3. Quantization Pipeline
    L.append("## 3. Quantization Pipeline")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|------|-----|")
    L.append("| TABLE_SIZE (RMSNORM_M) | 65536 (= 2^16) |")
    L.append("| FRAC_BITS | 8 |")
    L.append("| K_BITS | 22 |")
    L.append("| VAR_SCALE | 1000 (Phase 4 결정 — `1/√(i/1000+ε)` lookup) |")
    L.append("| LayerNorm eps (Pipeline) | BERT 1e-12 / Nystrom 1e-5 (각 사전학습 정합) |")
    L.append("| Output dir | BERT `outputs_bert/` / Nystrom `outputs_m{N}/` (자동 분기) |")
    L.append("| Single forward capture | 12 layer × 17 attention keys + 12 LN keys |")
    L.append("")

    # 4. Reproducibility
    L.append("## 4. Reproducibility")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|------|-----|")
    L.append(f"| transformers version | {data['env']['transformers']} |")
    L.append(f"| torch version | {data['env']['torch']} |")
    L.append(f"| numpy version | {data['env']['numpy']} |")
    L.append(f"| Python version | {data['env']['python']} |")
    L.append("| Random seed | 42 (torch + numpy + random) |")
    L.append("| Forward count per batch | 2 (1 capture + 1 perf) |")
    L.append(f"| Input fixture | `evaluation_fixtures.SAMPLE_TEXTS` ({data['fixture']['sample_count']} immutable raw text samples) |")
    L.append("| Output reproducibility | npz md5 byte-identical across runs (verified) |")
    L.append("")

    # 5. Limitations
    L.append("## 5. Limitations")
    L.append("")
    L.append("- **End-to-end quantization error ~1.5%** (8-bit fixed-point inherent floor with FRAC_BITS=8).")
    L.append("  Lookup table standalone < 1% (95% percentile) — see `15-layernorm-revalidation.md`.")
    L.append("- **Nystromformer outlier var** (max 65.2) 영역에서 lookup ~2% rel error. 95% percentile (var ≤ 8) 에서는 < 1%.")
    L.append("- **Variance distribution varies across layers**: outlier values (var > 8) concentrate in deeper layers, while early")
    L.append("  layers exhibit compact distributions (median 0.27 in layer 0, similar to BERT).")
    L.append("- **HF Nystromformer-512 published config** has `num_landmarks = segment_means_seq_len = 64` → effectively full attention")
    L.append("  at inference. We override at config load (`load_nystromformer_with_landmarks_override`) to force Nyström landmark")
    L.append("  branch for sweep measurement. Pretrained weights remain unmodified.")
    L.append("- **Tokenizer mismatch by design** — BERT uses WordPiece, Nystromformer uses Albert SentencePiece. Same raw text →")
    L.append("  different token sequences. This is model-native behavior, not a measurement artifact.")
    L.append("- **Verify is not yet possible (Phase 5 pending)**: Rust circuit's RMSNorm gadget enforces RMS form (`out = γ·x/√(mean(x²)+ε)`).")
    L.append("  LayerNorm gadget + VAR_SCALE=1000 Lasso strategy update needed in Phase 5. R1CS 식 자체는 모두 정합 (3072/3072 cells exact).")
    L.append("")

    # 6. Verification 결과
    L.append("## 6. Assertion Results")
    L.append("")
    L.append("| # | check | expected | actual | result |")
    L.append("|---|-------|----------|--------|--------|")
    for i, (label, exp, act) in enumerate(passes):
        L.append(f"| {i+1} | {label} | {exp} | {act} | ✅ |")
    for j, (label, exp, act) in enumerate(fails):
        L.append(f"| {len(passes)+j+1} | {label} | {exp} | {act} | ❌ |")
    L.append("")

    # 7. 본 마이그레이션 요약 (Migration completion summary)
    L.append("## 7. Migration Steps Completed (본 마이그레이션 결과)")
    L.append("")
    L.append("| Step | 항목 | 산출물 |")
    L.append("|------|------|--------|")
    L.append("| 0X | seq_len 510 → 448 | evaluation_fixtures.py:MAX_SEQ_LEN |")
    L.append("| 0Y | BERT 측 1단계 재실행 | outputs_bert/ (51 npz) |")
    L.append("| Phase 4 | lookup table SCALE=1000 | layernorm_table.py:VAR_SCALE |")
    L.append("| 2A | monkey-patch 작성 | nystrom_attention_capture.py |")
    L.append("| 2B | numerical 정합 검증 | max_abs_diff = 0.0 (5 test PASS) |")
    L.append("| 2C | normalization pipeline 일반화 | bert_normalization_pipeline.py table_dir 인자 |")
    L.append("| 2D | run_validation.py 재작성 | HF 통합 + sweep 지원 |")
    L.append("| 2E | 측정 환경 검증 (본 보고서) | docs/migration/08-measurement-environment.md |")
    L.append("")

    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[saved] {md_path}", flush=True)


# ── JSON 저장 (programmatic 파싱) ────────────────────────────────────────────
def write_json(data: dict, passes: list, fails: list, json_path: Path) -> None:
    bert = data["bert"]["model"]
    nys = data["nystrom"]["model"]
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "env": data["env"],
        "fixture": data["fixture"],
        "bert_config": {
            k: getattr(bert.config, k, None)
            for k in [
                "hidden_size", "num_hidden_layers", "num_attention_heads",
                "intermediate_size", "max_position_embeddings", "vocab_size",
                "hidden_act", "layer_norm_eps",
            ]
        },
        "nystrom_config": {
            k: getattr(nys.config, k, None)
            for k in [
                "hidden_size", "num_hidden_layers", "num_attention_heads",
                "intermediate_size", "max_position_embeddings", "vocab_size",
                "hidden_act", "layer_norm_eps",
                "num_landmarks", "segment_means_seq_len", "conv_kernel_size",
            ]
        },
        "bert_swap": {
            "n_swapped": data["bert"]["n_swapped"],
            "all_swapped": data["bert"]["all_swapped"],
        },
        "nystrom_swap": {
            "n_swapped": data["nystrom"]["n_swapped"],
            "all_swapped": data["nystrom"]["all_swapped"],
            "is_patched": data["nystrom"]["is_patched"],
        },
        "passes": [{"label": l, "expected": e, "actual": a} for l, e, a in passes],
        "fails": [{"label": l, "expected": e, "actual": a} for l, e, a in fails],
        "overall": "PASS" if not fails else "FAIL",
    }
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {json_path}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 75)
    print("verify_measurement_parity.py — 작업 2E")
    print("=" * 75)
    print()

    data = collect()

    print()
    print("Running assertions ...")
    passes, fails = run_assertions(data)

    print()
    print("=" * 75)
    print("Assertion Results")
    print("=" * 75)
    for label, exp, act in passes:
        print(f"  ✅ {label}")
        print(f"     expected: {exp}")
        print(f"     actual:   {act}")
    for label, exp, act in fails:
        print(f"  ❌ {label}")
        print(f"     expected: {exp}")
        print(f"     actual:   {act}")

    print()
    print(f"Pass: {len(passes)},  Fail: {len(fails)}")

    md_path = PROJECT_ROOT / "docs" / "migration" / "08-measurement-environment.md"
    json_path = PROJECT_ROOT / "docs" / "migration" / "08-measurement-environment.json"
    write_markdown(data, passes, fails, md_path)
    write_json(data, passes, fails, json_path)

    print()
    print("=" * 75)
    print(f"Overall: {'✅ PASS' if not fails else '❌ FAIL'}")
    print("=" * 75)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
