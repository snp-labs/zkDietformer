#!/usr/bin/env python3
"""
BERT-base-uncased 와 uw-madison/nystromformer-512 의 LayerNorm 입력 var 분포 측정.

사용법:
    cd <some workdir>  # outputs 가 만들어져도 무관한 위치
    python measure_var_distribution.py                      # 둘 다 측정 + JSON/MD 저장
    python measure_var_distribution.py --model bert-only    # BERT 만
    python measure_var_distribution.py --model nystrom-only # Nystrom 만
    python measure_var_distribution.py --json out.json --md out.md
    python measure_var_distribution.py --max-length 64       # 짧게

요구 환경: torch + transformers + numpy. 첫 실행 시 HF Hub에서 모델 다운로드.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ── 측정용 입력 텍스트 (재현성 위해 코드에 박음) ────────────────────────────
SAMPLE_TEXTS: list[str] = [
    "The capital of France is [MASK].",
    "Machine learning is a subset of [MASK] intelligence.",
    "Photosynthesis converts [MASK] dioxide into oxygen.",
    "The quick brown fox jumps over the lazy [MASK].",
    "She opened the book and began to [MASK] silently.",
    "Quantum computers use qubits instead of classical [MASK].",
    "The ancient ruins were discovered by an [MASK] team.",
    "Climate change is one of the most pressing [MASK] of our time.",
    "He grabbed his umbrella because the forecast predicted [MASK].",
    "Newton's third law states that for every action there is an equal and opposite [MASK].",
]


def _stats_from_array(arr: np.ndarray) -> dict:
    """min / 5% / 25% / median / 75% / 95% / max / mean / std + 영역별 비율."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return {"size": 0}

    pcts = np.percentile(arr, [5, 25, 50, 75, 95])
    out = {
        "size": int(arr.size),
        "min": float(arr.min()),
        "p05": float(pcts[0]),
        "p25": float(pcts[1]),
        "median": float(pcts[2]),
        "p75": float(pcts[3]),
        "p95": float(pcts[4]),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        # 영역별 비율 (보고서 표에 사용)
        "frac_lt_1":      float(np.mean(arr < 1.0)),
        "frac_1_to_10":   float(np.mean((arr >= 1.0) & (arr < 10.0))),
        "frac_10_to_100": float(np.mean((arr >= 10.0) & (arr < 100.0))),
        "frac_ge_100":    float(np.mean(arr >= 100.0)),
    }
    return out


def _measure_model(
    model_id: str,
    tokenizer_cls,
    model_cls,
    encoder_attr: str,           # 'bert' / 'nystromformer'
    texts: list[str],
    max_length: int,
    device: str,
) -> dict:
    """공통 측정 루틴.

    encoder_attr: 모델이 attribute 이름으로 노출하는 백본 ('bert', 'nystromformer').
    """
    import torch

    print(f"[{model_id}] tokenizer 로드 ...", flush=True)
    tokenizer = tokenizer_cls.from_pretrained(model_id)

    print(f"[{model_id}] 모델 로드 ...", flush=True)
    model = model_cls.from_pretrained(model_id)
    model.eval()
    model.to(device)

    # ── attention mask 사용해서 padding 위치 var 는 제외 ──────────────────
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    print(f"[{model_id}] input_ids.shape = {tuple(input_ids.shape)}", flush=True)

    # LayerNorm 모듈 모으기 — Hugging Face BERT/Nystromformer 두 모델 모두
    # encoder.layer[i].attention.output.LayerNorm  (post-attention LN)
    # encoder.layer[i].output.LayerNorm            (post-FFN LN)
    backbone = getattr(model, encoder_attr)
    layers = backbone.encoder.layer

    captured: list[dict] = []  # {layer, position, var_arr}

    handles = []
    for i, layer in enumerate(layers):
        for pos_label, ln_module in [
            ("post_attn", layer.attention.output.LayerNorm),
            ("post_ffn",  layer.output.LayerNorm),
        ]:
            def _make_hook(layer_i, pos):
                def _hook(module, inputs, _output):
                    x = inputs[0].detach()
                    # var = mean((x − μ)²) 의 per-position 값 (unbiased=False)
                    v = x.var(dim=-1, unbiased=False).cpu().numpy()  # [B, S]
                    captured.append({"layer": layer_i, "position": pos, "var": v})
                return _hook
            handles.append(ln_module.register_forward_hook(_make_hook(i, pos_label)))

    print(f"[{model_id}] forward (no_grad) ...", flush=True)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    for h in handles:
        h.remove()

    # ── attention_mask 적용해서 padding 위치 var 제외 ────────────────────
    mask_np = attention_mask.detach().cpu().numpy().astype(bool)  # [B, S]
    valid_var: list[float] = []
    per_layer = {}    # layer_idx → list[float]
    per_pos = {"post_attn": [], "post_ffn": []}

    for entry in captured:
        v = entry["var"]                 # [B, S]
        v_valid = v[mask_np]             # 1D
        valid_var.extend(v_valid.tolist())
        per_layer.setdefault(entry["layer"], []).extend(v_valid.tolist())
        per_pos[entry["position"]].extend(v_valid.tolist())

    arr_all = np.array(valid_var, dtype=np.float64)

    overall = _stats_from_array(arr_all)
    by_layer = {str(k): _stats_from_array(np.array(v)) for k, v in per_layer.items()}
    by_position = {k: _stats_from_array(np.array(v)) for k, v in per_pos.items()}

    # 가장 var 큰/작은 layer
    layer_medians = [(int(k), s["median"]) for k, s in by_layer.items()]
    layer_medians.sort(key=lambda x: x[1])
    smallest = layer_medians[0]
    largest = layer_medians[-1]

    return {
        "model_id": model_id,
        "num_layers": len(layers),
        "num_samples": int(input_ids.shape[0]),
        "max_length": int(input_ids.shape[1]),
        "num_valid_positions": int(mask_np.sum()),
        "num_var_observations": int(arr_all.size),
        "overall": overall,
        "by_layer": by_layer,
        "by_position": by_position,
        "smallest_var_layer": {"layer": smallest[0], "median_var": smallest[1]},
        "largest_var_layer": {"layer": largest[0], "median_var": largest[1]},
    }


def measure_bert(texts, max_length, device):
    from transformers import BertForMaskedLM, BertTokenizerFast
    return _measure_model(
        model_id="bert-base-uncased",
        tokenizer_cls=BertTokenizerFast,
        model_cls=BertForMaskedLM,
        encoder_attr="bert",
        texts=texts,
        max_length=max_length,
        device=device,
    )


def measure_nystromformer(texts, max_length, device):
    from transformers import AutoTokenizer
    # NystromformerForMaskedLM 가 있고 backbone 은 .nystromformer 로 노출됨
    from transformers import NystromformerForMaskedLM
    return _measure_model(
        model_id="uw-madison/nystromformer-512",
        tokenizer_cls=AutoTokenizer,
        model_cls=NystromformerForMaskedLM,
        encoder_attr="nystromformer",
        texts=texts,
        max_length=max_length,
        device=device,
    )


# ── Markdown 표 작성 헬퍼 ────────────────────────────────────────────────
def _stats_row(label: str, s: dict) -> str:
    return (
        f"| {label} | {s.get('size','-')} | "
        f"{s.get('min',0):.4g} | {s.get('p05',0):.4g} | {s.get('p25',0):.4g} | "
        f"{s.get('median',0):.4g} | {s.get('p75',0):.4g} | {s.get('p95',0):.4g} | "
        f"{s.get('max',0):.4g} | {s.get('mean',0):.4g} | {s.get('std',0):.4g} |"
    )


def _frac_row(label: str, s: dict) -> str:
    return (
        f"| {label} | "
        f"{100*s.get('frac_lt_1',0):.2f}% | "
        f"{100*s.get('frac_1_to_10',0):.2f}% | "
        f"{100*s.get('frac_10_to_100',0):.2f}% | "
        f"{100*s.get('frac_ge_100',0):.2f}% |"
    )


def _make_judgment(bert_overall: dict, nystrom_overall: dict) -> tuple[str, str]:
    """A/B/C 판단."""
    def _classify(s):
        if s["p05"] >= 10 and s["frac_lt_1"] < 0.05:
            return "safe"
        if s["median"] < 10 or s["frac_lt_1"] > 0.30:
            return "critical"
        return "warning"

    b = _classify(bert_overall)
    n = _classify(nystrom_overall)

    # 더 보수적인 판단으로 결정
    rank = {"safe": 0, "warning": 1, "critical": 2}
    worst = "safe" if max(rank[b], rank[n]) == 0 else ("warning" if max(rank[b], rank[n]) == 1 else "critical")

    if worst == "safe":
        return ("A — 안전 (Phase 4 후순위)",
                "BERT/Nystromformer 모두 var의 5% percentile ≥ 10 이고 catastrophic 영역 비율이 5% 미만. "
                "현재 lookup table 그대로 §7 결과 신뢰 가능. §7 limitation 한 줄 명시 권장.")
    if worst == "warning":
        return ("B — 주의 (Phase 4 권장)",
                "두 모델 중 하나의 25% percentile이 10 미만이거나 catastrophic 영역 비율이 5–30% 사이. "
                "Phase 4 진행 시 §7 정확도 향상. 마감 일정에 따라 우선순위 결정.")
    return ("C — 필수 (Phase 4 즉시)",
            "두 모델 중 하나의 median이 10 미만이거나 catastrophic 영역 비율이 30% 초과. "
            "현재 lookup으로는 §7 결과가 의미 없음. Phase 4 + 5 즉시 진행 필요. "
            "기존 dim=128 모델로 측정한 결과도 재검증 필요.")


def write_markdown(results: dict, md_path: Path) -> None:
    bert = results.get("bert")
    nys  = results.get("nystrom")

    lines = []
    lines.append("# 작업 3 측정 결과 — 자동 생성")
    lines.append("")
    lines.append(f"> 측정 텍스트 sample 수: {len(SAMPLE_TEXTS)}")
    lines.append(f"> 자동 생성: `measure_var_distribution.py`")
    lines.append("")

    for label, key in [("BERT-base-uncased", "bert"), ("uw-madison/nystromformer-512", "nystrom")]:
        r = results.get(key)
        if r is None:
            lines.append(f"## {label}\n\n*측정되지 않음*\n")
            continue
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- num_layers: {r['num_layers']}")
        lines.append(f"- num_samples: {r['num_samples']}, max_length: {r['max_length']}")
        lines.append(f"- valid token 수: {r['num_valid_positions']}")
        lines.append(f"- var 관측치 총량: {r['num_var_observations']}")
        lines.append(f"- 가장 var 작은 layer: {r['smallest_var_layer']}")
        lines.append(f"- 가장 var 큰 layer: {r['largest_var_layer']}")
        lines.append("")
        lines.append("### Overall var 통계")
        lines.append("")
        lines.append("| group | N | min | p5 | p25 | median | p75 | p95 | max | mean | std |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        lines.append(_stats_row("overall", r["overall"]))
        for pos_label, s in r["by_position"].items():
            lines.append(_stats_row(f"by position: {pos_label}", s))
        lines.append("")
        lines.append("### 영역별 비율")
        lines.append("")
        lines.append("| group | var ∈ [0, 1) (catastrophic) | [1, 10) | [10, 100) | ≥ 100 |")
        lines.append("|---|---|---|---|---|")
        lines.append(_frac_row("overall", r["overall"]))
        for pos_label, s in r["by_position"].items():
            lines.append(_frac_row(f"by position: {pos_label}", s))
        lines.append("")
        lines.append("### Layer별 (median var)")
        lines.append("")
        lines.append("| layer | N | median | p05 | p95 | frac<1 |")
        lines.append("|---|---|---|---|---|---|")
        for layer_idx in sorted(r["by_layer"].keys(), key=int):
            s = r["by_layer"][layer_idx]
            lines.append(
                f"| {layer_idx} | {s['size']} | {s['median']:.4g} | "
                f"{s['p05']:.4g} | {s['p95']:.4g} | {100*s['frac_lt_1']:.2f}% |"
            )
        lines.append("")

    if bert and nys:
        verdict, rationale = _make_judgment(bert["overall"], nys["overall"])
        lines.append("## 작업 4 — 판단")
        lines.append("")
        lines.append(f"**판단: {verdict}**")
        lines.append("")
        lines.append(rationale)
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {md_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["both", "bert-only", "nystrom-only"], default="both")
    parser.add_argument("--max-length", type=int, default=32,
                        help="토큰 max_length. 짧을수록 빠름.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default="var_distribution.json",
                        help="결과 JSON 저장 경로")
    parser.add_argument("--md", default="var_distribution.md",
                        help="결과 markdown 저장 경로")
    args = parser.parse_args()

    print(f"[info] PyTorch device: {args.device}", flush=True)
    print(f"[info] sample 수: {len(SAMPLE_TEXTS)}, max_length: {args.max_length}", flush=True)

    results = {}
    if args.model in ("both", "bert-only"):
        try:
            results["bert"] = measure_bert(SAMPLE_TEXTS, args.max_length, args.device)
            print("[ok] BERT 측정 완료", flush=True)
        except Exception as e:
            print(f"[error] BERT 측정 실패: {e}", file=sys.stderr)
            results["bert"] = None
    if args.model in ("both", "nystrom-only"):
        try:
            results["nystrom"] = measure_nystromformer(SAMPLE_TEXTS, args.max_length, args.device)
            print("[ok] Nystromformer 측정 완료", flush=True)
        except Exception as e:
            print(f"[error] Nystromformer 측정 실패: {e}", file=sys.stderr)
            results["nystrom"] = None

    Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {args.json}", flush=True)
    write_markdown(results, Path(args.md))

    # 콘솔 요약
    for key, label in [("bert", "BERT-base-uncased"), ("nystrom", "uw-madison/nystromformer-512")]:
        r = results.get(key)
        if r is None:
            continue
        s = r["overall"]
        print(f"\n=== {label} ===")
        print(f"  median var: {s['median']:.4g}  (p5={s['p05']:.4g}, p95={s['p95']:.4g})")
        print(f"  catastrophic [0,1) 비율: {100*s['frac_lt_1']:.2f}%")
        print(f"  warning      [1,10) 비율: {100*s['frac_1_to_10']:.2f}%")
        print(f"  good         [10,100) 비율: {100*s['frac_10_to_100']:.2f}%")
        print(f"  safe         ≥100 비율: {100*s['frac_ge_100']:.2f}%")


if __name__ == "__main__":
    main()
