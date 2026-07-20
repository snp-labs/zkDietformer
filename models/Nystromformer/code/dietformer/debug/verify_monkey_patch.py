from __future__ import annotations

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# 모듈 디렉터리 (nystrom_attention_capture.py 등) 를 sys.path 에 추가: debug/ → 1단계 위
HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent              # .../code/dietformer/
PROJECT_ROOT = HERE.parents[4]        # .../Dietformer/
sys.path.insert(0, str(MODULE_DIR))


# ── 검증 설정 ───────────────────────────────────────────────────────────────
SEQ_LEN = 448
THRESHOLD = 1e-5

TEST_CASES = [
    # (label, seed, num_landmarks)
    ("1차 검증 (사전학습 정합 m=64)", 42, 64),
    ("2차 검증 (anti-fluke seed=123)", 123, 64),
    ("3차 sweep — m=32", 42, 32),
    ("3차 sweep — m=64 (재확인)", 42, 64),  # 1차와 같은 입력 재실행 (재현성)
    ("3차 sweep — m=112", 42, 112),
]


# ── Helper: per-test 검증 ──────────────────────────────────────────────────
def run_one_test(label: str, seed: int, num_landmarks: int) -> dict:
    """단일 (seed, m) 조합으로 ref vs patched 비교."""
    import torch
    from nystrom_attention_capture import (
        load_nystromformer_with_landmarks_override,
        apply_monkey_patch,
    )

    print(f"\n{'=' * 75}")
    print(f"  {label}")
    print(f"  seed={seed}, num_landmarks={num_landmarks}, seq_len={SEQ_LEN}")
    print(f"{'=' * 75}")

    t0 = time.time()

    # Reference: HF original forward (config override 만 적용)
    print("  [load] reference model (HF original forward) ...", flush=True)
    ref = load_nystromformer_with_landmarks_override(
        num_landmarks=num_landmarks, seq_len=SEQ_LEN,
    )
    ref.eval()

    # Patched: 같은 config + apply_monkey_patch
    print("  [load] patched model (apply_monkey_patch) ...", flush=True)
    pat = load_nystromformer_with_landmarks_override(
        num_landmarks=num_landmarks, seq_len=SEQ_LEN,
    )
    registry = apply_monkey_patch(pat)
    pat.eval()

    # 같은 결정적 입력
    torch.manual_seed(seed)
    vocab_size = ref.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

    # 양쪽 forward — output_hidden_states + output_attentions 로 layer 별 추적
    print("  [forward] reference ...", flush=True)
    with torch.no_grad():
        out_ref = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    print("  [forward] patched ...", flush=True)
    with torch.no_grad():
        out_pat = pat(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    # ── 핵심 비교: logits ────────────────────────────────────────────────
    logits_diff = (out_ref.logits - out_pat.logits).abs().max().item()

    # ── 보조 비교: per-layer hidden state ─────────────────────────────────
    layer_h_diffs = []
    if hasattr(out_ref, "hidden_states") and out_ref.hidden_states is not None:
        for i, (h_ref, h_pat) in enumerate(
            zip(out_ref.hidden_states, out_pat.hidden_states)
        ):
            d = (h_ref - h_pat).abs().max().item()
            layer_h_diffs.append(d)

    # ── 보조 비교: per-layer attention_probs (HF: kernel_1 @ inv_kernel_2 = mid) ─
    layer_a_diffs = []
    if hasattr(out_ref, "attentions") and out_ref.attentions is not None:
        for i, (a_ref, a_pat) in enumerate(zip(out_ref.attentions, out_pat.attentions)):
            if a_ref is None or a_pat is None:
                layer_a_diffs.append(None)
                continue
            d = (a_ref - a_pat).abs().max().item()
            layer_a_diffs.append(d)

    # ── Capture registry 검증 (patched 모델 측 캡처가 잘 됐는지) ────────────
    captured_layers = list(registry.all().keys())
    captured_keys_per_layer = (
        sorted(registry.get(captured_layers[0]).keys()) if captured_layers else []
    )

    passed = logits_diff < THRESHOLD
    elapsed = time.time() - t0

    # 콘솔 출력
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  [result] logits max_abs_diff = {logits_diff:.3e}  "
          f"(threshold {THRESHOLD:.0e})  {status}")

    if layer_h_diffs:
        max_h = max(layer_h_diffs)
        max_h_layer = layer_h_diffs.index(max_h)
        print(f"  [result] per-layer hidden state max_abs_diff: "
              f"{max_h:.3e}  (at layer {max_h_layer}/{len(layer_h_diffs) - 1})")

    if layer_a_diffs and any(a is not None for a in layer_a_diffs):
        valid = [(i, a) for i, a in enumerate(layer_a_diffs) if a is not None]
        if valid:
            max_a_layer, max_a = max(valid, key=lambda x: x[1])
            print(f"  [result] per-layer attention_probs max_abs_diff: "
                  f"{max_a:.3e}  (at layer {max_a_layer}/{len(valid) - 1})")

    print(f"  [capture] {len(captured_layers)} layers captured "
          f"with {len(captured_keys_per_layer)} keys each")
    print(f"  [time] {elapsed:.1f}s")

    return {
        "label": label,
        "seed": seed,
        "num_landmarks": num_landmarks,
        "seq_len": SEQ_LEN,
        "threshold": THRESHOLD,
        "logits_max_abs_diff": logits_diff,
        "per_layer_hidden_state_max_abs_diff": layer_h_diffs,
        "per_layer_attention_max_abs_diff": layer_a_diffs,
        "captured_layers": captured_layers,
        "captured_keys_per_layer": captured_keys_per_layer,
        "elapsed_seconds": round(elapsed, 1),
        "pass": passed,
    }


# ── Markdown 자동 생성 ──────────────────────────────────────────────────────
def write_markdown_report(results: list[dict], md_path: Path) -> None:
    """검증 결과를 docs/migration/18-monkey-patch-validation.md 로 저장."""
    all_pass = all(r["pass"] for r in results)
    overall = "✅ **PASS — 작업 2C 진입 가능**" if all_pass else "❌ **FAIL — drift 분석 + prover-recompute 협의 필요**"

    lines: list[str] = []
    lines.append("# 작업 2B — Monkey-Patch Numerical 검증 결과")
    lines.append("")
    lines.append(f"> 생성: `verify_monkey_patch.py` 자동, {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"> 환경: dietformer (transformers 4.57.3, torch 2.9.1)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 0. 종합 판정
    lines.append("## 0. 종합 판정")
    lines.append("")
    lines.append(overall)
    lines.append("")

    # 1. 검증 방법
    lines.append("## 1. 검증 방법")
    lines.append("")
    lines.append(f"- Reference: `load_nystromformer_with_landmarks_override(num_landmarks, seq_len={SEQ_LEN})` + HF original forward")
    lines.append(f"- Patched: 같은 config + `apply_monkey_patch(model)`")
    lines.append(f"- 입력: `torch.randint(0, vocab_size, (1, {SEQ_LEN}))`, `attention_mask` 전체 1, `torch.manual_seed(seed)` 고정")
    lines.append(f"- 비교 1차: `(out_ref.logits - out_pat.logits).abs().max() < {THRESHOLD:.0e}`")
    lines.append(f"- 비교 2차: per-layer hidden_states 와 attention_probs 의 max_abs_diff (drift 추적)")
    lines.append("")

    # 2. 결과 표
    lines.append("## 2. 결과 — 5 test cases")
    lines.append("")
    lines.append("| # | label | seed | m | logits Δ | per-layer hidden Δ | per-layer attn Δ | PASS |")
    lines.append("|---|-------|------|---|----------|--------------------|--------------------|------|")
    for i, r in enumerate(results):
        h_max = max(r["per_layer_hidden_state_max_abs_diff"]) if r["per_layer_hidden_state_max_abs_diff"] else None
        a_valid = [a for a in r["per_layer_attention_max_abs_diff"] if a is not None]
        a_max = max(a_valid) if a_valid else None
        h_str = f"{h_max:.3e}" if h_max is not None else "-"
        a_str = f"{a_max:.3e}" if a_max is not None else "-"
        status = "✅" if r["pass"] else "❌"
        lines.append(
            f"| {i+1} | {r['label']} | {r['seed']} | {r['num_landmarks']} | "
            f"{r['logits_max_abs_diff']:.3e} | {h_str} | {a_str} | {status} |"
        )
    lines.append("")

    # 3. Per-layer drift breakdown
    lines.append("## 3. Per-layer drift breakdown (실패 case 진단용)")
    lines.append("")
    for i, r in enumerate(results):
        lines.append(f"### Test {i+1}: {r['label']} (seed={r['seed']}, m={r['num_landmarks']})")
        lines.append("")
        lines.append(f"- logits max_abs_diff: **{r['logits_max_abs_diff']:.3e}**  → {'✅ PASS' if r['pass'] else '❌ FAIL'}")
        lines.append(f"- elapsed: {r['elapsed_seconds']}s")
        lines.append(f"- captured: {len(r['captured_layers'])} layers × {len(r['captured_keys_per_layer'])} keys")
        if r["captured_keys_per_layer"]:
            lines.append(f"- captured npz keys (sorted): `{', '.join(r['captured_keys_per_layer'])}`")
        lines.append("")
        if r["per_layer_hidden_state_max_abs_diff"]:
            lines.append("**Hidden state diff per layer (embedding=0, encoder layers=1..N):**")
            lines.append("")
            lines.append("| layer | max_abs_diff |")
            lines.append("|-------|--------------|")
            for j, d in enumerate(r["per_layer_hidden_state_max_abs_diff"]):
                marker = " 🔥" if d >= THRESHOLD else ""
                lines.append(f"| {j} | {d:.3e}{marker} |")
            lines.append("")
        if r["per_layer_attention_max_abs_diff"] and any(a is not None for a in r["per_layer_attention_max_abs_diff"]):
            lines.append("**Attention probs diff per encoder layer:**")
            lines.append("")
            lines.append("| layer | max_abs_diff |")
            lines.append("|-------|--------------|")
            for j, d in enumerate(r["per_layer_attention_max_abs_diff"]):
                if d is None:
                    lines.append(f"| {j} | (None) |")
                else:
                    marker = " 🔥" if d >= THRESHOLD else ""
                    lines.append(f"| {j} | {d:.3e}{marker} |")
            lines.append("")

    # 4. 통과 분석
    lines.append("## 4. 통과 기준 분석")
    lines.append("")
    lines.append(f"- 통과 기준: 모든 test 의 logits max_abs_diff < {THRESHOLD:.0e}")
    lines.append(f"- 본 monkey-patch 가 HF original 의 forward 를 수학적으로 동일하게 재현하므로, **이상적으로 max_abs_diff = 0**.")
    lines.append(f"- 작은 drift (≤ 1e-7 정도) 는 PyTorch 의 reduction 순서 차이 (예: `+=` vs `+`) 에서 발생 가능.")
    lines.append(f"- {THRESHOLD:.0e} 임계는 그 floating-point noise 에 충분한 여유.")
    lines.append("")

    # 5. 후속 조치
    lines.append("## 5. 후속 조치")
    lines.append("")
    if all_pass:
        lines.append("**✅ 모든 검증 PASS** → 작업 2C (`normalization_pipeline.py` table_dir 인자 추가) 진입.")
        lines.append("")
        lines.append("후속 단계 권장 순서:")
        lines.append("1. 작업 2C — pipeline 함수 일반화")
        lines.append("2. 작업 2D — `run_validation.py` 대규모 재작성")
        lines.append("3. 작업 2E — `verify_measurement_parity.py` + 측정 환경 보고서")
        lines.append("4. Phase 5 — Rust 회로 가젯 (LayerNorm + VAR_SCALE=1000)")
    else:
        lines.append("**❌ 하나 이상의 검증 FAIL** → 다음 분석 후 사용자 협의:")
        lines.append("")
        lines.append("1. **Drift layer 추적**: 위 §3 의 per-layer breakdown 에서 🔥 표시된 layer 가 drift 시작점")
        lines.append("2. **가능한 원인**:")
        lines.append("   - patched_forward 의 inline → 변수 분리 시 산술 순서 미세 차이")
        lines.append("   - `+=` (in-place) vs `+` (new tensor) 의 floating-point reduction 차이 *(가능성 매우 낮음)*")
        lines.append("   - HF transformers 4.57.3 의 forward 본문이 우리가 fork 한 것과 다름 *(상세 비교 필요)*")
        lines.append("3. **대안 (코드 수정 안 함)**:")
        lines.append("   - `THRESHOLD` 를 1e-4 등으로 완화 (단 본 단계 1e-5 기준 fail 의미 정확히 보고)")
        lines.append("   - prover-recompute 방식으로 전환 (numpy 재계산, 08-nystromformer-migration-plan.md §3.2)")
        lines.append("4. **작업 2C 진입 보류** — 사용자 결정 후 진행")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] {md_path}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    print(f"verify_monkey_patch.py — 작업 2B")
    print(f"  threshold = {THRESHOLD:.0e}")
    print(f"  test cases = {len(TEST_CASES)}")
    print(f"  seq_len = {SEQ_LEN}")
    print()

    results: list[dict] = []
    for label, seed, m in TEST_CASES:
        try:
            r = run_one_test(label, seed, m)
        except Exception as e:
            import traceback
            print(f"\n  [error] test failed with exception: {e}")
            traceback.print_exc()
            r = {
                "label": label, "seed": seed, "num_landmarks": m,
                "seq_len": SEQ_LEN, "threshold": THRESHOLD,
                "error": str(e),
                "pass": False,
                "logits_max_abs_diff": float("inf"),
                "per_layer_hidden_state_max_abs_diff": [],
                "per_layer_attention_max_abs_diff": [],
                "captured_layers": [],
                "captured_keys_per_layer": [],
                "elapsed_seconds": 0.0,
            }
        results.append(r)

    # ── Summary 출력 ───────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("Summary")
    print(f"{'=' * 75}")
    print(f"{'#':>3}  {'label':<32}  {'seed':>5}  {'m':>4}  {'logits Δ':>11}  {'PASS'}")
    print("-" * 75)
    for i, r in enumerate(results):
        status = "✅" if r["pass"] else "❌"
        diff = r["logits_max_abs_diff"]
        diff_str = f"{diff:.3e}" if diff != float("inf") else "inf"
        print(f"{i+1:>3}  {r['label']:<32}  {r['seed']:>5}  {r['num_landmarks']:>4}  {diff_str:>11}  {status}")

    all_pass = all(r["pass"] for r in results)
    print()
    print(f"Overall: {'✅ PASS' if all_pass else '❌ FAIL'}")

    # ── JSON 저장 (programmatic 파싱용) ───────────────────────────────
    out_dir = PROJECT_ROOT / "docs" / "migration"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "18-monkey-patch-validation.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {json_path}")

    # ── Markdown 보고서 자동 생성 ────────────────────────────────────
    md_path = out_dir / "18-monkey-patch-validation.md"
    write_markdown_report(results, md_path)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
