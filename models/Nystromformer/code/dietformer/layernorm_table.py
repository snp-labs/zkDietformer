import numpy as np
import os

TABLE_SIZE = 1 << 16
FRAC_BITS = 8

# Phase 4: var 인덱스 fixed-point scaling.
#
# SCALE 변경 이력:
#   Phase 2 초기: 정수 인덱스 (real_var[i] = i) — var ∈ (0,1) 영역에서 catastrophic.
#   Phase 4 1차 (2026-04-28):  256 — Nystromformer var max 192 까지 커버하는 전제.
#   Phase 4 2차 (2026-04-28): 1000 — 사용자 측정 (BERT [0.13, 0.55], Nystrom
#                              [0.05, 65.2]) 기반 재선정. 두 모델 영역 100% 커버
#                              + 정수 인덱스 0% 오차. cover = 65536/1000 = 65.54
#                              ≈ Nystrom max 65.2 와 정확히 일치.
#                              (상세: docs/migration/14-scale-factor-analysis.md)
VAR_SCALE = 1000

# 모델별 LayerNorm eps (HF config의 layer_norm_eps)
EPS_BERT = 1e-12
EPS_NYSTROMFORMER = 1e-5
EPS_DEFAULT = EPS_BERT


def generate_layernorm_tables(out_dir, k_bits=26, frac_bits=8, eps=EPS_DEFAULT, var_scale=VAR_SCALE):
    """
    1/√(real_var + eps) 값을 fixed-point uint64로 표 작성.
    real_var[i] = i / var_scale   (i = 0, 1, ..., TABLE_SIZE-1)
    out_dir / 'layernorm_tables.npz' 에 저장. var_scale 도 메타로 함께 저장
    되어 회로 측이 동일한 scaling을 사용하는지 확인 가능.
    """
    os.makedirs(out_dir, exist_ok=True)

    indices = np.arange(TABLE_SIZE, dtype=np.float64)
    real_var = indices / float(var_scale)
    inv_std = 1.0 / np.sqrt(real_var + eps)
    fixed_inv_std = (inv_std * (1 << frac_bits)).astype(np.uint64)

    out_path = os.path.join(out_dir, "layernorm_tables.npz")
    np.savez_compressed(
        out_path,
        table=fixed_inv_std,
        frac_bits=np.int32(frac_bits),
        k_bits=np.int32(k_bits),
        eps=np.float64(eps),
        var_scale=np.int32(var_scale),
    )
    return out_path


def get_layernorm_index_multiplier(s_var, k_bits=26, var_scale=VAR_SCALE):
    """
    normalization_pipeline.py 가 호출. 
    VAR_SCALE 을 곱해서 인덱스가 var_real · VAR_SCALE 형태로 떨어지게 한다. 
    > 양자화된 분산 값을 테이블 인덱스로 변환하기 위한 숫자 M_ln을 구하는 함수

    수학:
        shifted_val ≈ var_real · dim / s_var      (pipeline에서 산출)
        scaled_val  = (shifted_val · M_ln) >> k_bits
                    = (var_real · dim / s_var) · (s_var · VAR_SCALE · 2^k_bits) >> k_bits
                    = var_real · dim · VAR_SCALE
        index       = scaled_val // dim
                    = var_real · VAR_SCALE          ← lookup 인덱스
    """
    return int(np.floor(s_var * var_scale * (1 << k_bits)))


def extract_inv_std_single(Q_var_sim, R_var, dim, ln_table, M_ln, k_bits):
    """
        양자화된 입력으로부터 테이블에 적혀있는 역수 값을 꺼내오는 함수
        index ≈ var_real · VAR_SCALE  (현재 VAR_SCALE = 1000)
    """
    shifted_val = (Q_var_sim.astype(object) * dim) + R_var.astype(object)
    numerator = shifted_val * M_ln
    scaled_val = numerator >> k_bits
    index = (scaled_val // dim).astype(np.uint64)

    index_clipped = np.clip(index, 0, TABLE_SIZE - 1)
    inv_std_total = ln_table[index_clipped.astype(int)]
    return inv_std_total.astype(np.uint64)
