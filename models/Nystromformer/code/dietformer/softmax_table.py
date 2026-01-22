import numpy as np
import os
import math
import argparse

# 행렬곱 결과가 양자화된 정수일 때, 그 정수를 limb로 분해했을 때 각 limb가 exp에 기여하는 값을 반영한 lookup table을 만듬
BASE_SCALE_EXP = 16
K_SEGMENTS = 2
LIMB_BITS = 16
LIMB_SIZE = 1 << LIMB_BITS

def float_to_fixed(x, frac_bits):
    """
    실수 값을 고정소수점 형식으로 안전하게 변환합니다.
    무한대나 너무 큰 값은 uint64의 최대값으로 클램핑합니다.
    """
    # 1. 스케일링 수행
    scaled_float = x * (1 << frac_bits)
    
    # 2. uint64가 표현 가능한 최대값 설정
    max_uint64 = np.iinfo(np.uint64).max
    
    # 3. 무한대(inf) 또는 범위를 벗어난 값을 최대값으로 대체 (Clipping)
    scaled_float = np.clip(scaled_float, 0, max_uint64)
    
    # 4. 반올림 후 형변환
    return np.round(scaled_float).astype(np.uint64)

def generate_tables_quantized(out_dir,
                              scale_input,
                              multiplier,
                              frac_bits=32,
                              k_segments=K_SEGMENTS,
                              limb_bits=LIMB_BITS):
    os.makedirs(out_dir, exist_ok=True)

    # 수치적 안전성을 위해 지수함수의 입력(contribs)이 가질 수 있는 최대값 계산
    # exp(contribs) * 2^frac_bits <= uint64_max를 만족해야 함
    max_allowed_contrib = math.log(np.iinfo(np.uint64).max / (1 << frac_bits))

    tables = []
    for k in range(k_segments):
        # pow_two: 2^(limb_bits * k)
        pow_two = 1 << (limb_bits * k)
        
        # limb가 가질 수 있는 모든 정수 값 준비 [0, ..., (2^limb_bits)-1]
        limb_vals = np.arange(0, 1 << limb_bits, dtype=np.float64)

        # 지수 기여도 계산: contribs = (L * 2^(bits*k)) / (multiplier * scale_input^2)
        denom = float(multiplier) * float(scale_input) * float(scale_input)
        contribs = (limb_vals * float(pow_two)) / denom

        # [Safety Check] 너무 큰 contribs는 exp 계산 시 Overflow를 일으키므로 제한
        if np.any(contribs > max_allowed_contrib):
            num_clipped = np.sum(contribs > max_allowed_contrib)
            # print(f"Warning: {num_clipped} values in segment {k} exceed safe exp range. Clipping applied.")
            contribs = np.clip(contribs, 0, max_allowed_contrib)

        # 지수 함수 적용 및 고정소수점 변환
        exps = np.exp(contribs)
        fixed = float_to_fixed(exps, frac_bits=frac_bits)
        tables.append(fixed)

        print(f"Generated table segment {k}: pow_two={pow_two}, denom={denom}")

    # (K, Table_Width) 형상의 행렬로 결합
    stacked = np.stack(tables, axis=0)
    out_path = os.path.join(out_dir, "softmax_tables_quantized.npz")
    
    np.savez_compressed(out_path,
                        tables=stacked,
                        frac_bits=np.int32(frac_bits),
                        scale_input=np.int64(scale_input),
                        multiplier=np.int64(multiplier),
                        base_scale_exp=np.int32(BASE_SCALE_EXP),
                        limb_bits=np.int32(limb_bits),
                        k_segments=np.int32(k_segments))
    
    print(f"Wrote quantized tables to {out_path} (Shape: {stacked.shape})")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str, help="Output directory for the .npz file")
    parser.add_argument("--scale_input", type=int, default=1<<8, help="Scale for Q, K (default: 256)")
    parser.add_argument("--multiplier", type=int, default=1, help="Multiplier used in quantization")
    parser.add_argument("--frac_bits", type=int, default=32, help="Fractional bits for exp() fixed-point")
    parser.add_argument("--k_segments", type=int, default=2)
    parser.add_argument("--limb_bits", type=int, default=16)
    args = parser.parse_args()

    generate_tables_quantized(args.out_dir,
                              scale_input=args.scale_input,
                              multiplier=args.multiplier,
                              frac_bits=args.frac_bits,
                              k_segments=args.k_segments,
                              limb_bits=args.limb_bits)
    

