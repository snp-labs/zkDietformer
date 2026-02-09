import numpy as np
import os
import math
import argparse

BASE_SCALE_EXP = 16
K_SEGMENTS = 5
LIMB_BITS = 16
LIMB_SIZE = 1 << LIMB_BITS
UINT64_MAX = (1 << 64) - 1

# 실수 값을 고정소수점 형식으로 변환
def float_to_fixed(x, frac_bits):
    # NaN이면 0으로, # -inf이면 0으로 대체
    x_safe = np.nan_to_num(x, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    scaled = np.round(x_safe * (1 << frac_bits))
    scaled = np.clip(scaled, 0, UINT64_MAX)
    return scaled.astype(np.uint64)

def generate_gelu_tables_quantized (out_dir,
                              scale_input,
                              input_0,
                              multiplier,
                              frac_bits=32,
                              k_segments=K_SEGMENTS,
                              limb_bits=LIMB_BITS):
    os.makedirs(out_dir, exist_ok=True)

    tables = []
    for k in range(k_segments):
        pow_two = 1 << (limb_bits * k) # limb가 전체 정수에서 차지하는 자리값
        limb_vals = np.arange(0, 1 << limb_bits, dtype=np.float64)
        
        # GeLU 수식 반영: -1.702 * (limb_val * pow_two) / (multiplier * scale_input)
        denom = float(multiplier) * float(scale_input)
        contribs = -1.702 * (limb_vals * pow_two) / denom
        if k == 0:
            contribs += (1.702 * float(input_0)) / float(scale_input)

        contribs = np.clip(contribs, None, 700.0) # 오버플로우 방지로 최대값클립해주고
        exps = np.exp(contribs)
        # 고정소수점으로 바꾸기 전에 오버플로우 안나게 최대값으로 한번 더 클립
        max_allowed_exp = float(UINT64_MAX) / float(1 << frac_bits)
        exps = np.minimum(exps, max_allowed_exp)

        fixed = float_to_fixed(exps, frac_bits=frac_bits)
        tables.append(fixed)

        print(f"Generated table segment {k}: pow_two={pow_two}")

    # stack to shape (K, 2^16)
    stacked = np.stack(tables, axis=0)
    out_path = os.path.join(out_dir, "gelu_tables_quantized.npz")
    np.savez_compressed(out_path,
                        tables=stacked,
                        frac_bits=np.int32(frac_bits),
                        scale_input=np.int64(scale_input),
                        multiplier=np.int64(multiplier),
                        base_scale_exp=np.int32(BASE_SCALE_EXP),
                        limb_bits=np.int32(limb_bits),
                        k_segments=np.int32(k_segments))
    print("Wrote quantized tables to", out_path)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--scale_input", type=int, default=1<<8, help="scale used for quantizing Q,K (e.g. 256)")
    parser.add_argument("--input_0", type=int, default=0)
    parser.add_argument("--multiplier", type=int, default=1, help="multiplier used in target = (s + c0) * multiplier")
    parser.add_argument("--frac_bits", type=int, default=32, help="fractional bits for fixed-point encoding of exp()")
    parser.add_argument("--k_segments", type=int, default=5)
    parser.add_argument("--limb_bits", type=int, default=16)
    args = parser.parse_args()

    generate_gelu_tables_quantized(args.out_dir,
                              scale_input=args.scale_input,
                              input_0=args.input_0,
                              multiplier=args.multiplier,
                              frac_bits=args.frac_bits,
                              k_segments=args.k_segments,
                              limb_bits=args.limb_bits)