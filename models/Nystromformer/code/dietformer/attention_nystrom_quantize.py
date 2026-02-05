import argparse
import numpy as np
import os
import math
import torch
from attention_nystrom import NystromAttention

def random_inputs_torch(batch, num_head, seq_len, head_dim, seed=42, device='cpu', dtype=torch.float32):
    torch.manual_seed(seed)
    Q = torch.randn(batch, num_head, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, num_head, seq_len, head_dim, device=device, dtype=dtype)
    mask = torch.ones(batch, seq_len, device=device, dtype=dtype)
    return Q, K, mask

def quantize_with_auto_zero(a_float, b_float, scale_input=1<<8, input_0_default=0, check_uint64=True):
    batch, num_head, rows_a, head_dim = a_float.shape
    rows_b = b_float.shape[2]

    a_raw = np.round(a_float * float(scale_input)).astype(np.int64) # 스케일링 후 반올림하여 값을 정수화시키기
    b_raw = np.round(b_float * float(scale_input)).astype(np.int64)
    # 두 행렬에서 최소값을 찾기(올바른 zero point 계산을 위함)
    min_val = min(int(a_raw.min()), int(b_raw.min()))
    # 행렬의 모든 값 + input_0 >= 0 이 되도록 zero point 결정
    input_0 = max(int(input_0_default), int(-min_val if min_val < 0 else 0))
    # 제로 포인트를 더해서 값의 중심축 잡아주기
    a_q = (a_raw + input_0).astype(np.int64)
    b_q = (b_raw + input_0).astype(np.int64)
    b_q = b_q.transpose(0,1,3,2).copy()  # to (batch,num_head,head_dim,seq_len)

    # optional safety checks before casting to uint64 / saving : 새로 추가한거
    if check_uint64:
        max_a = int(a_q.max())
        max_b = int(b_q.max())
        max_val = max(max_a, max_b)
        if max_val >= 2**64:
            raise ValueError(f"Quantized values too large for uint64: max={max_val}. Reduce scale or choose larger input_0 handling.")
        if input_0 < 0:
            raise ValueError(f"Computed input_0 negative: {input_0}")
        
    return a_q, b_q, input_0, a_raw, b_raw

def decompose_targets(a_q, b_q, a_raw, b_raw_t, input_0, c0=0, multiplier=1):
    # a_raw의 행과 b_raw_t의 열 크기를 각각 가져옴
    batch, num_head, seq_len, head_dim = a_raw.shape
    cols = b_raw_t.shape[3]
    mask22 = (1 << 22) - 1

    # 양자화를 하지 않은 원래값으로 행렬곱을 수행함
    # 결과 행렬의 크기에 맞게 s_vals 초기화 (리스트 대신 numpy object array 권장)
    s_vals = np.zeros((batch, num_head, seq_len, cols), dtype=object)
    min_s = None
    for n in range(batch):
        for h in range(num_head):
            A_raw = a_raw[n,h]  
            B_raw = b_raw_t[n,h]  
            for i in range(seq_len):
                for j in range(cols):
                    s = 0
                    for k in range(head_dim):
                        aval = int(A_raw[i,k])
                        bval = int(B_raw[k,j])
                        s += aval * bval
                    s_vals[n][h][i][j] = s
                    if min_s is None or s < min_s:
                        min_s = s

    # 결과값 행렬에서도 최소값을 찾기(올바른 zero point 계산을 위함)
    c0_needed = 0
    if min_s is not None and min_s < 0:
        c0_needed = -min_s
    c0_used = max(int(c0), int(c0_needed))

    c = np.zeros((batch, num_head, seq_len, cols), dtype=np.uint64)
    div = np.zeros_like(c)
    rem = np.zeros_like(c)
    for n in range(batch):
        for h in range(num_head):
            for i in range(seq_len):
                for j in range(cols):
                    s = int(s_vals[n][h][i][j])
                    target = (s + int(c0_used)) * int(multiplier) # 이때의 target값은 모두 양수여야 함
                    if target < 0:
                        target = 0
                    div_22 = target >> 22
                    rem_22 = target & mask22 # target의 하위 22비트
                    div_8 = div_22 >> 8 # div의 상위 8비트 
                    c_val = div_22 & 0xff # div의 하위 8비트 
                    c[n,h,i,j] = np.uint64(c_val)
                    div[n,h,i,j] = np.uint64(div_8)
                    rem[n,h,i,j] = np.uint64(rem_22)
    return c, div, rem, c0_used



# 모든 파일을 npz 파일로 패킹하여 저장
def export_npz(path, a,b,c,div,rem,r_int,input_0,c0,multiplier,frac_bits,meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    np.savez_compressed(
        path,
        a=a.astype(np.uint64),
        b=b.astype(np.uint64),
        c=c.astype(np.uint64),
        div=div.astype(np.uint64),
        rem=rem.astype(np.uint64),
        r=np.uint64(r_int),
        input_0=np.uint64(input_0),
        c_0_converted=np.uint64(c0),
        multiplier=np.uint64(multiplier),
        frac_bits=np.int32(frac_bits),
        meta=np.array([meta]) if meta is not None else np.array([0])
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_npz", type=str)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=(1<<8))
    parser.add_argument("--input_0", type=int, default=0, help="minimum zero-point hint (will be increased if needed)")
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--frac_bits", type=int, default=32, help="fractional bits used for exp fixed-point in lookup tables")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # 나이스트롬의 입력값 방식에 맞춤
    config = {"head_dim": args.head_dim, "num_head": args.num_head, "num_landmarks": args.seq_len, "seq_len": args.seq_len}
    attn = NystromAttention(config)

    Q_t, K_t, mask = random_inputs_torch(args.batch, args.num_head, args.seq_len, args.head_dim, seed=args.seed, device=args.device)
    scale_div = math.sqrt(math.sqrt(args.head_dim))
    mask_expand = mask.view(args.batch, 1, args.seq_len, 1)
    Q_proc = (Q_t * mask_expand) / scale_div
    K_proc = (K_t * mask_expand) / scale_div

    Q_np = Q_proc.cpu().numpy()
    K_np = K_proc.cpu().numpy()

    a_q, b_q, input_0_fin, a_raw, b_raw = quantize_with_auto_zero(Q_np, K_np, scale_input=args.scale, input_0_default=args.input_0, check_uint64=False)
    
    b_raw_t = b_raw.transpose(0,1,3,2).copy()

    # 양자화된 값이 u64 로 안전하게 들어가는지를 미리 점검하여 오버플로우 우려가 있으면 경고함.
    max_val = max(int(a_q.max()), int(b_q.max()))
    if max_val >= (1 << 63):
        print("Warning: quantized values are large; consider lowering scale to avoid overflow when casting to u64.")
    
    c, div, rem, c0_used = decompose_targets(a_q, b_q, a_raw, b_raw_t, input_0_fin, c0=0, multiplier=1)

    meta = {"batch": args.batch, "num_head": args.num_head, "seq_len": args.seq_len, "head_dim": args.head_dim, "scale": args.scale, "frac_bits": args.frac_bits}
    export_npz(args.out_npz, a_q, b_q, c, div, rem, args.r, input_0_fin, c0_used, 1, args.frac_bits, meta)
    print("Wrote", args.out_npz, "input_0_used=", input_0_fin, "c0_used=", c0_used)