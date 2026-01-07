import numpy as np
import sys

# npz로 패킹한거 풀기
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    a = d['a']   # batch,num_head,seq_len,head_dim, but stored as uint
    b = d['b']   # batch,num_head,head_dim,seq_len
    c = d['c']
    div = d['div']
    rem = d['rem']
    r = int(np.array(d['r']).tolist())
    input_0 = int(np.array(d['input_0']).tolist())
    c_0 = int(np.array(d['c_0_converted']).tolist())
    mult = int(np.array(d['multiplier']).tolist())
    meta = None
    if 'meta' in d:
        try:
            meta = d['meta'].tolist()
        except Exception:
            meta = d['meta']

    return a, b, c, div, rem, r, input_0, c_0, mult, meta

def reconstruct_target_from_c_div_rem(c_arr, div_arr, rem_arr):
    # 넘파이의 제한된 자료형을 파이썬 객체로 바꿔줌 
    c_o = c_arr.astype(object)
    div_o = div_arr.astype(object)
    rem_o = rem_arr.astype(object)
    # 양자화된 타겟값 다시 계산하기 : ((div << 8) + c) << 22 + rem
    return (((div_o << 8) + c_o) << 22) + rem_o

def compute_raw_target(a_q, b_q, input_0, c_0, mult):
    batch, num_head, seq_len, head_dim = a_q.shape
    raw = np.empty((batch, num_head, seq_len, seq_len), dtype=object)
    for n in range(batch):
        for h in range(num_head):
            A = a_q[n,h].astype(object)
            B = b_q[n,h].astype(object)
            for i in range(seq_len):
                for j in range(seq_len):
                    s = 0
                    for k in range(head_dim):
                        aval = int(A[i,k]) - int(input_0)
                        bval = int(B[k,j]) - int(input_0)
                        s += aval * bval
                    target = (s + int(c_0)) * int(mult)
                    if target < 0:
                        target = 0
                    raw[n,h,i,j] = int(target)
    return raw

def get_min_max(obj_array):
    flat = [int(x) for x in obj_array.ravel()]
    return min(flat), max(flat)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python verify_npz.py path/to/inputs.npz")
        sys.exit(1)
    path = sys.argv[1]
    a,b,c,div,rem,r,input_0,c_0,mult,meta = load_npz(path)

    print("PYTHON DEBUG"); # 기본정보 뽑아서 출력해보기
    print("info: shapes:","a", a.shape, "b", b.shape, "c", c.shape, "div", div.shape, "rem", rem.shape)
    print("a[0,0,0,0] =", int(a[0,0,0,0]))
    print("b[0,0,0,0] =", int(b[0,0,0,0]))
    print("c[0,0,0,0] =", int(c[0,0,0,0]))
    print("div[0,0,0,0] =", int(div[0,0,0,0]))
    print("rem[0,0,0,0] =", int(rem[0,0,0,0]))
    print("meta:", meta)

    # 양자화된 타겟값 계산하기(npz에서 가져온 값)
    target_recon = reconstruct_target_from_c_div_rem(c, div, rem)
    # 양자화 하지 않은 a,b로 양자화된 타겟값 계산하기
    target_raw = compute_raw_target(a.astype(object), b.astype(object), input_0, c_0, mult)

    # 두 값을 비교해보기
    mismatches = []
    batch, num_head, seq_len, _ = target_recon.shape
    for n in range(batch):
        for h in range(num_head):
            for i in range(seq_len):
                for j in range(seq_len):
                    trec = int(target_recon[n,h,i,j])
                    tdir = int(target_raw[n,h,i,j])
                    if trec != tdir:
                        mismatches.append(((n,h,i,j), trec, tdir))

    print("mismatch count:", len(mismatches))
    if mismatches:
        print("example mismatches (up to 10):")
        for e in mismatches[:10]:
            print(e)

    # 각 배열의 최소/최대값 출력해보기
    min_a, max_a = get_min_max(a.astype(object))
    min_b, max_b = get_min_max(b.astype(object))
    min_c, max_c = get_min_max(c.astype(object))
    min_div, max_div = get_min_max(div.astype(object))
    min_rem, max_rem = get_min_max(rem.astype(object))

    print(f"a_q: min={min_a}, max={max_a}")
    print(f"b_q: min={min_b}, max={max_b}")
    print(f"c: min={min_c}, max={max_c}")
    print(f"div: min={min_div}, max={max_div}")
    print(f"rem: min={min_rem}, max={max_rem}")
    print("r:", r, "input_0:", input_0, "c0:", c_0, "mult:", mult)

    # 타겟값도 출력해보기
    flat_targets = [int(target_raw[n,h,i,j]) for n in range(batch) for h in range(num_head) for i in range(seq_len) for j in range(seq_len)]
    print("max direct target:", max(flat_targets))
    print("min direct target:", min(flat_targets))

    if len(mismatches) == 0:
        print("All targets match reconstructed c/div/rem. .npz looks consistent.")
    else:
        print("There are mismatches; inspect quantization/decomposition code and parameters (scale/input_0).")