import numpy as np
import sys

def load(path):
    d = np.load(path, allow_pickle=True)
    a = d['a'].astype(object)   # (batch,num_head,seq_len,head_dim)
    b = d['b'].astype(object)   # (batch,num_head,head_dim,seq_len)
    c = d['c'].astype(object)
    div = d['div'].astype(object)
    rem = d['rem'].astype(object)
    r = int(np.array(d['r']).tolist())
    input_0 = int(np.array(d['input_0']).tolist())
    c_0 = int(np.array(d['c_0_converted']).tolist())
    mult = int(np.array(d['multiplier']).tolist())
    return a, b, c, div, rem, r, input_0, c_0, mult

def compute_terms(a,b,c,div,rem,r,input_0,c0,mult,n=0,h=0,i=0):
    seq_len = a.shape[2]
    head_dim = a.shape[3]
    # r의 제곱승 계산
    r_pows1 = [pow(r, j+1) for j in range(seq_len)]
    r_pows0 = [pow(r, j) for j in range(seq_len)]

    for name, r_pows in [("r^1", r_pows1), ("r^0", r_pows0)]:
        # (a - a0)(b - b0) = abr - a0b - ab0 + a0b0 을 계산하는 과정
        br = []
        for k in range(head_dim):
            s = 0
            for j in range(seq_len):
                s += int(b[n,h,k,j]) * r_pows[j]
            br.append(s)

        term1 = 0
        for k in range(head_dim):
            term1 += int(a[n,h,i,k]) * br[k]
        # term2: b0 * sum_r * row_sum(A)
        sum_r = sum(r_pows)
        row_sum_A = sum(int(x) for x in a[n,h,i,:])
        term2 = input_0 * sum_r * row_sum_A
        # term3_scalar = a0 * sum_br
        sum_br = sum(br)
        term3 = input_0 * sum_br
        # term4 = head_dim * a0 * b0 * sum_r
        term4 = head_dim * input_0 * input_0 * sum_r
        # 최종 좌항 계산하기
        c0_corr = c0 * sum_r
        left = (term1 - term2 - term3 + term4 + c0_corr) * mult 

        # 우항 계산하기
        cc_row = []
        for j in range(seq_len):
            div_22 = (int(div[n,h,i,j]) << 8) + int(c[n,h,i,j])
            target = ((div_22) << 22) + int(rem[n,h,i,j])
            cc_row.append(target)
        right = sum(cc_row[j] * r_pows[j] for j in range(seq_len))

        print("Convention:", name)
        print(" term1 =", term1)
        print(" term2 =", term2)
        print(" term3 =", term3)
        print(" term4 =", term4)
        print(" c0_corr =", c0_corr)
        print(" left =", left)
        print(" cc_row =", cc_row)
        print(" right =", right)
        print(" left - right =", left - right)
        print("--------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python debug_terms.py path/to/inputs.npz")
        sys.exit(1)
    path = sys.argv[1]

    a,b,c,div,rem,r,input_0,c_0,mult = load(path)
    print("loaded: r", r, "input_0", input_0, "c0", c_0, "mult", mult)
    compute_terms(a,b,c,div,rem,r,input_0,c_0,mult,n=0,h=0,i=0)