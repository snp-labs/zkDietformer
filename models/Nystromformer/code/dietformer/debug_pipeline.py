# debug_integrated_checks.py
import numpy as np
import sys
import math

if len(sys.argv) < 2:
    print("usage: python debug_integrated_checks.py path/to/integrated_inputs.npz")
    sys.exit(1)
p = sys.argv[1]
d = np.load(p, allow_pickle=True)

print("keys:", d.files)
# basic meta
meta = d['meta']
print("meta:", meta)

# 1) check reconstruct targets via c/div/rem (sanity)
a = d['a']; b = d['b']; c = d['c']; div = d['div']; rem = d['rem']
print("shapes: a", a.shape, "b", b.shape, "c", c.shape, "div", div.shape, "rem", rem.shape)
# reconstruct sample target[0,0,0,0]
def recon(c,div,rem):
    return (((div.astype(object) << 8) + c.astype(object)) << 22) + rem.astype(object)
targets_recon = recon(c,div,rem)
print("sample target[0,0,0,0] =", int(targets_recon[0,0,0,0]))

# 2) load softmax tables & limb indices, recompute exp_prod_int
tables = d['softmax_tables']         # shape (K, table_width)
k_segments = int(d['softmax_k_segments'])
limb_bits = int(d['softmax_limb_bits'])
print("softmax tables shape", tables.shape, "k_segments", k_segments, "limb_bits", limb_bits)
limb_indices = d['limb_indices']    # (batch,num_head,seq_len,seq_len,K)
print("limb_indices shape:", limb_indices.shape)

batch, num_head, seq_len, _, K = limb_indices.shape
table_width = tables.shape[1]

# compute exp_prod_int from tables and limb_indices
exp_prod_int = np.empty((batch,num_head,seq_len,seq_len), dtype=object)
for n in range(batch):
    for h in range(num_head):
        for i in range(seq_len):
            for j in range(seq_len):
                prod = 1
                for k in range(K):
                    idx = int(limb_indices[n,h,i,j,k])
                    if idx < 0 or idx >= table_width:
                        print("Index OOB at", (n,h,i,j,k), "idx", idx, "table_width", table_width)
                    prod *= int(tables[k, idx])
                exp_prod_int[n,h,i,j] = prod

print("exp_prod_int[0,0,0,0] =", exp_prod_int[0,0,0,0])

# 3) check second-matmul consistency: V_q * exp_prod -> X_q
V_q = d['V_q']   # int64
X_q = d['X_q']   # int64
print("V_q shape", V_q.shape, "X_q shape", X_q.shape)
mismatch = []
for n in range(batch):
    for h in range(num_head):
        for i in range(seq_len):
            for t in range(V_q.shape[3]):
                acc = 0
                for j in range(seq_len):
                    acc += int(exp_prod_int[n,h,i,j]) * int(V_q[n,h,j,t])
                expected = acc
                actual = int(X_q[n,h,i,t])
                if expected != actual:
                    mismatch.append(((n,h,i,t), expected, actual))
                    if len(mismatch) <= 5:
                        print("mismatch at", (n,h,i,t), "expected", expected, "actual", actual)
if len(mismatch) == 0:
    print("Second-matmul integer equality OK: sum_j exp_prod * V_q == X_q for all entries")
else:
    print("mismatch count:", len(mismatch))
    print("Example mismatches (up to 5) above")