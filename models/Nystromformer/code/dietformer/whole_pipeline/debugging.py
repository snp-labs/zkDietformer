import numpy as np
p = "attn_debug_outputs/attn_debug-model.transformer_0.mha.attn.witness.npz"
d = np.load(p, allow_pickle=True)

print("keys:", list(d.keys()))
need = [
  "a_k1","b_k1","c_k1","div_k1","rem_k1",
  "a_k2","b_k2","c_k2","div_k2","rem_k2",
  "a_k3","b_k3","c_k3","div_k3","rem_k3",
  "input_0_k1","c_0_converted_k1",
  "input_0_k2","c_0_converted_k2",
  "input_0_k3","c_0_converted_k3",
  "softmax_tables","r","r_challenge"
]
for k in need:
    print(k, "in file?", k in d)
    if k in d:
        arr = d[k]
        try:
            print("  dtype:", arr.dtype, "shape:", getattr(arr,'shape',None))
            # object dtype은 문제의 원인 가능성 높음
            if arr.dtype == object:
                print("  WARNING: object dtype - convert to numeric if possible")
        except Exception as e:
            print("  Could not inspect:", e)

# spot-check some arrays (print small slices)
for k in ["a_k1","b_k1","c_k1"]:
    if k in d:
        arr = np.array(d[k])
        print(k, "min/max:", arr.min(), arr.max(), "shape", arr.shape)