import numpy as np
import sys

def inspect_npz(path: str) -> None:
      data = np.load(path, allow_pickle=True)
      print(f"Loaded file: {path}\n")

      for name in sorted(data.files):
          arr = data[name]
          print(f"Tensor: {name}")
          print(f"  Shape : {arr.shape}  dtype={arr.dtype}")

          if arr.ndim == 4:
              batch, head, seq, head_dim = arr.shape
              print(f"  batch    = {batch}")
              print(f"  head     = {head}")
              print(f"  seq      = {seq}")
              print(f"  head_dim = {head_dim}")
          elif arr.ndim == 0:
              print(f"  scalar   = {arr.item()}")
          elif arr.ndim == 1:
              print(f"  length   = {arr.shape[0]}")
          elif arr.ndim == 2:
              rows, cols = arr.shape
              print(f"  rows     = {rows}")
              print(f"  cols     = {cols}")
          elif arr.ndim == 3:
              d0, d1, d2 = arr.shape
              print(f"  dim0     = {d0}")
              print(f"  dim1     = {d1}")
              print(f"  dim2     = {d2}")

          print()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "outputs_bert/outputs_bert-batch0-bert.encoder.layer.0.attention.self.witness.npz"
    inspect_npz(path)