import sys
import numpy as np

def inspect(npz_path):
    npz = np.load(npz_path)
    print("files in npz:", npz.files)
    # npz에 있는 각 배열에 대한 기본 정보 출력
    for k in npz.files:
        arr = npz[k]
        print(f"key='{k}': dtype={arr.dtype}, shape={arr.shape}, sample=", end="")
        if arr.size == 0:
            print("<empty>")
        else:
            # 맨처음 다섯개의 값만 출력하기
            flat = arr.flatten()
            print(flat[:5])

    # 테이블에 대한 정보를 출력
    if 'tables' in npz.files:
        tables = npz['tables']
        print("tables shape:", tables.shape)  # expecting (K, 2^limb_bits)
        for k in range(tables.shape[0]):
            print(f"segment {k} first 5:", tables[k][:5])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_inspect_npz.py <path/to/softmax_tables_quantized.npz>")
        sys.exit(1)
    inspect(sys.argv[1])