import json
import pickle
import os
import glob
import time
import traceback
from multiprocessing import Pool
from transformers import RobertaTokenizerFast


def tokenize(args):
    src, tgt = args
    if not os.path.exists(src):
        print(f"SKIP: {src} not found", flush=True)
        return
    if os.path.exists(tgt):
        print(f"SKIP: {tgt} already exists", flush=True)
        return

    try:
        t0 = time.time()
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        print(f"  [{src}] Start", flush=True)

        # 줄 단위로 읽어서 배치 토큰화
        all_token_ids = []
        batch = []
        batch_size = 1000
        line_count = 0

        with open(src, "r", encoding="utf-8") as read_f:
            for line in read_f:
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                line_count += 1

                if len(batch) >= batch_size:
                    encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
                    for ids in encoded:
                        all_token_ids.extend(ids)
                    batch = []

                    if line_count % 50000 == 0:
                        print(f"  [{src}] {line_count:,} lines, "
                              f"{len(all_token_ids):,} tokens ({time.time()-t0:.0f}s)",
                              flush=True)

            # 남은 배치 처리
            if batch:
                encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
                for ids in encoded:
                    all_token_ids.extend(ids)

        print(f"  [{src}] Done tokenizing: {len(all_token_ids):,} tokens "
              f"({time.time()-t0:.0f}s)", flush=True)

        with open(tgt, "wb") as dump_f:
            pickle.dump(all_token_ids, dump_f)
        print(f"  [{src}] Dumped {tgt} ({time.time()-t0:.0f}s)", flush=True)

    except Exception:
        print(f"ERROR processing {src}:", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.mkdir("data")

    # ============================================================
    # 1단계: books_small_p1/p2.txt를 partition txt들로 분할
    # 이미 partition 파일이 충분히 있으면 건너뜁니다.
    # ============================================================
    existing_partitions = sorted(glob.glob("data/bookcorpus-partition-*.txt"))
    if len(existing_partitions) > 0:
        print(f"Found {len(existing_partitions)} existing partition files. "
              f"Skipping partition step.", flush=True)
    else:
        print("No partition files found. Creating partitions...", flush=True)
        max_num_tokens = 20000000
        docs = []
        num_tokens = 0
        total_count = 0
        file_idx = 0

        for file in ["books_small_p1.txt", "books_small_p2.txt"]:
            if not os.path.exists(file):
                print(f"WARNING: {file} not found, skipping", flush=True)
                continue

            with open(file, "r", encoding="utf-8") as read_f:
                lines = read_f.read().split("\n")

                for line in lines:
                    text = line
                    approx_len = len(text.split(" "))
                    docs.append(text)
                    num_tokens += approx_len

                    if num_tokens > max_num_tokens:
                        with open(f"data/bookcorpus-partition-{file_idx:04}.txt", "w", encoding="utf-8") as dump_f:
                            for text in docs:
                                dump_f.write(text)
                                dump_f.write("\n")
                        print(f"dumped data/bookcorpus-partition-{file_idx:04}.txt", flush=True)
                        file_idx += 1
                        docs = []
                        num_tokens = 0

                    total_count += 1
                    if total_count % 10000 == 0:
                        print(f"{total_count}, {len(docs)}, {num_tokens}, "
                              f"{num_tokens / max_num_tokens:.3f}", flush=True)

                # 파일 끝에서 남은 docs 저장 (비어있지 않을 때만)
                if len(docs) > 0:
                    with open(f"data/bookcorpus-partition-{file_idx:04}.txt", "w", encoding="utf-8") as dump_f:
                        for text in docs:
                            dump_f.write(text)
                            dump_f.write("\n")
                    print(f"dumped data/bookcorpus-partition-{file_idx:04}.txt", flush=True)
                    file_idx += 1
                    docs = []
                    num_tokens = 0

        print("Partition step complete.", flush=True)

    # ============================================================
    # 2단계: partition txt들을 토큰화하여 pickle로 저장
    # ============================================================
    print("Pre-caching tokenizer...", flush=True)
    _ = RobertaTokenizerFast.from_pretrained("roberta-base")
    print("Tokenizer cached.", flush=True)

    txt_files = sorted(glob.glob("data/bookcorpus-partition-*.txt"))
    print(f"Found {len(txt_files)} partition files to tokenize", flush=True)

    if len(txt_files) == 0:
        print("ERROR: No partition files to tokenize. Exiting.", flush=True)
        exit(1)

    args_list = [(f, f.replace(".txt", "-roberta-base.pickle")) for f in txt_files]

    # 단일 프로세스로 첫 파일 테스트 — 메모리 사용량 확인용
    print("Testing with first file (single process)...", flush=True)
    tokenize(args_list[0])
    print("Test done.", flush=True)

    # 나머지 파일 처리
    # 16GB RAM 환경: num_workers=1 권장. 첫 파일 처리 중 메모리 모니터링 후 여유 있으면 2로.
    num_workers = 1
    if len(args_list) > 1:
        if num_workers > 1:
            with Pool(num_workers) as pool:
                pool.map(tokenize, args_list[1:])
        else:
            for arg in args_list[1:]:
                tokenize(arg)

    print("All done.", flush=True)

