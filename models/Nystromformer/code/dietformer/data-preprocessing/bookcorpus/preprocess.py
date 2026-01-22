import json
import re
import pickle
from multiprocessing import Pool
import os
from transformers import RobertaTokenizerFast

def tokenize(args):
    src, tgt = args
    if not os.path.exists(src): return

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    with open(src, "r", encoding="utf-8") as read_f:
        text = read_f.read()

    # 1. 전체 텍스트 토큰화
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 2. 자르지 않고 전체 ID 리스트를 저장 (전처리기 2단계에서 처리하도록)
    with open(tgt, "wb") as dump_f:
        pickle.dump(token_ids, dump_f)
    print(f"Dump {tgt} (Total Tokens: {len(token_ids)})", flush=True)

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.mkdir("data")

    max_num_tokens = 20000000
    docs = []
    num_tokens = 0
    total_count = 0
    file_idx = 0

    for file in ["books_small_p1.txt", "books_small_p2.txt"]:

        with open(file, "r", encoding = "utf-8") as read_f:
            lines = read_f.read().split("\n")

            for line in lines:

                text = line
                approx_len = len(text.split(" "))
                docs.append(text)
                num_tokens += approx_len

                if num_tokens > max_num_tokens:
                    with open(f"data/bookcorpus-partition-{file_idx:04}.txt", "w", encoding = "utf-8") as dump_f:
                        for text in docs:
                            dump_f.write(text)
                            dump_f.write("\n")
                    print(f"dumped data/bookcorpus-partition-{file_idx:04}.txt")
                    file_idx += 1
                    docs = []
                    num_tokens = 0

                total_count += 1
                if total_count % 10000 == 0:
                    print(f"{total_count}, {len(docs)}, {num_tokens}, {num_tokens / max_num_tokens}")

            with open(f"data/bookcorpus-partition-{file_idx:04}.txt", "w", encoding = "utf-8") as dump_f:
                for text in docs:
                    dump_f.write(text)
                    dump_f.write("\n")
            print(f"dumped data/bookcorpus-partition-{file_idx:04}.txt")
            file_idx += 1
            docs = []
            num_tokens = 0
    
    num_workers = 2
    if num_workers > 1:
        pool = Pool(num_workers)
    args = []
    for file_idx in range(10000):
        file = f"data/bookcorpus-partition-{file_idx:04}.txt"
        args.append((file, file.replace(".txt", "-roberta-base.pickle")))

        if len(args) == num_workers:
            if num_workers == 1:
                tokenize(args[0])
            else:
                pool.map(tokenize, args)
            args = []
    if num_workers > 1:
        pool.close()





