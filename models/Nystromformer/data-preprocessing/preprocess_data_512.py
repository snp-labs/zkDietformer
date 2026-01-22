import json
import pickle
import os
import random
from multiprocessing import Pool
from transformers import RobertaTokenizerFast

dump_folder = "512-roberta"
if not os.path.exists(dump_folder):
    os.mkdir(dump_folder)

def process(args):
    dataset, percentage = args
    data_folder = os.path.join(dataset, "data")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    max_seq_len = 512
    per_batch_inst = 8192
    block_size = max_seq_len - tokenizer.num_special_tokens_to_add(pair=False)

    files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".pickle")])
    data_buffer = []
    batch = []
    file_idx = 0

    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)
        
        # 데이터가 리스트의 리스트일 경우를 대비해 평탄화(flatten) 처리
        if len(data) > 0 and isinstance(data[0], list):
            for sublist in data: data_buffer.extend(sublist)
        else:
            data_buffer.extend(data)

        # 버퍼에 데이터가 충분할 때만 루프 실행
        processed_until = 0
        for start_idx in range(0, len(data_buffer) - block_size + 1, block_size):
            block = data_buffer[start_idx : start_idx + block_size]
            if random.random() < percentage:
                block = tokenizer.build_inputs_with_special_tokens(block)
                batch.append(block)
            
            processed_until = start_idx + block_size # 어디까지 처리했는지 기록

            if len(batch) >= per_batch_inst:
                dump_path = os.path.join(dump_folder, f"{dataset}-{file_idx:05}.pickle")
                with open(dump_path, "wb") as dump_f:
                    pickle.dump(batch, dump_f)
                batch = []
                file_idx += 1
        
        # 실제로 처리한 만큼만 버퍼에서 제거 (중요!)
        data_buffer = data_buffer[processed_until:]

    # 마지막에 남은 데이터가 있을 때만 저장
    if len(batch) > 0:
        dump_path = os.path.join(dump_folder, f"{dataset}-{file_idx:05}.pickle")
        with open(dump_path, "wb") as dump_f:
            pickle.dump(batch, dump_f)

            
if __name__ == "__main__":
    datasets = [("bookcorpus", 1.0)] # , ("english_wiki", 1.0), ("realnews", 1 / 3), ("stories", 1 / 3)
    pool = Pool(len(datasets))
    pool.map(process, datasets)
    pool.close()


    import json
    files = sorted([file for file in os.listdir(dump_folder) if file.endswith(".pickle")])
    print(json.dumps(files, indent = 4))
    random.seed(1)
    random.shuffle(files)
    split = int(len(files) * 0.1)

    with open(os.path.join(dump_folder, "dev.json"), "w") as f:
        json.dump(files[:split], f, indent = 4)

    with open(os.path.join(dump_folder, "train.json"), "w") as f:
        json.dump(files[split:], f, indent = 4)
