# zkDietFormer: Succinct and Subquadratic Proofs for Low-Rank Self-Attention via Polynomial Commitments

## Current experimental branch

- feat/nystrom-bert-huggingface

## Summary

<img width="1920" height="1080" alt="제목을 입력해주세요  (9)" src="https://github.com/user-attachments/assets/f47eddd9-8312-4358-8ca7-6db587ad89b3" />

## How to Start

### 1. Pretrain the Model
Before running the experiments, please create and activate the following Conda environment:

```bash
conda create -n dietformer python=3.11
conda activate dietformer

pip install transformers
pip install torch
pip install pandas
pip install protobuf
```

**[🚀 Updated]** In this branch, experiments can now be performed using the pre-trained Nystromformer and BERT models provided by Hugging Face.

> To pre-train the models from scratch, please use the `feat/circuit-ZEN` and `feat/circuit-ZEN-large-data` branches.

---


### 2. Extract Intermediate Values

Directory Path : Dietformer/models/Nystromformer/code/dietformer

To generate zero-knowledge proofs for the **Attention** and **MLP** components,

we first extract the required intermediate values from the model.

These values are saved as a compressed **`.npz`** file.

Set the arguments at `bert_attention_pipeline.py, nystrom_attention_pipeline.py, evaluation_fixtures.py` according to the model configuration you want to evaluate.

Run the following command for Nystromformer:

```
python run_validation.py \
    --num_landmarks 32 \
    --num_hidden_layers 2 \
    --hidden_size 128 \
    --num_attention_heads 2 \
    --batch_size 1 --num_batch 1 \
    2>&1 | tee logs/run_validation_m32_h128_$(date +%Y%m%d_%H%M%S).log
```
Run the following command for Bert:

```
python run_bert_validation.py \
        --hidden_size 128 \
        --num_hidden_layers 2 \
        --num_attention_heads 2 \
        --batch_size 1 --num_batch 1 \
        2>&1 | tee logs/run_bert_validation_n${N}_h128_$(date +%Y%m%d_%H%M%S).log
```
After running the command, npz files will be created. 
For example : Dietformer/models/Nystromformer/code/dietformer/outputs_n128_m32_L2_h128_nystrom

---

### 3. Generate and Verify Zero-Knowledge Proofs

For lookup verification using **Lasso**, the Rust toolchain must be set to **nightly**.

```
# 1. Install the latest nightly toolchain
rustup install nightly

# 2. Use nightly only in this project directory
rustup overrideset nightly
```

Then Go to Dietformer/circuits, and run the tests to generate and verify the proofs:

If you want to run all related experiments, execute the shell script corresponding to each model.
```
chmod +x measure_small.sh

nohup caffeinate -i bash measure_base.sh \
    > logs/master_$(date +%Y%m%d_%H%M%S).log 2>&1 &
disown $!
```
If you only want to run a specific experiment, invoke only the corresponding experiment function.
```
cargo test attention_test --release -- --no-capture
cargo test mlp_test --release -- --no-capture
```
