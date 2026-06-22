# zkDietFormer: Succinct and Subquadratic Proofs for Low-Rank Self-Attention via Polynomial Commitments

## Current experimental branch

- feat/circuit-ZEN

## Summary

<img width="1920" height="1080" alt="제목을 입력해주세요  (9)" src="https://github.com/user-attachments/assets/f47eddd9-8312-4358-8ca7-6db587ad89b3" />

## How to Start

### 1. Pretrain the Model

The low-rank Transformer used in our experiments is **Nyströmformer**.

This repository includes an implementation of [Nystromformer](https://github.com/mlpen/Nystromformer).

Please follow the instructions in the repository’s README to pretrain the model.

---


### 2. Extract Intermediate Values

To generate zero-knowledge proofs for the **Attention** and **MLP** components,

we first extract the required intermediate values from the model.

These values are saved as a compressed **`.npz`** file.

Run the following command:

```
python run_validation.py \
--checkpoint0 \
--batch_size1 \
--num_batch1 \
--save_attn_npz ./outputs
```

After running the command, an `outputs` directory will be created.

Move this directory to:

```
Dietformer/circuits/outputs
```

---

### 3. Generate and Verify Zero-Knowledge Proofs

For lookup verification using **Lasso**, the Rust toolchain must be set to **nightly**.

```
# 1. Install the latest nightly toolchain
rustup install nightly

# 2. Use nightly only in this project directory
rustup overrideset nightly
```

Then run the tests to generate and verify the proofs:

```
cargo test attention_test --release -- --no-capture
cargo test mlp_test --release -- --no-capture
```
