# zkDietFormer: Succinct and Subquadratic Proofs for Low-Rank Self-Attention via Polynomial Commitments

## Branch Descriptions

1. `feat/nystrom-bert-huggingface`

- Nyströmformer 기반 BERT 구현
- 입력 X와 가중치 W의 행렬 곱 연산을 포함하기 전 버전

2. `feat/nystrom-bert-huggingface-inputx`

- 입력 X와 가중치 W의 행렬 곱 연산을 포함한 버전
- 입력 데이터 X를 반영하도록 구현 변경

3. `feat/nystrom-bert-huggingface-global`

- Global Commitment 기능 추가
- Attention 수식 자체의 증명에 집중하는 `measure_onlyeq` 코드 추가
  - Attention 식의 등식(equation) 자체를 증명하도록 구현
- Q, K, V, Y, Z*에 대한 Commitment 추가
- Global Commitment 시간은 본문에서 별도로 한 번 언급하며, 실험 결과의 Table 2에는 반영하지 않음

4. `feat/nystrom-bert-huggingface-general` ⭐ (Current Branch)

- Attention 구조의 일반화 실험 추가
- Softmax Attention 및 Linformer를 대상으로 확장 실험 수행

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
