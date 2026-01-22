use anyhow::{Context, Result};
use ndarray::{Ix0, Ix4, IxDyn, Ix2};
use ndarray_npy::NpzReader;
use std::fs::File;
use ark_ff::PrimeField;
use crate::{
    circuits::softmax_circuit::{SoftmaxCircuit, K_SEGMENTS},
    gadgets::softmax::constraints::build_softmax_witnesses,
    util::load::read_array_u64,
};
use num_bigint::BigInt;

/// 각 커널(K1, K2, K3)에 대해 소프트맥스 서킷 데이터를 로드하고 검증하는 공통 함수
pub fn load_softmax_circuit<F: PrimeField>(
    npz: &mut NpzReader<File>,
    suffix: &str,               // "_k1", "_k2" 등
    tables_field: &Vec<Vec<F>>, // 필드 요소로 변환된 테이블
    tables_u64: &ndarray::Array2<u64>, // 검증용 원본 u64 테이블
    soft_limb_bits: usize,
    soft_frac_bits: u32,
) -> Result<(SoftmaxCircuit<F>, Vec<Vec<i128>>, Vec<BigInt>, usize)> {
    // 1. 데이터 로드
    let targets_arr = read_array_u64(npz, &format!("matmul_targets{}", suffix))
        .context(format!("matmul_targets{} not found", suffix))?;
    let t4 = targets_arr.into_dimensionality::<Ix4>()?;
    
    let limb_idx_arr = read_array_u64(npz, &format!("limb_indices{}", suffix))
        .context(format!("limb_indices{} not found", suffix))?;
    let limb_idx_dyn = limb_idx_arr.into_dimensionality::<IxDyn>()?;

    let shape = t4.shape();
    let (batch, num_head, rows, cols) = (shape[0], shape[1], shape[2], shape[3]);
    let k_dim = limb_idx_dyn.shape()[4];
    let table_width = tables_u64.shape()[1];

    // 2. 가변 배열 초기화
    let mut matmul_outputs_all = vec![vec![vec![vec![F::zero(); cols]; rows]; num_head]; batch];
    let mut limbs_all = vec![vec![vec![vec![[F::zero(); K_SEGMENTS]; cols]; rows]; num_head]; batch];
    let mut exp_limb_vals_all = vec![vec![vec![vec![[F::one(); K_SEGMENTS]; cols]; rows]; num_head]; batch];
    let mut softmax_outputs_all = vec![vec![vec![vec![F::zero(); cols]; rows]; num_head]; batch];
    let mut exp_prod_field_all = vec![vec![vec![vec![F::one(); cols]; rows]; num_head]; batch];

    // 룩업증명에 필요해서 반환해야 할 값들
    let mut combined_matmul_ints: Vec<Vec<i128>> = Vec::with_capacity(batch * num_head * rows);

    // 룩업증명에 필요해서 반환해야 할 값들
    let mut prod_bigints_flat: Vec<BigInt> = Vec::new();

    // 3. 위트니스 생성
    for n in 0..batch {
        for h in 0..num_head {
            let mut matmul_ints = vec![vec![0i128; cols]; rows];
            for i in 0..rows {
                for j in 0..cols {
                    matmul_ints[i][j] = t4[[n, h, i, j]] as i128;
                }
            }

            let (matmul_outputs_field, limbs_field_dyn, exp_limb_vals_dyn, softmax_outputs_field) =
                build_softmax_witnesses::<F>(matmul_ints, tables_field, soft_limb_bits);

            for i in 0..rows {
                // push combined row for lookup (this is x's rows)
                let mut row_for_lookup: Vec<i128> = Vec::with_capacity(cols);
                for j in 0..cols {
                    matmul_outputs_all[n][h][i][j] = matmul_outputs_field[0][0][i][j];
                    softmax_outputs_all[n][h][i][j] = softmax_outputs_field[0][0][i][j];
                    
                    let mut limb_fixed = [F::zero(); K_SEGMENTS];
                    let mut exp_fixed = [F::one(); K_SEGMENTS];
                    let mut prod_field = F::one();
                    for k in 0..K_SEGMENTS {
                        limb_fixed[k] = limbs_field_dyn[0][0][i][j][k];
                        exp_fixed[k] = exp_limb_vals_dyn[0][0][i][j][k];
                        prod_field *= exp_fixed[k];
                    }
                    limbs_all[n][h][i][j] = limb_fixed;
                    exp_limb_vals_all[n][h][i][j] = exp_fixed;
                    exp_prod_field_all[n][h][i][j] = prod_field;

                    // push the matmul int for lookup rows
                    row_for_lookup.push(t4[[n, h, i, j]] as i128);

                    // compute prod_u128 from tables_u64 using limb indices
                    let mut prod_u128: u128 = 1;
                    for k in 0..k_dim {
                        let idx = limb_idx_dyn[ndarray::IxDyn(&[n, h, i, j, k])] as usize;
                        let v = tables_u64[[k, idx]];
                        prod_u128 = prod_u128.saturating_mul(v as u128);
                    }
                    prod_bigints_flat.push(BigInt::from(prod_u128));
                }
                combined_matmul_ints.push(row_for_lookup);
            }
        }
    }

    // 4. 수치 일관성 디버그 체크
    println!("DEBUG: Running numeric consistency checks for suffix {}...", suffix);
    for n in 0..batch {
        for h in 0..num_head {
            for i in 0..rows {
                for j in 0..cols {
                    // (1) Limb Recomposition 체크
                    let mut recomposed: u128 = 0;
                    for k in 0..k_dim {
                        let idx_val = limb_idx_dyn[IxDyn(&[n, h, i, j, k])];
                        recomposed |= (idx_val as u128) << (k * soft_limb_bits);
                    }
                    if recomposed != t4[[n, h, i, j]] as u128 {
                        return Err(anyhow::anyhow!("DIAG MISMATCH: recomposed limb mismatch at n{} h{} i{} j{}", n, h, i, j));
                    }

                    // (2) Exp Prod 체크 (테이블 값 기반)
                    let mut prod_u128: u128 = 1;
                    for k in 0..k_dim {
                        let idx = limb_idx_dyn[IxDyn(&[n, h, i, j, k])] as usize;
                        let v = tables_u64[[k, idx]];
                        prod_u128 = prod_u128.saturating_mul(v as u128);
                    }
                    if F::from(prod_u128) != exp_prod_field_all[n][h][i][j] {
                        return Err(anyhow::anyhow!("DIAG MISMATCH: exp_prod_field mismatch at n{} h{} i{} j{}", n, h, i, j));
                    }
                }
                // (3) Softmax 정규화 체크 (s * denom == exp_prod)
                let mut denom = F::zero();
                for j in 0..cols { denom += exp_prod_field_all[n][h][i][j]; }
                for j in 0..cols {
                    let lhs = softmax_outputs_all[n][h][i][j] * denom;
                    let rhs = exp_prod_field_all[n][h][i][j];
                    if lhs != rhs {
                        return Err(anyhow::anyhow!("DIAG MISMATCH: softmax normalization mismatch at n{} h{} i{} j{}", n, h, i, j));
                    }
                }
            }
        }
    }
    println!("DEBUG: All checks PASSED for suffix {}", suffix);

    let softmax_circuit = SoftmaxCircuit {
        matmul_outputs: matmul_outputs_all,
        limbs: limbs_all,
        exp_limb_vals: exp_limb_vals_all,
        softmax_outputs: softmax_outputs_all,
        frac_bits: soft_frac_bits,
    };

    Ok((softmax_circuit, combined_matmul_ints, prod_bigints_flat, cols))
}