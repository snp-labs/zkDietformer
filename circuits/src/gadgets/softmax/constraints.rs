use std::fs::File;
use std::path::Path;
use std::error::Error;

use ark_ff::{PrimeField, Zero, One};
use ndarray::{Array2, Array0};
use ndarray_npy::NpzReader;

/// 128비트 정수(u128) 하나를 고정된 비트 크기(limb_bits)단위를 가지는 k 세그먼트로 분해함.
pub fn decompose_to_limbs_dynamic(mut unsigned: u128, limb_bits: usize, k_segments: usize) -> Vec<u128> {
    let limb_mask: u128 = if limb_bits >= 128 { !0u128 } else { (1u128 << limb_bits) - 1u128 }; 
    let mut limbs = vec![0u128; k_segments];
    for k in 0..k_segments {
        limbs[k] = (unsigned >> (k * limb_bits)) & limb_mask;
    }
    limbs
}

pub fn build_softmax_witnesses<F: PrimeField>(
    matmul_ints: Vec<Vec<i128>>, // rows x cols
    tables: &Vec<Vec<F>>,        // tables[k][limb_value]
    limb_bits: usize,
) -> (
    Vec<Vec<Vec<Vec<F>>>>,                  // matmul_outputs: [batch=1][head=1][rows][cols]
    Vec<Vec<Vec<Vec<Vec<F>>>>>,            // limbs:   [batch=1][head=1][rows][cols][K] as Vec for dynamic K
    Vec<Vec<Vec<Vec<Vec<F>>>>>,            // exp_limb_vals: per-limb exp lookup value (Field)
    Vec<Vec<Vec<Vec<F>>>>,                  // softmax_outputs same dims (normalized)
) {
    let rows = matmul_ints.len();
    let cols = if rows > 0 { matmul_ints[0].len() } else { 0 };

    let k_segments = tables.len();
    // row length (table width)
    let table_width = if k_segments > 0 { tables[0].len() } else { 0 };

    // matmul_outputs as F
    let mut matmul_outputs_field = vec![vec![vec![vec![F::zero(); cols]; rows]; 1]; 1];

    // limbs / exp_limb_vals / softmax outputs (use Vec to hold K dynamically)
    let mut limbs_field = vec![vec![vec![vec![vec![F::zero(); 0]; cols]; rows]; 1]; 1];
    let mut exp_limb_vals_field = vec![vec![vec![vec![vec![F::zero(); 0]; cols]; rows]; 1]; 1];
    let mut softmax_outputs_field = vec![vec![vec![vec![F::zero(); cols]; rows]; 1]; 1];

    // initialize inner K-size vectors
    for n in 0..1 {
        for h in 0..1 {
            for i in 0..rows {
                for j in 0..cols {
                    limbs_field[n][h][i][j] = vec![F::zero(); k_segments];
                    exp_limb_vals_field[n][h][i][j] = vec![F::zero(); k_segments];
                }
            }
        }
    }

    // First pass: decompose and lookup, store per-element exp_product in softmax_outputs_field
    for i in 0..rows {
        for j in 0..cols {
            let val = matmul_ints[i][j];
            if val < 0 {
                panic!("Negative matmul output encountered (value={}) — implement signed mapping if needed", val);
            }
            let unsigned = val as u128;
            let limbs = decompose_to_limbs_dynamic(unsigned, limb_bits, k_segments);

            // convert matmul output to F (assume non-negative)
            let x_f = F::from(unsigned);
            matmul_outputs_field[0][0][i][j] = x_f;

            // limbs -> to field; also look up per-limb exp fixed-point value from tables
            let mut exp_prod = F::one();
            for k in 0..k_segments {
                let limb_val = limbs[k];
                // bounds check: ensure limb_val < table_width
                if (limb_val as usize) >= table_width {
                    panic!(
                        "limb value {} out of bounds for table width {}. \
                         Check limb_bits used when generating tables (current limb_bits={}).",
                        limb_val, table_width, limb_bits
                    );
                }
                let limb_fe = F::from(limb_val);
                limbs_field[0][0][i][j][k] = limb_fe;

                // tables[k] is row k, index by limb_val
                let e_val = tables[k][limb_val as usize];
                exp_limb_vals_field[0][0][i][j][k] = e_val;

                exp_prod *= e_val;
            }

            // store exp product for normalization later
            softmax_outputs_field[0][0][i][j] = exp_prod;
        }
    }

    // normalize per-row: s = exp_prod / sum_j exp_prod_j
    for i in 0..rows {
        let mut denom = F::zero();
        for j in 0..cols {
            denom += softmax_outputs_field[0][0][i][j];
        }
        for j in 0..cols {
            let s = softmax_outputs_field[0][0][i][j] / denom;
            softmax_outputs_field[0][0][i][j] = s;
        }
    }

    (
        matmul_outputs_field, // 이게 생 값. limb단위로 decompose하기 전
        limbs_field, // 각 k segment번째 테이블 인덱스, 2^?까지 적용한 값임
        exp_limb_vals_field, // 하나의 e^i에 대하여 세그먼트 나누어서 저장, e^한 상태임.
        softmax_outputs_field, // 소프트 맥스 결과 e^i / sum 모든 e
    )
}