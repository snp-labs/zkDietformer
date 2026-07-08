//! Naive (non-Freivalds) quantized matrix multiplication constraint.
//!
//! Replaces the probabilistic Freivalds check with an explicit O(n^3)
//! element-wise verification of the ZEN quantized matmul relation:
//!
//!   Q_y[i][j] * 2^k + R[i][j] = M * (G1[i][j] - z_x * ΣQ_w[i] - z_w * ΣQ_xt[j] + n·z_w·z_x + M')
//!
//! where G1 = Q_w × Q_x_t is computed naively via nested loops.

use ark_ff::PrimeField;
use ark_r1cs_std::{eq::EqGadget, fields::fp::FpVar};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::fields::FieldVar;

use crate::gadgets::{
    activation::gelu::split_remainder_and_push,
    dot_prod::constraints::dot_product,
    matmul::constraints::MatrixVar,
    matmul::quantized::ZenParams,
};

/// Enforces the ZEN quantized matmul relation using naive O(n^3) matrix multiplication.
///
/// This is the non-Freivalds baseline: instead of checking via a random linear combination,
/// every output element is verified individually.
///
/// # Arguments
/// * `cs` — constraint system reference
/// * `w_matrix` — left matrix Q_w [rows × inner]
/// * `x_matrix_t` — right matrix Q_x^T [inner × cols]
/// * `q_y_sim_head` — quantized output Q_y [rows × cols]
/// * `r_matmul_head` — remainder R [rows × cols] (as FpVar)
/// * `r_matmul_raw` — remainder R [rows × cols] (as native field elements, for split)
/// * `params` — ZEN quantization parameters (M, M', z_w, z_x, z_y, 2^k)
/// * `h_r1_low_vars` — accumulator for low 16-bit remainder splits
/// * `h_r1_high_vars` — accumulator for high 6-bit remainder splits
pub fn enforce_naive_matmul<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    w_matrix: &MatrixVar<F>,
    x_matrix_t: &MatrixVar<F>,
    q_y_sim_head: &MatrixVar<F>,
    r_matmul_head: &MatrixVar<F>,
    r_matmul_raw: &Vec<Vec<F>>,
    params: &ZenParams<F>,
    h_r1_low_vars: &mut Vec<FpVar<F>>,
    h_r1_high_vars: &mut Vec<FpVar<F>>,
    bias: Option<&[F]>,
) -> Result<(), SynthesisError> {
    let rows = w_matrix.len();
    assert!(rows > 0);
    let inner = w_matrix[0].len();
    assert!(inner > 0);
    assert_eq!(x_matrix_t.len(), inner);
    let cols = x_matrix_t[0].len();
    assert!(cols > 0);

    // Precompute constants
    let n_dim = FpVar::Constant(F::from(inner as u64));
    // constant_term = n * z_w * z_x + M'
    let constant_term = &n_dim * &params.z_w * &params.z_x + &params.m_prime;

    // Precompute row sums of w_matrix: row_sum_w[i] = Σ_k w_matrix[i][k]
    let row_sums_w: Vec<FpVar<F>> = w_matrix
        .iter()
        .map(|row| {
            let s: FpVar<F> = row.iter().cloned().sum();
            s
        })
        .collect();

    // Precompute column sums of x_matrix_t: col_sum_xt[j] = Σ_k x_matrix_t[k][j]
    let mut col_sums_xt: Vec<FpVar<F>> = (0..cols).map(|_| FpVar::zero()).collect();
    for k in 0..inner {
        for j in 0..cols {
            col_sums_xt[j] = &col_sums_xt[j] + &x_matrix_t[k][j];
        }
    }

    // Transpose x_matrix_t columns for dot_product reuse
    let mut xt_cols: Vec<Vec<FpVar<F>>> = Vec::with_capacity(cols);
    for j in 0..cols {
        let mut col = Vec::with_capacity(inner);
        for k in 0..inner {
            col.push(x_matrix_t[k][j].clone());
        }
        xt_cols.push(col);
    }

    // For each output element (i, j):
    //   LHS = Q_y[i][j] * 2^k + R[i][j]
    //   G1[i][j] = Σ_k w_matrix[i][k] * x_matrix_t[k][j]
    //   RHS = M * (G1[i][j] - z_x * row_sum_w[i] - z_w * col_sum_xt[j] + constant_term)
    //   Enforce LHS == RHS
    for i in 0..rows {
        // Precompute per-row terms: z_x * row_sum_w[i]
        let g2_i = &params.z_x * &row_sums_w[i];
        // constant_offset_i = constant_term - g2_i
        let const_offset_i = &constant_term - &g2_i;

        for j in 0..cols {
            // G1[i][j] via dot product
            let g1_ij = dot_product(&w_matrix[i], &xt_cols[j]);

            // ZEN value = G1[i][j] - z_w * col_sum_xt[j] + const_offset_i [+ bias[j]]
            let mut zen_val = &g1_ij - &params.z_w * &col_sums_xt[j] + &const_offset_i;
            if let Some(b) = bias {
                zen_val += FpVar::Constant(b[j]);
            }

            // RHS = M * zen_val
            let rhs = &params.m * &zen_val;

            // LHS = Q_y[i][j] * 2^k + R[i][j]
            let lhs = &q_y_sim_head[i][j] * &params.two_k + &r_matmul_head[i][j];

            lhs.enforce_equal(&rhs)?;

            // Split remainder for 22-bit lookup
            let (low, high) = split_remainder_and_push(
                cs.clone(),
                &r_matmul_head[i][j],
                r_matmul_raw[i][j],
            )?;
            h_r1_low_vars.push(low);
            h_r1_high_vars.push(high);
        }
    }

    Ok(())
}
