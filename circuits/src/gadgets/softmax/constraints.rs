use crate::gadgets::activation::gelu::split_remainder_and_push;
use crate::gadgets::matmul::constraints::MatrixVar;
use crate::gadgets::matmul::quantized::new_matrix_witness_4d;

use ark_ff::{PrimeField, Zero, One};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::{
    fields::{FieldVar, fp::FpVar},
    prelude::*, R1CSVar,
};

/// v3 SoftmaxBundle: single table lookup (no multi-segment limb decomposition)
#[derive(Clone)]
pub struct SoftmaxBundle<F: PrimeField> {
    pub table_idx: Vec<Vec<Vec<Vec<F>>>>,  // [B][H][S][L] — index into exp table
    pub exp_total: Vec<Vec<Vec<Vec<F>>>>,  // [B][H][S][L] — exp(x) lookup result
    pub p: Vec<Vec<Vec<Vec<F>>>>,          // [B][H][S][L] — softmax probability
    pub r_div: Vec<Vec<Vec<Vec<F>>>>,      // [B][H][S][L] — division remainder (16-bit)
    pub r_sm: Vec<Vec<Vec<Vec<F>>>>,       // [B][H][S][L] — scaling remainder (52-bit)
    pub m_sm: F,                            // scalar scaling constant
}

impl<F: PrimeField> SoftmaxBundle<F> {
    pub fn alloc_vars(&self, cs: ConstraintSystemRef<F>) -> Result<SoftmaxVarBundle<F>, SynthesisError> {
        Ok(SoftmaxVarBundle {
            table_idx: new_matrix_witness_4d(cs.clone(), &self.table_idx),
            exp_total: new_matrix_witness_4d(cs.clone(), &self.exp_total),
            p: new_matrix_witness_4d(cs.clone(), &self.p),
            r_div: new_matrix_witness_4d(cs.clone(), &self.r_div),
            r_sm: new_matrix_witness_4d(cs.clone(), &self.r_sm),
            m_sm: FpVar::new_constant(cs.clone(), self.m_sm)?,
        })
    }
}

#[derive(Clone)]
pub struct SoftmaxVarBundle<F: PrimeField> {
    pub table_idx: Vec<Vec<MatrixVar<F>>>,  // [B][H] -> MatrixVar[S][L]
    pub exp_total: Vec<Vec<MatrixVar<F>>>,
    pub p: Vec<Vec<MatrixVar<F>>>,
    pub r_div: Vec<Vec<MatrixVar<F>>>,
    pub r_sm: Vec<Vec<MatrixVar<F>>>,
    pub m_sm: FpVar<F>,
}

/// v3 enforce_softmax: single table lookup
///
/// Constraints per element (i,j):
///   1. Scaling: (q_sim[i][j] * 2^k + r_matmul[i][j]) * m_sm = table_idx[i][j] * 2^(2k) + r_sm[i][j]
///   2. r_sm range: 52-bit split → 4 limbs (16+16+16+4)
///   3. table_idx collection into sm_idx_pool (Lasso CP-Link)
///   4. r_div split into 2 limbs (26-bit: 16+10) for range check
///   5. Division (per row): exp_total[i][j] * 2^frac = p[i][j] * exp_sum + r_div[i][j]
pub fn enforce_softmax<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_sim: &MatrixVar<F>,
    r_matmul: &MatrixVar<F>,
    r_sm: &MatrixVar<F>,         // 52-bit remainder
    m_sm: &FpVar<F>,
    table_idx: &MatrixVar<F>,    // v3: single table index
    exp_total: &MatrixVar<F>,    // v3: exp table lookup result
    p_out: &MatrixVar<F>,
    r_div: &MatrixVar<F>,
    r_div_raw: &Vec<Vec<F>>,     // raw r_div values for split
    frac_bits: usize,
    two_k: &FpVar<F>,
    sm_idx_pool: &mut Vec<FpVar<F>>,          // v3: single pool (C=1)
    r_sm_limbs: &mut [Vec<FpVar<F>>; 4],      // v3: 4 limbs for 52-bit r_sm
    r_div_lows: &mut Vec<FpVar<F>>,           // 26-bit split: low 16 bits
    r_div_highs: &mut Vec<FpVar<F>>,          // 26-bit split: high 10 bits
) -> Result<(), SynthesisError> {
    let s_len = q_sim.len();
    let l_len = q_sim[0].len();
    let frac_scale = F::from(1u128 << frac_bits);
    let two_2k = two_k * two_k;  // 2^(2*k_bits)

    for i in 0..s_len {
        let mut row_exp_sum = FpVar::zero();

        for j in 0..l_len {
            // 1. Scaling: (q_sim * 2^k + r_matmul) * m_sm = table_idx * 2^(2k) + r_sm
            let x_recovered = (q_sim[i][j].clone() * two_k) + &r_matmul[i][j];
            let lhs_scale = &x_recovered * m_sm;
            let rhs_scale = (&table_idx[i][j] * &two_2k) + &r_sm[i][j];
            lhs_scale.enforce_equal(&rhs_scale)?;

            // 2. r_sm 52-bit split → 4 limbs
            split_52bit_remainder(
                cs.clone(),
                &r_sm[i][j],
                r_sm[i][j].value().unwrap_or(F::zero()),
                r_sm_limbs,
            )?;

            // 3. Collect table_idx for Lasso CP-Link (C=1)
            sm_idx_pool.push(table_idx[i][j].clone());

            // 4. Split r_div into 2 limbs (26-bit: 16+10) for range check
            let (low, high) = split_remainder_and_push(
                cs.clone(),
                &r_div[i][j],
                r_div_raw[i][j],
            )?;
            r_div_lows.push(low);
            r_div_highs.push(high);

            // Accumulate row exp sum
            row_exp_sum += &exp_total[i][j];
        }

        // 5. Division constraint (per row):
        //    exp_total[i][j] * 2^frac = p[i][j] * exp_sum + r_div[i][j]
        for j in 0..l_len {
            let lhs_div = &exp_total[i][j] * frac_scale;
            let rhs_div = (&p_out[i][j] * &row_exp_sum) + &r_div[i][j];
            lhs_div.enforce_equal(&rhs_div)?;
        }
    }
    Ok(())
}

/// Split a 52-bit remainder into 4 limbs: [16, 16, 16, 4] bits.
/// Pushes each limb into the corresponding r_sm_limbs pool for range checking.
fn split_52bit_remainder<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    r_var: &FpVar<F>,
    r_val: F,
    limbs_out: &mut [Vec<FpVar<F>>; 4],
) -> Result<(), SynthesisError> {
    let r_u64 = r_val.into_bigint().as_ref()[0];
    let l0 = r_u64 & 0xFFFF;                 // bits [0:15]  — 16 bits
    let l1 = (r_u64 >> 16) & 0xFFFF;         // bits [16:31] — 16 bits
    let l2 = (r_u64 >> 32) & 0xFFFF;         // bits [32:47] — 16 bits
    let l3 = (r_u64 >> 48) & 0xF;            // bits [48:51] — 4 bits

    let l0_var = FpVar::new_witness(cs.clone(), || Ok(F::from(l0)))?;
    let l1_var = FpVar::new_witness(cs.clone(), || Ok(F::from(l1)))?;
    let l2_var = FpVar::new_witness(cs.clone(), || Ok(F::from(l2)))?;
    let l3_var = FpVar::new_witness(cs.clone(), || Ok(F::from(l3)))?;

    // Reconstruct: l0 + l1*2^16 + l2*2^32 + l3*2^48
    let shift16 = F::from(1u128 << 16);
    let shift32 = F::from(1u128 << 32);
    let shift48 = F::from(1u128 << 48);

    let reconstructed = &l0_var
        + &l1_var * shift16
        + &l2_var * shift32
        + &l3_var * shift48;

    r_var.enforce_equal(&reconstructed)?;

    limbs_out[0].push(l0_var);
    limbs_out[1].push(l1_var);
    limbs_out[2].push(l2_var);
    limbs_out[3].push(l3_var);

    Ok(())
}
