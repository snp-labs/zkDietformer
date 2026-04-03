
use ark_ff::{PrimeField, BigInteger};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::fp::FpVar};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::fields::FieldVar;
use ark_r1cs_std::R1CSVar;
use crate::gadgets::{activation::gelu::split_remainder_and_push, dot_prod::constraints::*, matmul::constraints::*};

#[derive(Clone)]
pub struct MatmulBundle<F: PrimeField> {
    pub r_mean: Vec<Vec<Vec<Vec<F>>>>,
    pub q_sim: Vec<Vec<Vec<Vec<F>>>>,
    pub r_matmul: Vec<Vec<Vec<Vec<F>>>>,
    pub m: F,
    pub m_prime: F,
    pub z_sim: F,
}

#[derive(Clone)]
pub struct ZenParams<F: PrimeField> {
    pub m: FpVar<F>,
    pub m_prime: FpVar<F>,
    pub z_w: FpVar<F>,
    pub z_x: FpVar<F>,
    pub z_y: FpVar<F>,
    pub two_k: FpVar<F>,
}

#[derive(Clone)]
pub struct MatmulData<F: PrimeField> {
    pub q_sim: Vec<Vec<MatrixVar<F>>>,
    pub r_matmul: Vec<Vec<MatrixVar<F>>>,
}

#[derive(Clone)]
pub struct PseudoinverseBundle<F: PrimeField> {
    pub q_inv: Vec<Vec<Vec<Vec<F>>>>,
    pub z_inv: F,
    pub m_inv: [F; 4],       // M_inv1, 2, 3, 4
    pub m_prime_inv: [F; 4], // M_prime_inv1, 2, 3, 4
    pub q_y_intermediates: Vec<Vec<Vec<Vec<Vec<F>>>>>, // [Y1, Y3]
    pub r_inv_remainders: Vec<Vec<Vec<Vec<Vec<F>>>>>,  // [R1, R2, R3, R4]
}

pub fn new_matrix_witness_4d<F: PrimeField>(
    cs: ConstraintSystemRef<F>, 
    data: &Vec<Vec<Vec<Vec<F>>>>
) -> Vec<Vec<MatrixVar<F>>> {
    data.iter().map(|heads| {
        heads.iter().map(|matrix| {
            new_matrix_witness(cs.clone(), matrix)
        }).collect()
    }).collect()
}

impl<F: PrimeField> MatmulBundle<F> {
    pub fn alloc_and_setup(
        &self, 
        cs: ConstraintSystemRef<F>, 
        z_q: F, 
        z_k: F, 
        two_k: &FpVar<F>
    ) -> (MatmulData<F>, ZenParams<F>) {
        // 1. 위트니스 할당
        let data = MatmulData {
            q_sim: new_matrix_witness_4d(cs.clone(), &self.q_sim),
            r_matmul: new_matrix_witness_4d(cs, &self.r_matmul),
        };

        // 2. 파라미터 설정
        let params = ZenParams {
            m: FpVar::Constant(self.m),
            m_prime: FpVar::Constant(self.m_prime),
            z_w: FpVar::Constant(z_q),
            z_x: FpVar::Constant(z_k),
            z_y: FpVar::Constant(self.z_sim),
            two_k: two_k.clone(),
        };

        (data, params)
    }
}

pub fn gen_challenge_vector<F: PrimeField>(
    r: &FpVar<F>,
    len: usize
) -> Vec<FpVar<F>> {
    let mut powers = Vec::with_capacity(len);
    let mut curr = r.clone();
    for _ in 0..len {
        powers.push(curr.clone());
        curr = curr * r;
    }
    powers
}

pub fn compute_circuit_landmarks<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    q_k_proc: &Vec<MatrixVar<F>>, 
    r_mean_k: &Vec<MatrixVar<F>>, 
    seg_size: usize,
) -> Result<Vec<MatrixVar<F>>, SynthesisError> {
    let mut q_landmarks = Vec::new();
    let seg_fp = FpVar::Constant(F::from(seg_size as u64));

    for (h, head_matrix) in q_k_proc.iter().enumerate() {
        let seq_len = head_matrix.len();
        let dim = head_matrix[0].len();
        let num_landmarks = seq_len / seg_size;
        let mut head_landmarks = Vec::with_capacity(num_landmarks);

        for l in 0..num_landmarks {
            let base = l * seg_size;
            let mut row_landmarks = Vec::with_capacity(dim);
            for d in 0..dim {
                // 1. 서킷 내에서 구간 합(Sum) 계산
                let mut sum_val = FpVar::zero();
                for s in 0..seg_size {
                    sum_val += &head_matrix[base + s][d];
                }

                // 2. Landmark 값을 위트니스로 할당 (몫 계산)
                let landmark_val = FpVar::new_witness(cs.clone(), || {
                    let sum_field = sum_val.value()?;
                    // Convert the field element to a BigInteger
                    let big_int = sum_field.into_bigint();
                    
                    // Perform the division directly
                    let sum_u64 = big_int.as_ref()[0] / seg_size as u64;
                    
                    Ok(F::from(sum_u64))
                })?;

                // 3. ZEN 식: Sum = (Landmark * Seg) + Remainder 검증
                let check_eq = (&landmark_val * &seg_fp) + &r_mean_k[h][l][d];
                sum_val.enforce_equal(&check_eq)?;

                row_landmarks.push(landmark_val);
            }
            head_landmarks.push(row_landmarks);
        }
        q_landmarks.push(head_landmarks);
    }
    Ok(q_landmarks)
}

pub fn compute_grouped_rhs<F: PrimeField>(
    q_w: &MatrixVar<F>,        
    q_x_t: &MatrixVar<F>,       
    m_var: &FpVar<F>,           
    m_prime_var: &FpVar<F>,
    z_q: &FpVar<F>, z_k: &FpVar<F>, z_y: &FpVar<F>,
    r_vec: &[FpVar<F>]          
) -> Result<Vec<FpVar<F>>, SynthesisError> {
    let n_dim = FpVar::Constant(F::from(q_w[0].len() as u64));
    let two_k = FpVar::Constant(F::from(1u64 << 22));
    
    // constant_term = (n * zq * zk) + M'
    let constant_term = (&n_dim * z_q * z_k) + m_prime_var;

    // 프리발드 최적화: G1*r = Q_w * (Q_x_t * r)
    let qxt_r = mat_vec_mul(q_x_t, r_vec)?; 
    let g1_r = mat_vec_mul(q_w, &qxt_r)?;   

    // G2*r = (z_k * Q_w) * r_sum
    let r_sum: FpVar<F> = r_vec.iter().sum();
    let g2_r: Vec<FpVar<F>> = q_w.iter().map(|row| {
        let row_sum: FpVar<F> = row.iter().sum();
        row_sum * z_k * &r_sum
    }).collect();

    // G3*r = z_q * (Q_x_t * r).sum()
    let qxt_r_sum: FpVar<F> = qxt_r.iter().sum();
    let g3_r = z_q * &qxt_r_sum;

    // Hoist loop-invariant products: both are constant across i.
    let const_r_sum = &constant_term * &r_sum;
    let const_offset = &const_r_sum - &g3_r;

    // M * (G1_r[i] + constant_term*r_sum - G2_r[i] - G3_r)
    let mut rhs = Vec::with_capacity(g1_r.len());
    for i in 0..g1_r.len() {
        let val = &g1_r[i] + &const_offset - &g2_r[i];
        rhs.push(val * m_var);
    }
    Ok(rhs)
}

pub fn transpose_matrix<F: PrimeField>(m: &MatrixVar<F>) -> MatrixVar<F> {
    let rows = m.len();
    if rows == 0 { return vec![]; }
    let cols = m[0].len();
    
    let mut transposed = vec![vec![FpVar::zero(); rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = m[i][j].clone();
        }
    }
    transposed
}

pub fn enforce_freivald<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    w_matrix: &MatrixVar<F>,        
    x_matrix_t: &MatrixVar<F>,      
    q_y_sim_head: &MatrixVar<F>,    
    r_matmul_head: &MatrixVar<F>,   
    r_matmul_raw: &Vec<Vec<F>>,
    params: &ZenParams<F>,
    r_pows: &[FpVar<F>],
    h_r1_low_vars: &mut Vec<FpVar<F>>,
    h_r1_high_vars: &mut Vec<FpVar<F>>,
) -> Result<(), SynthesisError> {
    
    // 1. LHS : (Q_y * 2^k + R) * r
    let qy_r = mat_vec_mul(q_y_sim_head, r_pows)?; 
    let r_mat_r = mat_vec_mul(r_matmul_head, r_pows)?;
    
    let lhs: Vec<FpVar<F>> = qy_r.iter().zip(r_mat_r.iter())
        .map(|(y, r)| y * &params.two_k + r).collect();

    // 2. RHS : M * (G1 - G2 - G3 + Const) * r
    let rhs = compute_grouped_rhs(
        w_matrix,
        x_matrix_t, 
        &params.m, 
        &params.m_prime, 
        &params.z_w, 
        &params.z_x, 
        &params.z_y, 
        r_pows)?;

    for (l, r) in lhs.iter().zip(rhs.iter()) {
        l.enforce_equal(r)?;
    }

    // 2. 22-bit lookup index
    for i in 0..r_matmul_head.len() {
        for j in 0..r_matmul_head[i].len() {
            let (low, high) = split_remainder_and_push(
                cs.clone(), 
                &r_matmul_head[i][j], 
                r_matmul_raw[i][j]
            )?;
            h_r1_low_vars.push(low);
            h_r1_high_vars.push(high);
        }
    }
    
    Ok(())
}