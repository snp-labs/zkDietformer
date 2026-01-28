//! Constraint implementations for matrix multiplication and linear layers.

use ark_ff::PrimeField;
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::fp::FpVar};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::fields::FieldVar;

use crate::gadgets::dot_prod::constraints::*;

/// Row-major matrix representation inside a circuit.
pub type MatrixVar<F> = Vec<Vec<FpVar<F>>>;

/// Row-major constant matrix.
pub type MatrixConst<F> = Vec<Vec<F>>;

/// Allocates a matrix witness in row-major form.
pub fn new_matrix_witness<F: PrimeField>(cs: ConstraintSystemRef<F>, m: &[Vec<F>]) -> MatrixVar<F> {
    let mut out = Vec::with_capacity(m.len());
    for row in m.iter() {
        // Vec<FpVar<F>> implements AllocVar<[F], F>, so this works.
        let row_var = Vec::<FpVar<F>>::new_witness(cs.clone(), || Ok(row.clone())).unwrap();
        out.push(row_var);
    }
    out
}

/// Allocates a vector witness.
pub fn new_vector_witness<F: PrimeField>(cs: ConstraintSystemRef<F>, v: &[F]) -> Vec<FpVar<F>> {
    Vec::<FpVar<F>>::new_witness(cs, || Ok(v.to_vec())).unwrap()
}

/// Computes matrix multiplication `C = A * B`.
///
/// - `A`: r × k  
/// - `B`: k × c  
/// - `C`: r × c
pub fn matmul<F: PrimeField>(a: &MatrixVar<F>, b: &MatrixVar<F>) -> MatrixVar<F> {
    let r = a.len();
    assert!(r > 0);
    let k = a[0].len();
    assert!(k > 0);

    debug_assert!(a.iter().all(|row| row.len() == k));

    assert_eq!(b.len(), k);
    let c = b[0].len();
    assert!(c > 0);
    debug_assert!(b.iter().all(|row| row.len() == c));

    let mut b_cols: Vec<Vec<FpVar<F>>> = Vec::with_capacity(c);
    for j in 0..c {
        let mut col = Vec::with_capacity(k);
        for t in 0..k {
            col.push(b[t][j].clone());
        }
        b_cols.push(col);
    }

    let mut out: MatrixVar<F> = Vec::with_capacity(r);
    for i in 0..r {
        let mut row_out: Vec<FpVar<F>> = Vec::with_capacity(c);
        for j in 0..c {
            let v = dot_product(&a[i], &b_cols[j]);
            row_out.push(v);
        }
        out.push(row_out);
    }
    out
}

/// Computes matrix-vector multiplication `y = M * v`.
pub fn mat_vec_mul<F: PrimeField>(
    m: &MatrixVar<F>,
    v: &[FpVar<F>],
) -> Result<Vec<FpVar<F>>, SynthesisError> {
    let rows = m.len();
    assert!(rows > 0);
    let cols = m[0].len();
    assert!(cols > 0);
    assert_eq!(v.len(), cols);

    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        out.push(dot_product(&m[i], v));
    }
    Ok(out)
}

/// Enforces Freivalds' check for `A * B = C`.
///
/// Enforces `A * (B * r) = C * r`.
pub fn enforce_freivalds<F: PrimeField>(
    a: &MatrixVar<F>,
    b: &MatrixVar<F>,
    c: &MatrixVar<F>,
    r_vec: &[FpVar<F>],
) -> Result<(), SynthesisError> {
    let r_rows = a.len();
    assert!(r_rows > 0);
    let k = a[0].len();
    assert!(k > 0);
    debug_assert!(a.iter().all(|row| row.len() == k));

    assert_eq!(b.len(), k);
    let c_cols = b[0].len();
    assert!(c_cols > 0);
    debug_assert!(b.iter().all(|row| row.len() == c_cols));

    assert_eq!(c.len(), r_rows);
    debug_assert!(c.iter().all(|row| row.len() == c_cols));

    assert_eq!(r_vec.len(), c_cols);

    // 1) br = B * r   (k x c) * (c) = (k)
    let br = mat_vec_mul(b, r_vec)?;

    // 2) left = A * br  (r x k) * (k) = (r)
    let left = mat_vec_mul(a, &br)?;

    // 3) right = C * r  (r x c) * (c) = (r)
    let right = mat_vec_mul(c, r_vec)?;

    // 4) Enforce left[i] == right[i]
    for (li, ri) in left.iter().zip(right.iter()) {
        li.enforce_equal(ri)?;
    }

    Ok(())
}

/// Computes a linear layer `y = W * x + b`.
pub fn linear_vec<F: PrimeField>(
    w: &MatrixVar<F>,
    x: &[FpVar<F>],
    b: &[FpVar<F>],
) -> Result<Vec<FpVar<F>>, SynthesisError> {
    let out_dim = w.len();
    assert!(out_dim > 0);
    let in_dim = w[0].len();
    assert!(in_dim > 0);

    debug_assert!(w.iter().all(|row| row.len() == in_dim));
    assert_eq!(x.len(), in_dim);
    assert_eq!(b.len(), out_dim);

    // y = W * x
    let mut y = mat_vec_mul(w, x)?;

    // y += b
    for (yi, bi) in y.iter_mut().zip(b.iter()) {
        *yi += bi.clone();
    }
    Ok(y)
}

/// Computes a batched linear layer `Y = X * W + b`.
pub fn linear_mat<F: PrimeField>(
    x: &MatrixVar<F>,
    w: &MatrixVar<F>,
    b: &[FpVar<F>],
) -> Result<MatrixVar<F>, SynthesisError> {
    let batch = x.len();
    assert!(batch > 0);
    let in_dim = x[0].len();
    assert!(in_dim > 0);
    debug_assert!(x.iter().all(|row| row.len() == in_dim));

    assert_eq!(w.len(), in_dim);
    let out_dim = w[0].len();
    assert!(out_dim > 0);
    debug_assert!(w.iter().all(|row| row.len() == out_dim));

    assert_eq!(b.len(), out_dim);

    // Y = X * W
    let mut y = matmul(x, w);

    // Broadcast bias
    for i in 0..batch {
        for j in 0..out_dim {
            y[i][j] += b[j].clone();
        }
    }

    Ok(y)
}

/// 행렬 곱셈 한세트
#[derive(Clone)]
pub struct MatmulBundle<F: PrimeField> {
    pub a: Vec<Vec<Vec<Vec<F>>>>,   // [Batch, Head, Seq, Dim]
    pub b: Vec<Vec<Vec<Vec<F>>>>,   // [Batch, Head, Dim, Seq]
    pub c: Vec<Vec<Vec<Vec<F>>>>,   // [Batch, Head, Seq, Seq]
    pub div: Vec<Vec<Vec<Vec<F>>>>,
    pub rem: Vec<Vec<Vec<Vec<F>>>>,
    // 각 행렬 연산마다 고유한 양자화 상수
    pub input_0: F,
    pub c_0_converted: F,
}

/// freivald with ZEN quantization
pub fn enforce_quantized_freivalds<F: PrimeField>(
    a_var: &MatrixVar<F>,         // [seq_len x head_dim]
    b_var: &MatrixVar<F>,         // [head_dim x seq_len] (전치된 상태)
    c_var: &MatrixVar<F>,         // [seq_len x seq_len]
    div_var: &MatrixVar<F>,       // [seq_len x seq_len]
    rem_var: &MatrixVar<F>,       // [seq_len x seq_len]
    input_0_var: &FpVar<F>,       // zero point (a0, b0)
    c_0_var: &FpVar<F>,           // c zero point
    r_var: &FpVar<F>,             // Fiat-Shamir challenge scalar
    m_var: &FpVar<F>,             // multiplier (s1*s2/s3)
) -> Result<(), SynthesisError> {
    let a_rows = a_var.len();
    let a_cols = if !a_var.is_empty() { a_var[0].len() } else { 0 };
    let b_rows = b_var.len();
    let b_cols = if !b_var.is_empty() { b_var[0].len() } else { 0 };

    // 1. 차원 검증
    if a_cols != b_rows { return Err(SynthesisError::Unsatisfiable); }

    // 2. 상수 설정
    let two_power_8_constant = FpVar::<F>::Constant(F::from(256u64));
    let two_power_22_constant = FpVar::<F>::Constant(F::from(1u64 << 22));

    // 3. r_pows 생성 [r^0, r^1, ..., r^{n-1}]
    let mut r_pows: Vec<FpVar<F>> = Vec::with_capacity(b_cols);
    let mut curr_r = FpVar::<F>::Constant(F::one());
    for _ in 0..b_cols {
        r_pows.push(curr_r.clone());
        curr_r = &curr_r * r_var;
    }

    let sum_r = r_pows.iter().fold(FpVar::zero(), |acc, x| acc + x);

    // --- Left side: m * ((A-a0)(B-b0) r + c0_corr) ---
    // (AB)r = A(Br)
    let br = mat_vec_mul(b_var, &r_pows)?;
    let term1 = mat_vec_mul(a_var, &br)?;

    // (Ab0)r = b0 * sum_r * (RowSums of A)
    let b0_sum_r = input_0_var * &sum_r;
    let term2: Vec<FpVar<F>> = a_var.iter()
        .map(|row| {
            let row_sum = row.iter().fold(FpVar::zero(), |acc, x| acc + x);
            row_sum * &b0_sum_r
        })
        .collect();

    // (a0B)r = a0 * sum(br)
    let sum_br = br.iter().fold(FpVar::zero(), |acc, x| acc + x);
    let term3_scalar = &sum_br * input_0_var;

    // (a0b0)r = head_dim * a0 * b0 * sum_r
    let head_dim_fpvar = FpVar::Constant(F::from(a_cols as u64));
    let term4_val = input_0_var * input_0_var * &sum_r * &head_dim_fpvar;

    // c0_corr = c0 * sum_r
    let c0_corr = c_0_var * &sum_r;

    let mut left_side: Vec<FpVar<F>> = Vec::with_capacity(a_rows);
    for i in 0..a_rows {
        let res = (&term1[i] - &term2[i] - &term3_scalar + &term4_val + &c0_corr) * m_var;
        left_side.push(res);
    }

    // --- Right side: ((C + div*2^8)*2^22 + rem) * r ---
    let mut cc_var_matrix: MatrixVar<F> = Vec::with_capacity(a_rows);
    for i in 0..a_rows {
        let mut row = Vec::with_capacity(b_cols);
        for j in 0..b_cols {
            let val = (&c_var[i][j] + (&div_var[i][j] * &two_power_8_constant)) 
                      * &two_power_22_constant + &rem_var[i][j];
            row.push(val);
        }
        cc_var_matrix.push(row);
    }

    let right_side = mat_vec_mul(&cc_var_matrix, &r_pows)?;

    // Equality Enforce
    for (l, r) in left_side.iter().zip(right_side.iter()) {
        l.enforce_equal(r)?;
    }

    Ok(())
}


#[cfg(test)]
#[allow(missing_docs)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_r1cs_std::{R1CSVar, alloc::AllocVar};
    use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef};
    use ark_std::{
        Zero,
        rand::{SeedableRng, rngs::StdRng},
    };

    fn native_matmul(a: &[Vec<Fr>], b: &[Vec<Fr>]) -> Vec<Vec<Fr>> {
        let r = a.len();
        let k = a[0].len();
        let c = b[0].len();
        let mut out = vec![vec![Fr::zero(); c]; r];

        for i in 0..r {
            for j in 0..c {
                let mut acc = Fr::zero();
                for t in 0..k {
                    acc += a[i][t] * b[t][j];
                }
                out[i][j] = acc;
            }
        }
        out
    }

    fn native_linear_vec(w: &[Vec<Fr>], x: &[Fr], b: &[Fr]) -> Vec<Fr> {
        // y = W * x + b
        // W: out x in, x: in, b: out
        let out_dim = w.len();
        let in_dim = w[0].len();
        assert_eq!(x.len(), in_dim);
        assert_eq!(b.len(), out_dim);

        let mut y = vec![Fr::zero(); out_dim];
        for i in 0..out_dim {
            let mut acc = Fr::zero();
            for t in 0..in_dim {
                acc += w[i][t] * x[t];
            }
            y[i] = acc + b[i];
        }
        y
    }

    fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<Fr>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| Fr::rand(rng)).collect())
            .collect()
    }

    fn random_vector(len: usize, rng: &mut StdRng) -> Vec<Fr> {
        (0..len).map(|_| Fr::rand(rng)).collect()
    }

    #[test]
    fn test_constraint_compare_naive_vs_freivalds() {
        let n: usize = 1 << 10;
        let r = n;
        let k = n;
        let c = n;

        let mut rng = StdRng::seed_from_u64(42);

        // A: r x k, B: k x c
        let a_vals = random_matrix(r, k, &mut rng);
        let b_vals = random_matrix(k, c, &mut rng);

        // C = A*B (native)
        let c_vals = native_matmul(&a_vals, &b_vals);

        // -----------------------------
        // 1) Naive: compute A*B in-circuit and enforce equals C
        // -----------------------------
        let cs_naive = ConstraintSystem::<Fr>::new_ref();

        let a_naive = new_matrix_witness::<Fr>(cs_naive.clone(), &a_vals);
        let b_naive = new_matrix_witness::<Fr>(cs_naive.clone(), &b_vals);
        let c_naive = new_matrix_witness::<Fr>(cs_naive.clone(), &c_vals);

        let computed = matmul(&a_naive, &b_naive);
        for i in 0..r {
            for j in 0..c {
                computed[i][j].enforce_equal(&c_naive[i][j]).unwrap();
            }
        }

        let naive_constraints = cs_naive.num_constraints();
        println!(
            "number of constraints (naive A*B == C), n={}: {}",
            n, naive_constraints
        );
        assert!(cs_naive.is_satisfied().unwrap());

        // -----------------------------
        // 2) Freivalds: enforce A*(B*r) == C*r
        // -----------------------------
        let cs_frei = ConstraintSystem::<Fr>::new_ref();

        let a_frei = new_matrix_witness::<Fr>(cs_frei.clone(), &a_vals);
        let b_frei = new_matrix_witness::<Fr>(cs_frei.clone(), &b_vals);
        let c_frei = new_matrix_witness::<Fr>(cs_frei.clone(), &c_vals);

        // r_vec: length c
        let r_native = random_vector(c, &mut rng);
        let r_vec = Vec::<FpVar<Fr>>::new_witness(cs_frei.clone(), || Ok(r_native)).unwrap();

        enforce_freivalds(&a_frei, &b_frei, &c_frei, &r_vec).unwrap();

        let frei_constraints = cs_frei.num_constraints();
        println!(
            "number of constraints (freivalds A*(B*r)==C*r), n={}: {}",
            n, frei_constraints
        );
        assert!(cs_frei.is_satisfied().unwrap());

        // -----------------------------
        // Compare
        // -----------------------------
        println!(
            "constraint ratio naive / freivalds (n={}): {:.2}x",
            n,
            (naive_constraints as f64) / (frei_constraints as f64)
        );
        assert!(naive_constraints > frei_constraints);
    }

    #[test]
    fn test_matmul() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let mut rng = StdRng::seed_from_u64(12345);

        // A: 5 x 7
        // B: 7 x 3
        // C: 5 x 3
        let a_vals = random_matrix(5, 7, &mut rng);
        let b_vals = random_matrix(7, 3, &mut rng);
        let expected = native_matmul(&a_vals, &b_vals);

        let a = new_matrix_witness::<Fr>(cs.clone(), &a_vals);
        let b = new_matrix_witness::<Fr>(cs.clone(), &b_vals);

        let c = matmul(&a, &b);

        assert_eq!(c.len(), 5);
        assert_eq!(c[0].len(), 3);

        for i in 0..5 {
            for j in 0..3 {
                assert_eq!(c[i][j].value().unwrap(), expected[i][j]);
            }
        }

        println!(
            "number of constraints (matmul 5x7 * 7x3): {}",
            cs.num_constraints()
        );
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_freivalds_enforce() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let mut rng = StdRng::seed_from_u64(4242);

        // A: 5 x 7
        // B: 7 x 3
        // C: 5 x 3
        let a_vals = random_matrix(5, 7, &mut rng);
        let b_vals = random_matrix(7, 3, &mut rng);
        let c_vals = native_matmul(&a_vals, &b_vals);

        let a = new_matrix_witness::<Fr>(cs.clone(), &a_vals);
        let b = new_matrix_witness::<Fr>(cs.clone(), &b_vals);
        let c = new_matrix_witness::<Fr>(cs.clone(), &c_vals);

        // r: length 3 (columns of C)
        let r_native = random_vector(3, &mut rng);
        let r = Vec::<FpVar<Fr>>::new_witness(cs.clone(), || Ok(r_native)).unwrap();

        enforce_freivalds(&a, &b, &c, &r).unwrap();

        println!(
            "constraints (freivalds 5x7 * 7x3): {}",
            cs.num_constraints()
        );
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn test_linear_vec() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let mut rng = StdRng::seed_from_u64(2025);

        // W: out x in, x: in, b: out
        let out_dim = 5usize;
        let in_dim = 7usize;

        let w_vals = random_matrix(out_dim, in_dim, &mut rng);
        let x_vals = random_vector(in_dim, &mut rng);
        let b_vals = random_vector(out_dim, &mut rng);

        let expected = native_linear_vec(&w_vals, &x_vals, &b_vals);

        let w = new_matrix_witness::<Fr>(cs.clone(), &w_vals);
        let x = new_vector_witness::<Fr>(cs.clone(), &x_vals);
        let b = new_vector_witness::<Fr>(cs.clone(), &b_vals);

        let y = linear_vec(&w, &x, &b).unwrap();

        assert_eq!(y.len(), out_dim);
        for i in 0..out_dim {
            assert_eq!(y[i].value().unwrap(), expected[i]);
        }

        println!(
            "number of constraints (linear_vec {}x{}): {}",
            out_dim,
            in_dim,
            cs.num_constraints()
        );
        assert!(cs.is_satisfied().unwrap());
    }
}
