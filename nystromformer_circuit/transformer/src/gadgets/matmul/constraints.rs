use ark_ff::PrimeField;
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::fp::FpVar};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

use crate::gadgets::dot_prod::constraints::*;

/// Row-major matrix: [r][c]
pub type MatrixVar<F> = Vec<Vec<FpVar<F>>>;
pub type MatrixConst<F> = Vec<Vec<F>>;

pub fn new_matrix_witness<F: PrimeField>(cs: ConstraintSystemRef<F>, m: &[Vec<F>]) -> MatrixVar<F> {
    let mut out = Vec::with_capacity(m.len());
    for row in m.iter() {
        // Vec<FpVar<F>> implements AllocVar<[F], F>, so this works.
        let row_var = Vec::<FpVar<F>>::new_witness(cs.clone(), || Ok(row.clone())).unwrap();
        out.push(row_var);
    }
    out
}

/// Compute C = A * B
/// A: r x k
/// B: k x c
/// C: r x c
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

/// Freivalds check constraints for A * B = C:
/// Enforce  A * (B * r) == C * r
///
/// Dimensions:
///   A: r x k
///   B: k x c
///   C: r x c
///   r: c
///
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

#[cfg(test)]
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

    fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<Fr>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| Fr::rand(rng)).collect())
            .collect()
    }

    fn random_vector(len: usize, rng: &mut StdRng) -> Vec<Fr> {
        (0..len).map(|_| Fr::rand(rng)).collect()
    }

    #[test]
    fn test_constraint_compare_naive_vs_freivalds_random_size() {
        // 설정
        let n: usize = 100; // 예: 100, 1000
        let r = n;
        let k = n;
        let c = n;

        // 재현 가능한 랜덤
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
    fn test_matmul_basic() {
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

        // A: 5 x 7
        // B: 7 x 3
        // C: 5 x 3
        let a_vals = vec![
            vec![
                Fr::from(1u64),
                Fr::from(2u64),
                Fr::from(3u64),
                Fr::from(4u64),
                Fr::from(5u64),
                Fr::from(6u64),
                Fr::from(7u64),
            ],
            vec![
                Fr::from(8u64),
                Fr::from(9u64),
                Fr::from(10u64),
                Fr::from(11u64),
                Fr::from(12u64),
                Fr::from(13u64),
                Fr::from(14u64),
            ],
            vec![
                Fr::from(15u64),
                Fr::from(16u64),
                Fr::from(17u64),
                Fr::from(18u64),
                Fr::from(19u64),
                Fr::from(20u64),
                Fr::from(21u64),
            ],
            vec![
                Fr::from(22u64),
                Fr::from(23u64),
                Fr::from(24u64),
                Fr::from(25u64),
                Fr::from(26u64),
                Fr::from(27u64),
                Fr::from(28u64),
            ],
            vec![
                Fr::from(29u64),
                Fr::from(30u64),
                Fr::from(31u64),
                Fr::from(32u64),
                Fr::from(33u64),
                Fr::from(34u64),
                Fr::from(35u64),
            ],
        ];

        let b_vals = vec![
            vec![Fr::from(36u64), Fr::from(37u64), Fr::from(38u64)],
            vec![Fr::from(39u64), Fr::from(40u64), Fr::from(41u64)],
            vec![Fr::from(42u64), Fr::from(43u64), Fr::from(44u64)],
            vec![Fr::from(45u64), Fr::from(46u64), Fr::from(47u64)],
            vec![Fr::from(48u64), Fr::from(49u64), Fr::from(50u64)],
            vec![Fr::from(51u64), Fr::from(52u64), Fr::from(53u64)],
            vec![Fr::from(54u64), Fr::from(55u64), Fr::from(56u64)],
        ];

        let c_vals = native_matmul(&a_vals, &b_vals);

        let a = new_matrix_witness::<Fr>(cs.clone(), &a_vals);
        let b = new_matrix_witness::<Fr>(cs.clone(), &b_vals);
        let c = new_matrix_witness::<Fr>(cs.clone(), &c_vals);

        // r: length 3 (columns of C)
        let r_native = vec![Fr::from(3u64), Fr::from(5u64), Fr::from(7u64)];
        let r = Vec::<FpVar<Fr>>::new_witness(cs.clone(), || Ok(r_native.clone())).unwrap();

        enforce_freivalds(&a, &b, &c, &r).unwrap();

        println!(
            "number of constraints (freivalds 5x7 * 7x3): {}",
            cs.num_constraints()
        );
        assert!(cs.is_satisfied().unwrap());
    }
}
