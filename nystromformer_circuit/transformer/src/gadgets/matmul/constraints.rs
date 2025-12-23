use ark_ff::PrimeField;
use ark_r1cs_std::{alloc::AllocVar, fields::fp::FpVar};
use ark_relations::r1cs::ConstraintSystemRef;

use crate::gadgets::dot_prod::constraints::*;

/// Row-major matrix: [r][c]
pub type MatrixVar<F> = Vec<Vec<FpVar<F>>>;
pub type MatrixConst<F> = Vec<Vec<F>>;

fn new_witness<F: PrimeField>(cs: ConstraintSystemRef<F>, m: &[Vec<F>]) -> MatrixVar<F> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_r1cs_std::{R1CSVar, alloc::AllocVar};
    use ark_relations::r1cs::{ConstraintSystem, ConstraintSystemRef};
    use ark_std::Zero;

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

    #[test]
    fn test_matmul_basic() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // A: 1 x 4
        // B: 4 x 2
        // C: 1 x 2
        let a_vals = vec![vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
        ]];
        let b_vals = vec![
            vec![Fr::from(5u64), Fr::from(6u64)],
            vec![Fr::from(7u64), Fr::from(8u64)],
            vec![Fr::from(9u64), Fr::from(10u64)],
            vec![Fr::from(11u64), Fr::from(12u64)],
        ];

        let a = new_witness::<Fr>(cs.clone(), &a_vals);
        let b = new_witness::<Fr>(cs.clone(), &b_vals);

        let c = matmul(&a, &b);
        let expected = native_matmul(&a_vals, &b_vals);

        assert_eq!(c.len(), expected.len());
        for i in 0..c.len() {
            assert_eq!(c[i].len(), expected[i].len());
            for j in 0..c[i].len() {
                assert_eq!(c[i][j].value().unwrap(), expected[i][j]);
            }
        }

        println!("number of constraints: {}", cs.num_constraints());
        assert!(cs.is_satisfied().unwrap());
    }
}
