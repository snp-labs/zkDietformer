use ark_ec::pairing::Pairing;
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::Zero;
use ark_std::marker::PhantomData;
use ark_std::ops::Mul;
use ark_std::vec;
use ark_std::vec::Vec;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

/// CoeffPos: A struct to help build sparse matrices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoeffPos<T> {
    pub(crate) val: T,     // value of the coefficient
    pub(crate) pos: usize, // position of the coefficient in the column
}

// a column is a vector of CoeffPos-s
pub(crate) type Col<T> = Vec<CoeffPos<T>>;

/// Column-Major Sparse Matrix
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseMatrix<T> {
    pub(crate) cols: Vec<Col<T>>, // a vector of columns
    pub nr: usize,     // number of rows
    pub nc: usize,     // number of columns
}

impl<T: Copy> SparseMatrix<T> {
    // NB: Given column by column
    // create a new SparseMatrix with given number of rows and columns
    pub fn new(nr: usize, nc: usize) -> SparseMatrix<T> {
        SparseMatrix {
            cols: vec![vec![]; nc], // create an empty vector of columns
            nr,
            nc,
        }
    }

    // insert a value in the specified row and column
    pub fn insert_val(&mut self, r: usize, c: usize, v: &T) {
        let coeff_pos = CoeffPos { pos: r, val: *v };
        self.cols[c].push(coeff_pos); // add the coefficient to the column vector
    }

    // insert a slice of values in the specified row starting from the specified column offset
    pub fn insert_row_slice(&mut self, r: usize, c_offset: usize, vs: &Vec<T>) {
        for (i, x) in vs.iter().enumerate() {
            self.insert_val(r, c_offset + i, x); // insert each value in the row
        }
    }

    // get the column vector at the specified column index
    pub fn get_col(&self, c: usize) -> &Col<T> {
        &self.cols[c]
    }
}

pub struct SparseLinAlgebra<E: Pairing> {
    pairing_engine_type: PhantomData<E>, // used for holding a reference to E
}

impl<E: Pairing> SparseLinAlgebra<E> {
    /// Inner product of a sparse column with a dense scalar vector, computed
    /// via batched MSM (Pippenger's algorithm).
    ///
    /// Replaces the previous O(n) individual `g.mul(scalar)` + `add_assign`
    /// loop with a single `msm_unchecked` call, yielding an O(n / log n)
    /// speedup for large columns.
    pub fn sparse_inner_product(v: &Vec<E::ScalarField>, w: &Col<E::G1Affine>) -> E::G1Affine {
        // Fast path: empty column → identity (zero point).
        if w.is_empty() {
            return E::G1::zero().into_affine();
        }

        // Collect bases and scalars for a single batched MSM.
        let mut bases: Vec<E::G1Affine> = Vec::with_capacity(w.len());
        let mut scalars: Vec<E::ScalarField> = Vec::with_capacity(w.len());
        for cp in w {
            bases.push(cp.val);
            scalars.push(v[cp.pos]);
        }

        // Pippenger MSM: O(n / log n) group ops instead of O(n).
        E::G1::msm_unchecked(&bases, &scalars).into_affine()
    }

    pub fn sparse_vector_matrix_mult(
        v: &Vec<E::ScalarField>,
        m: &SparseMatrix<E::G1Affine>,
        _t: usize,
    ) -> Vec<E::G1Affine> {
        #[cfg(feature = "multicore")]
        {
            m.cols
                .par_iter()
                .map(|col| Self::sparse_inner_product(v, col))
                .collect()
        }
        #[cfg(not(feature = "multicore"))]
        {
            m.cols
                .iter()
                .map(|col| Self::sparse_inner_product(v, col))
                .collect()
        }
    }
}

/// MSM-based inner product between a scalar vector and a G1 vector.
pub fn inner_product<E: Pairing>(v: &[E::ScalarField], w: &[E::G1Affine]) -> E::G1Affine {
    assert_eq!(v.len(), w.len());
    E::G1::msm_unchecked(w, v).into_affine()
}

/// Scale each element of `v` by scalar `a`.  Parallelized when `multicore`
/// is enabled.
pub fn scalar_vector_mult<E: Pairing>(
    a: &E::ScalarField,
    v: &[E::ScalarField],
    _l: usize,
) -> Vec<E::ScalarField> {
    #[cfg(feature = "multicore")]
    { v.par_iter().map(|vi| a.mul(vi)).collect() }
    #[cfg(not(feature = "multicore"))]
    { v.iter().map(|vi| a.mul(vi)).collect() }
}
