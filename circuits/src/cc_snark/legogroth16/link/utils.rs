//! Utils for matrix and vector operations

use ark_ec::{
    AffineRepr, CurveGroup, VariableBaseMSM, pairing::Pairing,
};
use ark_std::{
    marker::PhantomData,
    ops::Mul,
    vec,
    vec::Vec,
};

#[cfg(feature = "multicore")]
use rayon::prelude::*;

use crate::cc_snark::legogroth16::link::error::LinkError;

/// CoeffPos: A struct to help build sparse matrices.
#[derive(Clone, Debug)]
pub struct CoeffPos<T> {
    val: T,
    pos: usize,
}

// a column is a vector of CoeffPos-s
type Col<T> = Vec<CoeffPos<T>>;

/* TODO: One could consider a cache-friendlier implementation for the 2-row case*/

/// Column-Major Sparse Matrix
#[derive(Clone, Debug)]
pub struct SparseMatrix<T> {
    cols: Vec<Col<T>>, // a vector of columns
    pub nr: usize,     // no. of rows
    pub nc: usize,     // no. of columns
}

impl<T: Copy> SparseMatrix<T> {
    // NB: Given column by column
    pub fn new(nr: usize, nc: usize) -> SparseMatrix<T> {
        SparseMatrix {
            cols: vec![vec![]; nc],
            nr,
            nc,
        }
    }

    /// Insert value `v` in the column index `c` at row index `r`
    pub fn insert_val(&mut self, r: usize, c: usize, v: T) -> Result<(), LinkError> {
        if self.cols.len() < c {
            return Err(LinkError::InvalidIndex(c, self.cols.len()));
        }
        let coeff_pos = CoeffPos { pos: r, val: v };
        self.cols[c].push(coeff_pos);
        Ok(())
    }

    /// insert a continuous sequence of values at row r starting from c_offset
    pub fn insert_row_slice(
        &mut self,
        r: usize,
        c_offset: usize,
        vs: Vec<T>,
    ) -> Result<(), LinkError> {
        // NB: could be improved in efficiency by first extending the vector
        for (i, x) in vs.into_iter().enumerate() {
            self.insert_val(r, c_offset + i, x)?;
        }
        Ok(())
    }

    pub fn get_col(&self, c: usize) -> Result<&Col<T>, LinkError> {
        if self.cols.len() < c {
            return Err(LinkError::InvalidIndex(c, self.cols.len()));
        }
        Ok(&self.cols[c])
    }
}

pub struct SparseLinAlgebra<PE: Pairing> {
    pairing_engine_type: PhantomData<PE>,
}

impl<PE: Pairing> SparseLinAlgebra<PE> {

    pub fn sparse_inner_product(
        v: &Vec<PE::ScalarField>,
        w: &Col<PE::G1Affine>,
    ) -> Result<PE::G1Affine, LinkError> {
        // Fast path: empty column → identity (zero point).
        if w.is_empty() {
            return Ok(PE::G1Affine::zero());
        }

        // Collect bases and scalars for a single batched MSM.
        let mut bases: Vec<PE::G1Affine> = Vec::with_capacity(w.len());
        let mut scalars: Vec<PE::ScalarField> = Vec::with_capacity(w.len());
        for cp in w {
            if v.len() < cp.pos {
                return Err(LinkError::InvalidIndex(cp.pos, v.len()));
            }
            bases.push(cp.val);
            scalars.push(v[cp.pos]);
        }

        // Pippenger MSM: O(n / log n) group ops instead of O(n).
        Ok(PE::G1::msm_unchecked(&bases, &scalars).into_affine())
    }

    pub fn sparse_vector_matrix_mult(
        v: &Vec<PE::ScalarField>,
        m: &SparseMatrix<PE::G1Affine>,
    ) -> Result<Vec<PE::G1Affine>, LinkError> {
        #[cfg(feature = "multicore")]
        {
            // Parallel: process all columns concurrently.
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

/// MSM between a scalar vector and a G1 vector
pub fn inner_product<PE: Pairing>(a: &[PE::ScalarField], b: &[PE::G1Affine]) -> PE::G1Affine {
    PE::G1::msm_unchecked(b, &a).into_affine()
}

pub fn scale_vector<PE: Pairing>(
    a: &PE::ScalarField,
    v: &[PE::ScalarField],
) -> Vec<PE::ScalarField> {
    #[cfg(feature = "multicore")]
    { v.par_iter().map(|vi| a.mul(vi)).collect() }
    #[cfg(not(feature = "multicore"))]
    { v.iter().map(|vi| a.mul(vi)).collect() }
}


pub fn multiples_of_g<G: AffineRepr>(g: &G, multiples: &[G::ScalarField]) -> Vec<G> {
    #[cfg(feature = "multicore")]
    let muls: Vec<G::Group> = multiples
        .par_iter()
        .map(|s| g.mul(*s))
        .collect();
    #[cfg(not(feature = "multicore"))]
    let muls: Vec<G::Group> = multiples
        .iter()
        .map(|s| g.mul(*s))
        .collect();

    G::Group::normalize_batch(&muls)
}