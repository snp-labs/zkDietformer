use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ff::Zero;
use ark_std::marker::PhantomData;
use ark_std::ops::{AddAssign, Mul};
use ark_std::vec;
use ark_std::vec::Vec;

/// CoeffPos: A struct to help build sparse matrices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoeffPos<T> {
    val: T,     // value of the coefficient
    pos: usize, // position of the coefficient in the column
}

// a column is a vector of CoeffPos-s
type Col<T> = Vec<CoeffPos<T>>;

/// Column-Major Sparse Matrix
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseMatrix<T> {
    cols: Vec<Col<T>>, // a vector of columns, each column is represented by a vector of CoeffPos-s
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
    // compute the sparse inner product of two vectors: v and w
    pub fn sparse_inner_product(v: &Vec<E::ScalarField>, w: &Col<E::G1Affine>) -> E::G1Affine {
        let mut res: E::G1 = E::G1::zero(); // initialize the result as the group's identity element
        for coeffpos in w {
            let g = coeffpos.val;
            let i = coeffpos.pos;
            // multiply the coefficient with the corresponding element in v and add to the result
            let tmp = g.mul(v[i]);
            res.add_assign(&tmp);
        }
        res.into_affine() // convert the result back to affine form
    }

    // compute the sparse vector-matrix multiplication of a vector v and a sparse matrix m, return the result as a vector
    pub fn sparse_vector_matrix_mult(
        v: &Vec<E::ScalarField>,
        m: &SparseMatrix<E::G1Affine>,
        t: usize,
    ) -> Vec<E::G1Affine> {
        // the result should contain every column of m multiplied by v
        let mut res: Vec<E::G1Affine> = Vec::with_capacity(t);
        for c in 0..m.nc {
            res.push(Self::sparse_inner_product(&v, &m.get_col(c)));
        }
        res
    }
}

pub fn inner_product<E: Pairing>(v: &[E::ScalarField], w: &[E::G1Affine]) -> E::G1Affine {
    // Take two slices of `ScalarField` and `G1Affine` elements and compute their inner product
    assert_eq!(v.len(), w.len());
    let mut res: E::G1 = E::G1::zero(); // Initialize a variable to hold the result of the inner product
    for i in 0..v.len() {
        let tmp = w[i].mul(v[i]); // Multiply the i-th element of `v` and `w` and store it in a temporary variable
        res.add_assign(&tmp); // Add the result of the multiplication to `res`
    }
    res.into_affine() // Convert `res` to an affine point and return it
}

pub fn scalar_vector_mult<E: Pairing>(
    a: &E::ScalarField,
    v: &[E::ScalarField],
    l: usize,
) -> Vec<E::ScalarField> {
    // Multiply each element of the `v` slice by a scalar field element `a`
    let mut res: Vec<E::ScalarField> = Vec::with_capacity(l); // Create a new vector to hold the result of the multiplication
    for i in 0..v.len() {
        let x: E::ScalarField = a.mul(&v[i]); // Multiply the i-th element of `v` by `a` and store it in a new variable `x`
        res.push(x); // Add `x` to the result vector
    }
    res // Return the result vector
}
