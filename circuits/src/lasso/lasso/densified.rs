use ark_ec::CurveGroup;
use ark_ff::PrimeField;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

use crate::lasso::poly::dense_mlpoly::DensePolynomial;
use super::surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialCommitment};

pub struct DensifiedRepresentation<F: PrimeField, const C: usize> {
  pub dim_usize: [Vec<usize>; C],
  pub dim: [DensePolynomial<F>; C],
  pub read: [DensePolynomial<F>; C],
  pub r#final: [DensePolynomial<F>; C],
  pub combined_l_variate_polys: DensePolynomial<F>,
  pub combined_log_m_variate_polys: DensePolynomial<F>,
  pub s: usize, // sparsity
  pub log_m: usize,
  pub m: usize,
}

impl<F: PrimeField, const C: usize> From<&SparseLookupMatrix<C>> for DensifiedRepresentation<F, C> {
  #[tracing::instrument(skip_all, name = "Densify")]
  fn from(sparse: &SparseLookupMatrix<C>) -> Self {
    // Each dimension is independent: reads from sparse.nz[*][i] and builds
    // its own dim/read/final vectors.  Parallelize across dimensions.
    let process_dim = |i: usize| -> (Vec<usize>, DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>) {
      let mut access_sequence = sparse
        .nz
        .iter()
        .map(|indices| indices[i])
        .collect::<Vec<usize>>();
      access_sequence.resize(sparse.s, 0usize);

      let mut final_timestamps = vec![0usize; sparse.m];
      let mut read_timestamps = vec![0usize; sparse.s];

      for j in 0..sparse.s {
        let memory_address = access_sequence[j];
        debug_assert!(memory_address < sparse.m);
        let ts = final_timestamps[memory_address];
        read_timestamps[j] = ts;
        final_timestamps[memory_address] = ts + 1;
      }

      let dim_poly = DensePolynomial::from_usize(&access_sequence);
      let read_poly = DensePolynomial::from_usize(&read_timestamps);
      let final_poly = DensePolynomial::from_usize(&final_timestamps);
      (access_sequence, dim_poly, read_poly, final_poly)
    };

    #[cfg(feature = "multicore")]
    let results: Vec<_> = (0..C).into_par_iter().map(process_dim).collect();
    #[cfg(not(feature = "multicore"))]
    let results: Vec<_> = (0..C).map(process_dim).collect();

    let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(C);
    let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    let mut read: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    let mut r#final: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    for (acc, d, r_poly, f) in results {
      dim_usize.push(acc);
      dim.push(d);
      read.push(r_poly);
      r#final.push(f);
    }

    let l_variate_polys = [dim.as_slice(), read.as_slice()].concat();

    let combined_l_variate_polys = DensePolynomial::merge(&l_variate_polys);
    let combined_log_m_variate_polys = DensePolynomial::merge(&r#final);

    DensifiedRepresentation {
      dim_usize: dim_usize.try_into().unwrap(),
      dim: dim.try_into().unwrap(),
      read: read.try_into().unwrap(),
      r#final: r#final.try_into().unwrap(),
      combined_l_variate_polys,
      combined_log_m_variate_polys,
      s: sparse.s,
      log_m: sparse.log_m,
      m: sparse.m,
    }
  }
}

impl<F: PrimeField, const C: usize> DensifiedRepresentation<F, C> {
  #[tracing::instrument(skip_all, name = "DensifiedRepresentation.commit")]
  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    gens: &SparsePolyCommitmentGens<G>,
  ) -> SparsePolynomialCommitment<G> {
    // Two independent polynomial commitments — run in parallel when multicore.
    #[cfg(feature = "multicore")]
    let (l_variate_polys_commitment, log_m_variate_polys_commitment) = {
      let (l_res, log_m_res) = rayon::join(
        || self.combined_l_variate_polys.commit(&gens.gens_combined_l_variate, None),
        || self.combined_log_m_variate_polys.commit(&gens.gens_combined_log_m_variate, None),
      );
      (l_res.0, log_m_res.0)
    };
    #[cfg(not(feature = "multicore"))]
    let (l_variate_polys_commitment, log_m_variate_polys_commitment) = {
      let (l, _) = self.combined_l_variate_polys.commit(&gens.gens_combined_l_variate, None);
      let (m, _) = self.combined_log_m_variate_polys.commit(&gens.gens_combined_log_m_variate, None);
      (l, m)
    };

    SparsePolynomialCommitment {
      l_variate_polys_commitment,
      log_m_variate_polys_commitment,
      s: self.s,
      log_m: self.log_m,
      m: self.m,
    }
  }
}
