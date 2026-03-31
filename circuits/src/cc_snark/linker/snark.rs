use std::marker::PhantomData;

use crate::cc_snark::linker::{
    matrix::{scalar_vector_mult, SparseLinAlgebra, SparseMatrix},
    // relation_generator::generate_cp_relation,
    relation_generator::*,
    Linker,
};

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::Rng, One, UniformRand, Zero};
use std::ops::Mul;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

#[derive(Clone, Default, PartialEq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PP<
    G1: Clone + Default + CanonicalSerialize + CanonicalDeserialize,
    G2: Clone + Default + CanonicalSerialize + CanonicalDeserialize,
> {
    pub l: usize, // # of rows
    pub t: usize, // # of cols
    pub g1: G1,
    pub g2: G2,
}

impl<
        G1: Clone + Default + CanonicalSerialize + CanonicalDeserialize,
        G2: Clone + Default + CanonicalSerialize + CanonicalDeserialize,
    > PP<G1, G2>
{
    pub fn new(l: usize, t: usize, g1: &G1, g2: &G2) -> PP<G1, G2> {
        PP {
            l,
            t,
            g1: g1.clone(),
            g2: g2.clone(),
        }
    }
}

#[derive(Clone, Default, PartialEq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct EK<G1: Clone + Default + CanonicalSerialize + CanonicalDeserialize> {
    pub p: Vec<G1>,
}

#[derive(Clone, Default, PartialEq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VK<G2: Clone + Default + CanonicalSerialize + CanonicalDeserialize> {
    pub c: Vec<G2>,
    pub a: G2,
}

/// Convert a scalar vector to G2 points by multiplying each scalar with
/// `pp.g2`.  Parallelized with rayon + batch affine normalization.
fn vec_to_g2<E: Pairing>(
    pp: &PP<E::G1Affine, E::G2Affine>,
    v: &Vec<E::ScalarField>,
) -> Vec<E::G2Affine> {
    #[cfg(feature = "multicore")]
    let projective: Vec<E::G2> = v.par_iter().map(|x| pp.g2.mul(*x)).collect();
    #[cfg(not(feature = "multicore"))]
    let projective: Vec<E::G2> = v.iter().map(|x| pp.g2.mul(*x)).collect();

    // Batch affine conversion: single Montgomery inversion chain.
    E::G2::normalize_batch(&projective)
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct LinkSnark<E: Pairing> {
    _curve: PhantomData<E>,
}

impl<E: Pairing> Linker<E> for LinkSnark<E> {
    type Instance = Vec<E::G1Affine>;
    type Witness = Vec<E::ScalarField>;

    type CM = PhantomData<E>;
    type EK = EK<E::G1Affine>;
    type VK = VK<E::G2Affine>;

    type CRS = SparseMatrix<E::G1Affine>;
    type PP = PP<E::G1Affine, E::G2Affine>;
    type Proof = E::G1Affine;

    fn setup<R: Rng>(
        num: usize,
        ck: Vec<E::G1Affine>,
        snark_ck: Vec<E::G1Affine>,
        mode: &str,
        rng: &mut R,
    ) -> (Self::PP, Self::CRS) {
        let g1 = E::G1::rand(rng).into_affine();
        let g2 = E::G2::rand(rng).into_affine();

        let link_crs = match mode {
            "1toN_Vector" => {
                let n_indiv = num; 
                
                let mut cks_indiv = Vec::with_capacity(n_indiv);
                let mut current_ck = Vec::new();
                for point in snark_ck {
                    if point.is_zero() {
                        if !current_ck.is_empty() {
                            cks_indiv.push(current_ck.clone());
                            current_ck.clear();
                        }
                    } else {
                        current_ck.push(point);
                    }
                }
                if !current_ck.is_empty() {
                    cks_indiv.push(current_ck);
                }
                
                assert_eq!(cks_indiv.len(), n_indiv, "구분자로 분리된 키 개수가 num과 일치하지 않습니다.");
                let msg_lens: Vec<usize> = cks_indiv.iter().map(|k| k.len() - 1).collect();
                generate_cp_1_to_n_vector_relation::<E>(n_indiv, msg_lens, ck, cks_indiv)
            },
            "arithm" => generate_cp_arithm_relation::<E>(num, ck, snark_ck),
            "bound" => generate_cp_bound_relation::<E>(num, ck, snark_ck),
            "ctt" => generate_cp_ctt_relation::<E>(num, ck, snark_ck),
            "wt" => generate_cp_wt_relation::<E>(num, ck, snark_ck),
            _ => SparseMatrix::new(0, 0),
        };

        let l = link_crs.nr;
        let t = link_crs.nc;

        let mut link_pp = PP::<E::G1Affine, E::G2Affine> { l, t, g1, g2 };
        (link_pp, link_crs)
    }

    fn keygen<R: Rng>(pp: &Self::PP, crs: Self::CRS, rng: &mut R) -> (Self::EK, Self::VK) {
        // Create an empty vector to hold k values
        let mut k = Vec::with_capacity(pp.l);
        // Generate random k values and append them to the k vector
        for _ in 0..pp.l {
            k.push(E::ScalarField::rand(rng));
        }

        // Generate a random scalar field value a
        let a = E::ScalarField::rand(rng);
        // Calculate the vector p as the result of multiplying the sparse matrix m with k
        let p = SparseLinAlgebra::<E>::sparse_vector_matrix_mult(&k, &crs, pp.t);
        // Calculate the vector c as the result of multiplying a with each element in k
        let c = scalar_vector_mult::<E>(&a, &k, pp.l);
        // Create an EK struct with the calculated vector p
        let ek = EK::<E::G1Affine> { p };
        // Convert the vector c to G2 and create a VK struct with the converted vector c and the value (pp.g2 * a)
        let vk = VK::<E::G2Affine> {
            c: vec_to_g2::<E>(pp, &c),
            a: (pp.g2 * a).into_affine(),
        };
        // Return the EK and VK structs as a tuple
        (ek, vk)
    }

    fn prove<R: Rng>(
        pp: &Self::PP,
        ek: &Self::EK,
        witness: Self::Witness,
        rng: &mut R,
    ) -> (Self::Proof, Self::CM) {
        (
            Self::inner_product(ek.p.clone(), witness.to_vec()),
            PhantomData,
        )
    }

    fn verify(pp: &Self::PP, vk: &Self::VK, instance: &Self::Instance, prf: &Self::Proof) -> bool {
        assert_eq!(pp.l, instance.len());
        let mut g1 = vec![];
        let mut g2 = vec![];
        for i in 0..instance.len() {
            g1.push(E::G1Prepared::from(instance[i]));
            g2.push(E::G2Prepared::from(vk.c[i]));
        }
        g1.push(E::G1Prepared::from(*prf));
        g2.push(E::G2Prepared::from(-vk.a.into_group()));
        E::TargetField::one() == E::multi_pairing(g1.into_iter(), g2.into_iter()).0
    }

    fn generate_witness(
        r: Vec<E::ScalarField>,
        u: Vec<E::ScalarField>,
        snark_witness: Vec<E::ScalarField>,
    ) -> <LinkSnark<E> as Linker<E>>::Witness {
        let witness = generate_cp_witness::<E>(r, u, snark_witness).unwrap();

        witness
    }

    fn generate_instance(
        cm: Vec<E::G1Affine>,
        snark_cm: E::G1Affine,
        aux_cm: <LinkSnark<E> as Linker<E>>::CM,
    ) -> <LinkSnark<E> as Linker<E>>::Instance {
        let instance = generate_cp_instance::<E>(cm, snark_cm).unwrap();

        instance
    }
}

impl<E: Pairing> LinkSnark<E> {
    /// Inner product via batched MSM (Pippenger).
    fn inner_product(w: Vec<E::G1Affine>, v: Vec<E::ScalarField>) -> E::G1Affine {
        assert_eq!(v.len(), w.len());
        E::G1::msm_unchecked(&w, &v).into_affine()
    }
}
