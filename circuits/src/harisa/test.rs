use std::{ops::Bound, str::FromStr};

use crate::{
    harisa::{arithm::ArithmCircuit, harisa::Harisa, type_conversion::bigint_to_fr},
    linker::Linker,
    ConstraintF,
};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::{Field, PrimeField};
use ark_r1cs_std::pairing::PairingVar;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_std::{
    rand::{Rng, RngCore, SeedableRng},
    test_rng, One, UniformRand,
};
use num_bigint::BigInt;

use super::cc_gro::prepare_verifying_key;

const SET_SIZE: usize = 32;

fn test_harisa<E: Pairing, LNK: Linker<E>>(set: Vec<BigInt>, l_size: usize)
where
    <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    LNK: Clone,
{
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    // u commit
    let mut u = Vec::new();

    for i in 0..l_size {
        u.push(set[i].clone());
    }

    let arithm_circuit = ArithmCircuit::<E::ScalarField>::mock(l_size);

    // setup
    let (pp, tree, _) =
        Harisa::<E, LNK>::generate_harisa_parameters(set.clone(), arithm_circuit, &mut rng)
            .unwrap();

    let accum = tree[0].clone().modpow(&set[0].clone(), &pp.mod_n.clone());

    let mut u_scalar: Vec<E::ScalarField> = Vec::new();
    for u_i in u.clone() {
        u_scalar.push(bigint_to_fr(u_i));
    }

    let o_u = E::ScalarField::rand(&mut rng);
    let mut cm_u = (pp.ck[0].clone() * o_u).into();

    for (g_i, u_i) in pp
        .ck
        .clone()
        .iter()
        .skip(1)
        .zip(u_scalar.clone().into_iter())
    {
        cm_u = (cm_u + *g_i * u_i).into();
    }

    // prove
    let proof = Harisa::<E, LNK>::generate_harisa_opt_proof(
        pp.clone(),
        tree,
        accum.clone(),
        cm_u.clone(),
        u.clone(),
        o_u.clone(),
        &mut rng,
        true,
    )
    .unwrap();

    // verify
    assert!(
        Harisa::<E, LNK>::harisa_verify(pp, accum, cm_u.clone(), proof).unwrap(),
        "[Harisa] Verify Failed"
    );
}

fn set(n: usize) -> Vec<BigInt> {
    use crate::harisa::constants::ODD_PRIME;

    let mut res = Vec::new();
    for i in 0..n {
        res.push(BigInt::from(ODD_PRIME[i]));
    }

    res
}

#[test]
fn test_harisa_bn254() {
    use crate::linker::snark::LinkSnark;
    use ark_bn254::{Bn254 as E, Fr as F};
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    let set = set(256);

    test_harisa::<E, LinkSnark<E>>(set, SET_SIZE);
}
