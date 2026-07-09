use crate::cc_snark::legogroth16::*;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM, pairing::Pairing};
use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::Variable;
use ark_serialize::CanonicalSerialize;
use ark_std::{ops::AddAssign, vec::Vec};
use sha3::{Sha3_256,Digest, digest::Input};

/// Compute `cm1 = proof.d1` from committed witnesses (Group 1) and blinding `v1`, using vk's commitment bases.
/// `public_inputs_count` is the number of public inputs in the circuit.
pub fn cm1_from_vk<E: Pairing>(
    vk: &VerifyingKey<E>,
    public_inputs_count: usize,
    committed_group1: &[E::ScalarField],
    v1: &E::ScalarField,
) -> E::G1Affine
where
    E::ScalarField: PrimeField,
{
    let scalars = committed_group1
        .iter()
        .map(|s| s.into_bigint())
        .collect::<Vec<_>>();

    // Group 1 bases start immediately after public inputs (index 0 is '1', so start at 1 + public_inputs_count)
    let start = 1 + public_inputs_count;
    let end = start + scalars.len();

    // Safety check for bounds
    if end > vk.gamma_abc_g1.len() {
        panic!("cm1_from_vk: Not enough bases in vk.gamma_abc_g1 for Group 1");
    }

    let bases = &vk.gamma_abc_g1[start..end];

    let mut d1 = E::G1::msm_bigint(bases, &scalars);
    d1.add_assign(vk.eta1_gamma_inv_g1.mul_bigint(v1.into_bigint()));
    d1.into_affine()
}

/// Compute `cm2 = proof.d2` from committed witnesses (Group 2) and blinding `v2`, using vk's commitment bases.
/// `public_inputs_count` is the number of public inputs.
/// `group1_count` is the number of witnesses in Group 1 (needed to find the offset for Group 2).
pub fn cm2_from_vk<E: Pairing>(
    vk: &VerifyingKey<E>,
    public_inputs_count: usize,
    committed_group2: &[E::ScalarField],
    v2: &E::ScalarField,
) -> E::G1Affine
where
    E::ScalarField: PrimeField,
{
    let scalars = committed_group2
        .iter()
        .map(|s| s.into_bigint())
        .collect::<Vec<_>>();

    // Group 2 bases start after Public Inputs AND Group 1 witnesses.
    // We use vk.commit_witness_count_1 directly if accessible, otherwise pass it as argument.
    // Assuming we use the field from VK:
    let group1_count = vk.commit_witness_count_1 as usize;

    let start = 1 + public_inputs_count + group1_count;
    let end = start + scalars.len();

    // Safety check for bounds
    if end > vk.gamma_abc_g1.len() {
        panic!("cm2_from_vk: Not enough bases in vk.gamma_abc_g1 for Group 2");
    }

    let bases = &vk.gamma_abc_g1[start..end];

    let mut d2 = E::G1::msm_bigint(bases, &scalars);
    d2.add_assign(vk.eta2_gamma_inv_g1.mul_bigint(v2.into_bigint()));
    d2.into_affine()
}

/// Derive r_vec of length `len` from cm.
/// r[i] = SHA3-256( "freivalds-r-v1/expand" || seed || i ) mapped into the scalar field,
/// where seed = SHA3-256( "freivalds-r-v1" || serialize(cm) ).
pub fn hash_cm_to_field_vec<E: Pairing>(cm: &E::G1Affine, len: usize) -> Vec<E::ScalarField>
where
    E::ScalarField: PrimeField,
{
    // 1) serialize(cm)
    let mut cm_bytes = Vec::new();
    cm.serialize_compressed(&mut cm_bytes)
        .expect("serialize cm");

    // 2) seed = H("freivalds-r-v1" || cm_bytes)
    let mut h0 = Sha3_256::new();
    <Sha3_256 as Input>::input(&mut h0, b"freivalds-r-v1");
    <Sha3_256 as Input>::input(&mut h0, &cm_bytes);
    let seed = h0.result(); // 32 bytes

    // 3) expand: r[i] = MapToField( H(".../expand" || seed || i_le_u32) )
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let mut hi = Sha3_256::new();
        <Sha3_256 as Input>::input(&mut hi, b"freivalds-r-v1/expand");
        <Sha3_256 as Input>::input(&mut hi, &seed);
        <Sha3_256 as Input>::input(&mut hi, (i as u32).to_le_bytes());
        let digest = hi.result();
        out.push(E::ScalarField::from_le_bytes_mod_order(&digest));
    }

    out
}

// challenge 값 생성용
pub fn hash_cm_to_field<E: Pairing>(cm: &E::G1Affine) -> E::ScalarField
where
    E::ScalarField: PrimeField,
{
    let mut cm_bytes = Vec::new();
    cm.serialize_compressed(&mut cm_bytes)
        .expect("serialize cm");

    let mut hasher = Sha3_256::new();
    <Sha3_256 as Input>::input(&mut hasher, b"freivalds-r-v1");
    <Sha3_256 as Input>::input(&mut hasher, &cm_bytes);
    let seed = hasher.result(); // hash 결과값

    E::ScalarField::from_le_bytes_mod_order(&seed)
}

/// Extract witness indices from committed FpVar slices.
/// Each FpVar must be an allocated Witness variable.
pub fn extract_witness_indices<F: PrimeField>(vars: &[FpVar<F>]) -> Vec<u32> {
    vars.iter().map(|v| match v {
        FpVar::Var(alloc) => match alloc.variable {
            Variable::Witness(idx) => idx as u32,
            other => panic!("Expected Witness variable, got {:?}", other),
        },
        _ => panic!("Expected allocated FpVar, got Constant"),
    }).collect()
}