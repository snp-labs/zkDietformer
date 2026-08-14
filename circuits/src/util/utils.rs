use crate::cc_snark::legogroth16::*;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM, pairing::Pairing};
use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::Variable;
use ark_serialize::CanonicalSerialize;
use ark_std::{ops::AddAssign, vec::Vec, UniformRand, rand::Rng};
use sha3::{Sha3_256, Digest, digest::Input};

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

/// Compute `cm3 = proof.d3` from committed witnesses (Group 3 / Aux) and blinding `v3`, using vk's commitment bases.
/// `public_inputs_count` is the number of public inputs.
pub fn cm_aux_from_vk<E: Pairing>(
    vk: &VerifyingKey<E>,
    public_inputs_count: usize,
    committed_aux: &[E::ScalarField],
    v3: &E::ScalarField,
) -> E::G1Affine
where
    E::ScalarField: PrimeField,
{
    let scalars = committed_aux
        .iter()
        .map(|s| s.into_bigint())
        .collect::<Vec<_>>();

    // Group 3 bases start after Public Inputs + Group 1 + Group 2
    let group1_count = vk.commit_witness_count_1 as usize;
    let group2_count = vk.commit_witness_count_2 as usize;

    let start = 1 + public_inputs_count + group1_count + group2_count;
    let end = start + scalars.len();

    if end > vk.gamma_abc_g1.len() {
        panic!("cm_aux_from_vk: Not enough bases in vk.gamma_abc_g1 for Group 3 (Aux)");
    }

    let bases = &vk.gamma_abc_g1[start..end];

    let mut d3 = E::G1::msm_bigint(bases, &scalars);
    d3.add_assign(vk.eta3_gamma_inv_g1.mul_bigint(v3.into_bigint()));
    d3.into_affine()
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

/// Onlyeq Fiat-Shamir challenge: `r = H("zkdiet-onlyeq-fs" || cm1 || cm_aux)`.
/// Used by onlyeq circuits that have d1 + d3 but no d2/c_W/Poseidon.
pub fn hash_cm_with_aux_to_field<E: Pairing>(cm1: &E::G1Affine, cm_aux: &E::G1Affine) -> E::ScalarField
where
    E::ScalarField: PrimeField,
{
    let mut hasher = Sha3_256::new();
    <Sha3_256 as Input>::input(&mut hasher, b"zkdiet-onlyeq-fs");

    let mut buf = Vec::new();
    cm1.serialize_compressed(&mut buf).expect("serialize cm1");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    cm_aux.serialize_compressed(&mut buf).expect("serialize cm_aux");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    E::ScalarField::from_le_bytes_mod_order(&hasher.result())
}

/// Generate a fresh global commitment key of length `n+1`: `[h_glob, g_0, ..., g_{n-1}]`.
/// Used to commit model weights with `c_W = MSM(g_i, w_i) + o * h_glob`.
pub fn generate_global_ck<E: Pairing, R: Rng>(n: usize, rng: &mut R) -> Vec<E::G1Affine> {
    let projective: Vec<E::G1> = (0..n + 1).map(|_| E::G1::rand(rng)).collect();
    E::G1::normalize_batch(&projective)
}

/// Compute the global weight commitment: `c_W = MSM(ck_glob[1..], values) + o * ck_glob[0]`.
pub fn commit_global<E: Pairing>(
    ck_glob: &[E::G1Affine],
    values: &[E::ScalarField],
    o: &E::ScalarField,
) -> E::G1Affine
where
    E::ScalarField: PrimeField,
{
    let scalars: Vec<_> = values.iter().map(|s| s.into_bigint()).collect();
    let bases = &ck_glob[1..1 + scalars.len()];
    let mut cm = E::G1::msm_bigint(bases, &scalars);
    cm.add_assign(ck_glob[0].mul_bigint(o.into_bigint()));
    cm.into_affine()
}

/// Multi-commitment Fiat-Shamir challenge (v2).
/// `r = H("zkdiet-fs-v2" || c_W || cm1 || cm2 || input_hash || output_hash)`.
pub fn hash_commitments_to_field<E: Pairing>(
    c_w: &E::G1Affine,
    cm_lk: &E::G1Affine,
    cm_w: &E::G1Affine,
    input_hash: &E::ScalarField,
    output_hash: &E::ScalarField,
) -> E::ScalarField
where
    E::ScalarField: PrimeField,
{
    let mut hasher = Sha3_256::new();
    <Sha3_256 as Input>::input(&mut hasher, b"zkdiet-fs-v2");

    let mut buf = Vec::new();
    c_w.serialize_compressed(&mut buf).expect("serialize c_w");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    cm_lk.serialize_compressed(&mut buf).expect("serialize cm_lk");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    cm_w.serialize_compressed(&mut buf).expect("serialize cm_w");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    input_hash.serialize_compressed(&mut buf).expect("serialize input_hash");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    output_hash.serialize_compressed(&mut buf).expect("serialize output_hash");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    let digest = hasher.result();
    E::ScalarField::from_le_bytes_mod_order(&digest)
}

/// Multi-commitment Fiat-Shamir challenge (v3) — includes cm_aux for Freivalds binding.
/// `r = H("zkdiet-fs-v3" || c_W || cm1 || cm2 || cm_aux || input_hash || output_hash)`.
/// cm_aux binds Freivalds-constrained intermediates (Q, K, V, Z*, Y) separately from
/// the global weight commitment c_W, preserving c_W's reusability across proofs.
pub fn hash_commitments_to_field_v3<E: Pairing>(
    c_w: &E::G1Affine,
    cm_lk: &E::G1Affine,
    cm_w: &E::G1Affine,
    cm_aux: &E::G1Affine,
    input_hash: &E::ScalarField,
    output_hash: &E::ScalarField,
) -> E::ScalarField
where
    E::ScalarField: PrimeField,
{
    let mut hasher = Sha3_256::new();
    <Sha3_256 as Input>::input(&mut hasher, b"zkdiet-fs-v3");

    let mut buf = Vec::new();
    c_w.serialize_compressed(&mut buf).expect("serialize c_w");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    cm_lk.serialize_compressed(&mut buf).expect("serialize cm_lk");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    cm_w.serialize_compressed(&mut buf).expect("serialize cm_w");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    cm_aux.serialize_compressed(&mut buf).expect("serialize cm_aux");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    input_hash.serialize_compressed(&mut buf).expect("serialize input_hash");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    buf.clear();
    output_hash.serialize_compressed(&mut buf).expect("serialize output_hash");
    <Sha3_256 as Input>::input(&mut hasher, &buf);

    let digest = hasher.result();
    E::ScalarField::from_le_bytes_mod_order(&digest)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Bn254, Fr, G1Projective};
    use ark_std::{UniformRand, rand::{SeedableRng, rngs::StdRng}};

    type E = Bn254;

    /// Test B: FS hash (v3) changes when cm_aux changes, even if all other inputs are identical.
    /// This validates that Freivalds-constrained intermediates are bound before r derivation.
    #[test]
    fn test_fs_hash_v3_reflects_aux_commitment() {
        let mut rng = StdRng::seed_from_u64(42);

        // Fixed commitments (c_W, cm1, cm2, hashes)
        let c_w = <E as Pairing>::G1::rand(&mut rng).into_affine();
        let cm1 = <E as Pairing>::G1::rand(&mut rng).into_affine();
        let cm2 = <E as Pairing>::G1::rand(&mut rng).into_affine();
        let input_hash = Fr::rand(&mut rng);
        let output_hash = Fr::rand(&mut rng);

        // Two different aux commitments (simulating honest vs tampered intermediates)
        let cm_aux_honest = <E as Pairing>::G1::rand(&mut rng).into_affine();
        let cm_aux_tampered = <E as Pairing>::G1::rand(&mut rng).into_affine();

        let r1 = hash_commitments_to_field_v3::<E>(
            &c_w, &cm1, &cm2, &cm_aux_honest, &input_hash, &output_hash,
        );
        let r2 = hash_commitments_to_field_v3::<E>(
            &c_w, &cm1, &cm2, &cm_aux_tampered, &input_hash, &output_hash,
        );

        assert_ne!(r1, r2, "FS challenge must change when cm_aux changes");
    }

    /// Verify that cm_aux is computed independently from c_W.
    /// Changing aux values doesn't affect c_W (preserving global reusability).
    #[test]
    fn test_cm_aux_independent_from_cw() {
        let mut rng = StdRng::seed_from_u64(42);

        // Shared weights
        let weights: Vec<Fr> = (0..10).map(|_| Fr::rand(&mut rng)).collect();
        let ck_w = generate_global_ck::<E, _>(weights.len(), &mut rng);
        let o_w = Fr::rand(&mut rng);
        let c_w = commit_global::<E>(&ck_w, &weights, &o_w);

        // Two different aux vectors
        let aux1: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        let aux2: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();

        let ck_aux = generate_global_ck::<E, _>(5, &mut rng);
        let o_aux = Fr::rand(&mut rng);
        let cm_aux1 = commit_global::<E>(&ck_aux, &aux1, &o_aux);
        let cm_aux2 = commit_global::<E>(&ck_aux, &aux2, &o_aux);

        // c_W unchanged regardless of aux
        let c_w2 = commit_global::<E>(&ck_w, &weights, &o_w);
        assert_eq!(c_w, c_w2, "c_W must be unchanged by different aux");

        // But cm_aux differs
        assert_ne!(cm_aux1, cm_aux2, "cm_aux must differ for different intermediates");
    }

    /// Verify v2 and v3 hash produce different challenges (domain separation).
    #[test]
    fn test_fs_v2_v3_domain_separation() {
        let mut rng = StdRng::seed_from_u64(42);

        let c_w = <E as Pairing>::G1::rand(&mut rng).into_affine();
        let cm1 = <E as Pairing>::G1::rand(&mut rng).into_affine();
        let cm2 = <E as Pairing>::G1::rand(&mut rng).into_affine();
        let cm_aux = <E as Pairing>::G1::rand(&mut rng).into_affine();
        let input_hash = Fr::rand(&mut rng);
        let output_hash = Fr::rand(&mut rng);

        let r_v2 = hash_commitments_to_field::<E>(
            &c_w, &cm1, &cm2, &input_hash, &output_hash,
        );
        let r_v3 = hash_commitments_to_field_v3::<E>(
            &c_w, &cm1, &cm2, &cm_aux, &input_hash, &output_hash,
        );

        assert_ne!(r_v2, r_v3, "v2 and v3 must produce different challenges (domain separation)");
    }
}