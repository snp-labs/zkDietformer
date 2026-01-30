use std::{ops::Bound, str::FromStr, path::Path};
use std::error::Error;

use ndarray::{Array2, Array0};
use ndarray_npy::NpzReader;
use std::fs::File;
use ark_ff::UniformRand;
use num_bigint::BigInt;
use ark_bn254::{Bn254, Fr as F};
use ark_std::{test_rng, rand::SeedableRng, One, Zero};
use rand::RngCore;
use ark_ec::pairing::Pairing;

use crate::harisa::Membership;
use crate::linker::Linker;
use crate::harisa::precompute::rsa_setup;
use crate::{
    harisa::{
        harisa::Harisa,
        arithm::ArithmCircuit,
        type_conversion::bigint_to_fr,
        hash_to_prime::primality_test,
    },
    linker::snark::LinkSnark,
    lookup::{
        copy_this_or_that::CTTCircuit,
        well_transformed::WTCircuit,
        lookup::HarisaPlus,
        data_structure::{LookupPP, LookupProof},
    },
    util::{load::load_tables, table::lookup_setgen_from_precomputed},
};

// cm을 만들었던 value (세그먼트별로 회로 입력으로 들어가는 스칼라들)
pub struct LookupSegment<E: Pairing> {
    pub harisa_scalar: Vec<<E as Pairing>::ScalarField>,
    pub f_hat_scalar: Vec<<E as Pairing>::ScalarField>,
    pub f_scalar: Vec<<E as Pairing>::ScalarField>,
    pub z_scalar: Vec<<E as Pairing>::ScalarField>,
}

// cp 링크 정보: 룩업 증명 생성 시 필요한 커밋(들), 커밋 키(pp), 오프닝 값들, accum, 그리고 실제 룩업 증명 객체
pub struct CPLinkInfo<E: Pairing, M, LNK>
where
    M: Membership<E, LNK>,
    LNK: Linker<E>,
{
    pub cm_f_prime: <E as Pairing>::G1Affine, // harisa_scalar = [f_hat_scalar.clone(), z_scalar.clone()]
    pub cm_f_hat: <E as Pairing>::G1Affine, // f_hat_scalar
    pub cm_f: <E as Pairing>::G1Affine, // f_scalar
    pub cm_z: <E as Pairing>::G1Affine,
    // 이안에 ck가 있음
    pub pp: LookupPP<E, M, LNK>,

    pub o_f_prime: <E as Pairing>::ScalarField,
    pub o_f_hat: <E as Pairing>::ScalarField,
    pub o_f: <E as Pairing>::ScalarField,
    pub o_z: <E as Pairing>::ScalarField,

     // 검증을 밖에서 하도록 끌어냄
    pub accum: BigInt,

    // 검증을 밖에서 하도록 끌어냄
    pub lookup_proof: LookupProof<E, M, LNK>,

    // 실질적 비교할 스칼라 값 (원래(flat) 순서에 맞춘 proven 'f' 값들)
    pub f_bigints: Vec<BigInt>,
    pub f_fields: Vec<<E as Pairing>::ScalarField>,

    // 추가: 링크 증명 생성을 위해 외부에서 재사용할 필드 (이미 field로 변환되어 있음)
    // f_hat와 z 스칼라 (create_lookup_proof에서 만든 E::ScalarField 벡터)
    pub f_hat_fields: Vec<<E as Pairing>::ScalarField>,
    pub z_fields: Vec<<E as Pairing>::ScalarField>,

    // wt 관련 SNARK opening (wt_proof.open) — LinkSnark의 snark_witness 로 사용
    pub wt_opening: <E as Pairing>::ScalarField,
    pub non_proven_elem: Vec<BigInt>,
}

pub fn create_lookup_proof<E: Pairing>(
    npz_path: &str,
    matmul_ints: Vec<Vec<i128>>,
    batch_size: usize,
) -> Result<CPLinkInfo<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>, Box<dyn Error>>
where
    <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
{
    // RNG
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(test_rng().next_u64());

    // 1) 테이블 로드
    let (raw_tables,raw_primes, raw_z, _frac_bits, limb_bits) = load_tables(npz_path)?;
    // let (raw_tables, _frac_bits, limb_bits) = load_tables(npz_path)?;

    let k_segments = raw_tables.len();
    if k_segments == 0 {
        return Err("No table segments found".into());
    }
    let table_width = raw_tables[0].len();
    if table_width == 0 {
        return Err("Table width is zero".into());
    }

    // 2) flat_table 구성 (row-major: segment0[..], segment1[..], ...)
    let flat_table: Vec<BigInt> = raw_tables.into_iter().flatten().collect();
    let flat_prime: Vec<BigInt> = raw_primes.into_iter().flatten().collect();
    let flat_z: Vec<BigInt> = raw_z.into_iter().flatten().collect();

    // 3) u_bigints: matmul_ints -> limb 인덱스로 찾아 flat_table 항목 수집
    let mut u_bigints: Vec<BigInt> = Vec::new();
    let limb_mask: u128 = if limb_bits >= 128 { !0u128 } else { (1u128 << limb_bits) - 1u128 };
    for i in 0..matmul_ints.len() {
        for j in 0..matmul_ints[i].len() {
            let val = matmul_ints[i][j];
            if val < 0 {
                return Err(format!("Negative matmul input not supported: ({},{}) = {}", i, j, val).into());
            }
            let unsigned = val as u128;
            for k in 0..k_segments {
                let limb_val = ((unsigned >> (k * limb_bits)) & limb_mask) as usize;
                let flat_idx = k * table_width + limb_val;
                // safety check
                if flat_idx >= flat_table.len() {
                    return Err(format!("Computed flat index out of range: {}", flat_idx).into());
                }
                u_bigints.push(flat_table[flat_idx].clone());
            }
        }
    }

    // 4) sanity check for batch_size
    if batch_size == 0 {
        return Err("batch_size must be > 0".into());
    }
    if batch_size > u_bigints.len() {
        return Err(format!(
            "batch_size ({}) is larger than number of referenced table entries (u_bigints.len() = {}) — \
            typically batch_size should be <= rows*cols*k_segments",
            batch_size,
            u_bigints.len()
        ).into());
    }

    // 5) prime/z 테이블 생성 (lookup_setgen)
    let (mut prime_table, mut z_table) = lookup_setgen_from_precomputed(u_bigints.clone(), &flat_table, &flat_prime, &flat_z);

    // safety: prime_table.len() == u_bigints.len()
    if prime_table.len() != u_bigints.len() || z_table.len() != u_bigints.len() {
        return Err("prime_table/z_table length mismatch with u_bigints".into());
    }

    // 6) build vec_table WITH original indices, sort 및 proven/non_proven 분할
    // include original index so we can restore original ordering after selecting proven entries
    let mut vec_table: Vec<(BigInt, BigInt, BigInt, usize)> = Vec::with_capacity(prime_table.len());
    for i in 0..prime_table.len() {
        vec_table.push((prime_table[i].clone(), u_bigints[i].clone(), z_table[i].clone(), i));
    }
    // sort by prime (first component) (stable)
    vec_table.sort_by(|a, b| a.0.cmp(&b.0));

    // Build proven arrays by selecting the smallest `batch_size` primes, but then restore them
    // into the ORIGINAL flat/u_bigints order.
    let mut selected_map: std::collections::HashMap<usize, (BigInt, BigInt, BigInt)> = std::collections::HashMap::new();
    for i in 0..batch_size {
        let (p, u, z, orig_idx) = vec_table[i].clone();
        selected_map.insert(orig_idx, (p, u, z));
    }

    // Build hat_f, t, z_f in original order by iterating original indices
    let mut hat_f: Vec<BigInt> = Vec::new();
    let mut t: Vec<BigInt> = Vec::new();
    let mut z_f: Vec<BigInt> = Vec::new();

    for orig_idx in 0..u_bigints.len() {
        if let Some((p, u, z)) = selected_map.get(&orig_idx) {
            hat_f.push(p.clone());
            t.push(u.clone());
            z_f.push(z.clone());
        }
    }

    // remainder: non_proven built from remaining vec_table entries (their primes/z)
    let mut proven_elem: Vec<BigInt> = [hat_f.clone(), z_f.clone()].concat();
    let mut non_proven_elem: Vec<BigInt> = Vec::new();
    let mut non_proven_z: Vec<BigInt> = Vec::new();
    for i in batch_size..u_bigints.len() {
        let (ref p, _, ref zval, _) = vec_table[i];
        non_proven_elem.push(p.clone());
        non_proven_z.push(zval.clone());
    }

    // sanity checks used by original code
    let base_2: i32 = 2;
    let expo = 14;
    let mut bit_shift = BigInt::from(base_2.pow(expo));

    assert!(!hat_f.is_empty(), "hat_f must not be empty after selection");
    assert_eq!(
        hat_f[0].clone(),
        t[0].clone() * bit_shift + z_f[0].clone(),
        "Sorting/selection inconsistency"
    );

    let mut prod_prime_table: BigInt = BigInt::one();
    for i in 0..batch_size {
        prod_prime_table *= hat_f[i].clone();
        prod_prime_table *= z_f[i].clone();
    }

    println!("전: {}", non_proven_elem.len());
    non_proven_elem.extend(non_proven_z.clone());
    println!("후: {}", non_proven_elem.len());
    let prod_non_proven: BigInt = non_proven_elem.clone().iter().product();

    let mut set_table = [proven_elem.clone(), non_proven_elem.clone()].concat();
    let prod_set_table: BigInt = set_table.clone().iter().product();

    assert_eq!(
        prod_set_table.clone(),
        prod_non_proven.clone() * prod_prime_table.clone(),
        "Exponentiation Failed"
    );

    let arithm_circuit = ArithmCircuit::<E::ScalarField>::mock(2 * batch_size);
    let ctt_circuit = CTTCircuit::<E::ScalarField>::mock(2 * batch_size, 2 * batch_size);
    let wt_circuit = WTCircuit::<E::ScalarField>::mock(batch_size, batch_size, batch_size);

    // 8) generate Harisa parameters & tree (this can be expensive)
    let (pp, tree) = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::generate_lookup_parameters(
        set_table.clone(),
        ctt_circuit,
        wt_circuit,
        arithm_circuit,
        &mut rng,
    ).unwrap();

    let mut prod_set: BigInt = set_table.clone().iter().product();

    // 9) accum as in test.rs
    println!("DEBUG [Return]: prod_set_table (first 10 digits) = {}", &prod_set_table.to_string()[..10]);
    println!("DEBUG [Return]: mod_n (first 10 digits) = {}", &pp.m_pp.mod_n.to_string()[..10]);
    let accum = tree[0].clone().modpow(&set_table[0].clone(), &pp.m_pp.mod_n);
    
    // 10) Prepare circuit inputs (convert BigInt -> Field)
    let mut circuit_hat_f: Vec<E::ScalarField> = Vec::new();
    let mut circuit_t: Vec<E::ScalarField> = Vec::new();
    let mut circuit_z_f: Vec<E::ScalarField> = Vec::new();

    for hat_f_i in hat_f.clone() {
        circuit_hat_f.push(bigint_to_fr(hat_f_i.clone()));
    }

    for t_i in t.clone() {
        circuit_t.push(bigint_to_fr(t_i.clone()));
    }

    for z_i in z_f.clone() {
        circuit_z_f.push(bigint_to_fr(z_i.clone()));
    }

    // 11) build circuit instances used for the Harisa proof
    let ctt_elem = [circuit_hat_f.clone(), circuit_z_f.clone()].concat();
    let ctt_circuit = CTTCircuit::<E::ScalarField>::new(ctt_elem.clone(), ctt_elem.clone());
    let wt_circuit = WTCircuit::<E::ScalarField>::new(circuit_hat_f.clone(), circuit_t.clone(), circuit_z_f.clone());

    // f_hat (hat_f)
    let mut f_hat_scalar: Vec<E::ScalarField> = Vec::new();
    for u_i in hat_f.clone() {
        f_hat_scalar.push(bigint_to_fr(u_i));
    }

    // f (t / original table values) -- note: t is now in original(flat) order for proven items
    let mut f_scalar: Vec<E::ScalarField> = Vec::new();
    for f_i in t.clone() {
        f_scalar.push(bigint_to_fr(f_i));
    }

    // z (randomizers)
    let mut z_scalar: Vec<E::ScalarField> = Vec::new();
    for u_i in z_f.clone() {
        z_scalar.push(bigint_to_fr(u_i));
    }



    // 12) generate lookup proof (prove)
    let proof = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::generate_lookup_proof(
        pp.clone(),
        accum.clone(),
        tree.clone(),
        hat_f.clone(),
        t.clone(),
        z_f.clone(),
        ctt_circuit,
        wt_circuit,
        &mut rng,
        non_proven_elem.clone(),
    )?;

    println!("DEBUG set_table.len = {}", set_table.len());
    println!("DEBUG set_table head = {:?}", &set_table[..std::cmp::min(8, set_table.len())]);
    println!("DEBUG set_table tail = {:?}", &set_table[set_table.len().saturating_sub(std::cmp::min(8, set_table.len()))..]);
    println!("DEBUG accum = {}", accum);
    println!("DEBUG tree0 = {}", tree[0]);
    println!("DEBUG mod_n = {}", pp.m_pp.mod_n);
    println!("DEBUG cm_f_prime = {:?}", proof.cm_f_prime);
    println!("DEBUG cm_f_hat = {:?}", proof.cm_f_hat);
    println!("DEBUG cm_f = {:?}", proof.cm_f);
    println!("DEBUG cm_z = {:?}", proof.cm_z);
    println!("DEBUG openings o_f_prime/o_f_hat/o_f/o_z = {}/{}/{}/{}", proof.o_f_prime, proof.o_f_hat, proof.o_f, proof.o_z);

    
    // 5. Harisa Lookup 최종 검증
    let flag = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::verify_lookup(
        pp.clone(),
        accum.clone(),
        proof.cm_f_hat.clone(),
        proof.cm_f.clone(),
        proof.cm_z.clone(),
        proof.clone(),
    )?;
    println!("Harisa Lookup Verification: {}", flag);

    if !flag {
        return Err(anyhow::anyhow!("Harisa lookup proof failed verification").into());
    }

    // 13) CPLinkInfo 조립 — 반환되는 f_bigints/f_fields는 proven t 값들을 "원래(flat) 순서"로 정렬한 버전
    let f_bigints = t.clone(); // proven values, in original(flat) order
    let f_fields = f_scalar.clone(); // already converted to field in same order

    let f_hat_fields = f_hat_scalar.clone();
    let z_fields = z_scalar.clone();
    // extract wt opening from proof (ensure LookupProof has wt_prf.open)
    let wt_opening = proof.wt_prf.open;

    let cpinfo = CPLinkInfo {
        cm_f_prime: proof.cm_f_prime,
        cm_f_hat: proof.cm_f_hat,
        cm_f: proof.cm_f,
        cm_z: proof.cm_z,
        pp: pp.clone(),
        o_f_prime: proof.o_f_prime,
        o_f_hat: proof.o_f_hat,
        o_f: proof.o_f,
        o_z: proof.o_z,
        accum: accum.clone(),
        lookup_proof: proof,
        f_bigints,
        f_fields,
        f_hat_fields,
        z_fields,
        wt_opening,
        non_proven_elem,
    };
    println!("--- PoKE prover debug ---");
    println!("pp.mod_n = {}", cpinfo.pp.m_pp.mod_n.to_string()); // 또는 pp.mod_n 위치
    println!("prover computed prod_set_table = {}", prod_set_table.to_string());
    println!("prover accum (tree[0]^prod_set mod n) = {}", accum.to_string());
    
    
    //
    Ok(cpinfo)
}