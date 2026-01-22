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
    util::{load::load_tables, table::lookup_setgen},
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
    pub cm_f_prime: <E as Pairing>::G1Affine,
    pub cm_f_hat: <E as Pairing>::G1Affine,
    pub cm_f: <E as Pairing>::G1Affine,
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
    // let (raw_tables, raw_prime, raw_z, _frac_bits, limb_bits) = load_tables(npz_path)?;
    let (raw_tables, _frac_bits, limb_bits) = load_tables(npz_path)?;

    let k_segments = raw_tables.len();
    if k_segments == 0 {
        return Err("No table segments found".into());
    }
    let table_width = raw_tables[0].len();
    if table_width == 0 {
        return Err("Table width is zero".into());
    }

    // 2) flat_table 구성 (row-major: segment0[..], segment1[..], ...)
    let mut flat_table: Vec<BigInt> = Vec::with_capacity(k_segments * table_width);
    for k in 0..k_segments {
        for idx in 0..table_width {
            flat_table.push(BigInt::from(raw_tables[k][idx]));
        }
    }

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

    // print!("u_bigints: {:?}", u_bigints);

    // 5) prime/z 테이블 생성 (lookup_setgen)
    let (mut prime_table, mut z_table) = lookup_setgen(u_bigints.clone());

    // safety: prime_table.len() == u_bigints.len()
    if prime_table.len() != u_bigints.len() || z_table.len() != u_bigints.len() {
        return Err("prime_table/z_table length mismatch with u_bigints".into());
    }

    // 6) build vec_table WITH original indices, sort 및 proven/non_proven 분할
    let mut vec_table: Vec<(BigInt, BigInt, BigInt, usize)> = Vec::with_capacity(prime_table.len());
    for i in 0..prime_table.len() {
        vec_table.push((prime_table[i].clone(), u_bigints[i].clone(), z_table[i].clone(), i));
    }
    // sort by prime (first component)
    vec_table.sort_by(|a, b| a.0.cmp(&b.0));

    let mut proven_indices: Vec<usize> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        proven_indices.push(vec_table[i].3);
    }

    use std::collections::HashSet;
    let proven_set: HashSet<usize> = proven_indices.iter().cloned().collect();

    // Build proven arrays in ORIGINAL (flat/u_bigints) order by iterating u_bigints indices
    let mut hat_f: Vec<BigInt> = Vec::new();
    let mut t: Vec<BigInt> = Vec::new();
    let mut z_f: Vec<BigInt> = Vec::new();

    // create quick maps from original index -> hat_prime, t_val, z
    let mut hat_map: std::collections::HashMap<usize, BigInt> = std::collections::HashMap::new();
    let mut t_map: std::collections::HashMap<usize, BigInt> = std::collections::HashMap::new();
    let mut z_map: std::collections::HashMap<usize, BigInt> = std::collections::HashMap::new();
    for i in 0..vec_table.len() {
        let (ref p, ref u, ref zval, idx) = vec_table[i];
        // store mapping for this original index: prime p corresponds to u at idx, zval at idx
        hat_map.insert(idx, p.clone());
        t_map.insert(idx, u.clone());
        z_map.insert(idx, zval.clone());
    }

    for orig_idx in 0..u_bigints.len() {
        if proven_set.contains(&orig_idx) {
            // push entries in the original flat order for proven indices
            hat_f.push(hat_map.get(&orig_idx).ok_or_else(|| "hat_map missing index")?.clone());
            t.push(t_map.get(&orig_idx).ok_or_else(|| "t_map missing index")?.clone());
            z_f.push(z_map.get(&orig_idx).ok_or_else(|| "z_map missing index")?.clone());
        }
    }

    // Build non_proven by taking remaining vec_table entries (their primes/z)
    let mut non_proven_elem: Vec<BigInt> = Vec::new();
    let mut non_proven_z: Vec<BigInt> = Vec::new();
    for i in batch_size..vec_table.len() {
        let (ref p, _, ref zval, idx) = vec_table[i];
        non_proven_elem.push(p.clone());
        non_proven_z.push(zval.clone());
    }

    // remainder of original code but now hat_f, t, z_f are in ORIGINAL order (flat order) for proven elements
    let base_2: i32 = 2;
    let expo = 14;

    let mut bit_shift = BigInt::from(base_2.pow(expo));

    assert!(!hat_f.is_empty(), "hat_f must not be empty after selection");
    assert_eq!(
        hat_f[0].clone(),
        t[0].clone() * bit_shift + z_f[0].clone(),
        "Sorting/selection inconsistency"
    );

    let mut proven_elem: Vec<BigInt> = [hat_f.clone(), z_f.clone()].concat();

    let mut prod_prime_table: BigInt = BigInt::one();
    for i in 0..hat_f.len() {
        prod_prime_table *= hat_f[i].clone();
        prod_prime_table *= z_f[i].clone();
    }

    non_proven_elem.extend(non_proven_z.clone());
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
    let o_f_hat = E::ScalarField::rand(&mut rng);
    let mut cm_f_hat = (pp.ck[0].clone() * o_f_hat).into();
    for (g_i, u_i) in pp
        .ck
        .clone()
        .iter()
        .skip(1)
        .zip(f_hat_scalar.clone().into_iter())
    {
        cm_f_hat = (cm_f_hat + *g_i * u_i).into();
    }

    // f (t / original table values) -- note: t is now in original(flat) order for proven items
    let mut f_scalar: Vec<E::ScalarField> = Vec::new();
    for f_i in t.clone() {
        f_scalar.push(bigint_to_fr(f_i));
    }
    let o_f = E::ScalarField::rand(&mut rng);
    let mut cm_f = (pp.ck[0].clone() * o_f).into();
    for (g_i, u_i) in pp
        .ck
        .clone()
        .iter()
        .skip(1)
        .zip(f_scalar.clone().into_iter())
    {
        cm_f = (cm_f + *g_i * u_i).into();
    }

    // z (randomizers)
    let mut z_scalar: Vec<E::ScalarField> = Vec::new();
    for u_i in z_f.clone() {
        z_scalar.push(bigint_to_fr(u_i));
    }
    let o_z = E::ScalarField::rand(&mut rng);
    let mut cm_z = (pp.ck[0].clone() * o_z).into();
    for (g_i, u_i) in pp
        .ck
        .clone()
        .iter()
        .skip(1)
        .zip(z_scalar.clone().into_iter())
    {
        cm_z = (cm_z + *g_i * u_i).into();
    }

    // harisa element (concatenate lookup and rand)
    let harisa_elem = [hat_f.clone(), z_f.clone()].concat();
    let harisa_scalar = [f_hat_scalar.clone(), z_scalar.clone()].concat();

    // f_prime commitment (on harisa_scalar)
    let o_f_prime = E::ScalarField::rand(&mut rng);
    let mut cm_f_prime = (pp.ck[0].clone() * o_f_prime).into();
    for (g_i, u_i) in pp
        .ck
        .clone()
        .iter()
        .skip(1)
        .zip(harisa_scalar.clone().into_iter())
    {
        cm_f_prime = (cm_f_prime + *g_i * u_i).into();
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

    // 13) CPLinkInfo 조립 — 반환되는 f_bigints/f_fields는 proven t 값들을 "원래(flat) 순서"로 정렬한 버전
    // zkdiet에서 만든 prod_bigints_flat 순서와 일치
    let f_bigints = t.clone(); // proven values, in original(flat) order
    let f_fields = f_scalar.clone(); // already converted to field in same order

    let f_hat_fields = f_hat_scalar.clone();
    let z_fields = z_scalar.clone();
    // extract wt opening from proof (ensure LookupProof has wt_prf.open)
    let wt_opening = proof.wt_prf.open;

    let cpinfo = CPLinkInfo {
        cm_f_prime,
        cm_f_hat,
        cm_f,
        cm_z,
        pp: pp.clone(),
        o_f_prime,
        o_f_hat,
        o_f,
        o_z,
        accum: accum.clone(),
        lookup_proof: proof,
        f_bigints,
        f_fields,
        f_hat_fields,
        z_fields,
        wt_opening,
    };

    Ok(cpinfo)
}

fn main() {
    use ark_bn254::Bn254;

    let npz_path = "tmp_tables/softmax_tables_quantized.npz";

    let matmul_ints: Vec<Vec<i128>> = vec![vec![1i128, 1i128, 1i128, 1i128]];

    // 테이블 로드하기
    let (raw_tables, _frac_bits, limb_bits) = load_tables(npz_path)
        .expect("failed to load npz in test (ensure tmp_tables/softmax_tables_quantized.npz exists)");

    let k_segments = raw_tables.len();
    assert!(k_segments > 0, "k_segments must be > 0");

    let rows = matmul_ints.len();
    let cols = if rows > 0 { matmul_ints[0].len() } else { 0 };

    let batch_size = rows * cols * k_segments;

    println!("Test info: rows={}, cols={}, k_segments={}, batch_size={}", rows, cols, k_segments, batch_size);
    // 룩업 증명 생성
    let cpinfo = create_lookup_proof::<Bn254>(npz_path, matmul_ints.clone(), batch_size)
        .expect("create_lookup_proof failed");

    println!("create_lookup_proof completed successfully; received CPLinkInfo");
}