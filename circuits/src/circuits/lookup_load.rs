use anyhow::{Context, Result};
use ark_ec::pairing::Pairing;
use ark_ff::{PrimeField, Zero};
use ark_std::{rand::{rngs::StdRng, SeedableRng}, UniformRand, str::FromStr};
use num_bigint::BigInt;
use std::ops::Mul;
use crate::{
    circuits::lookup_pr_return::{create_lookup_proof, CPLinkInfo},
    harisa::{type_conversion::bigint_to_fr, Membership, harisa::Harisa},
    linker::snark::{LinkSnark},
};
use ark_ec::CurveGroup;
use crate::linker::Linker;

pub struct LookupLinkData<E: Pairing, M, LNK>
where
    M: Membership<E, LNK>,
    LNK: Linker<E>, 
{
    pub cpinfo: CPLinkInfo<E, M, LNK>,
    pub link_instance: Vec<E::G1Affine>,
    pub link_witness: LNK::Witness,
    pub prod_fields: Vec<E::ScalarField>,
    pub link_rng: StdRng,
}

pub fn prepare_lookup_link_data<E: Pairing>(
    lookup_npz_path: &str,
    combined_matmul_ints: Vec<Vec<i128>>,
    prod_bigints_flat: Vec<BigInt>,
    k_segments: usize,
    seq_len: usize,
) -> Result<LookupLinkData<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>> 
where
    <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
    let rows = combined_matmul_ints.len();
    let lookup_batch_size = rows * seq_len * k_segments;

    // 1. Harisa Lookup Proof 생성
    let cpinfo = create_lookup_proof::<E>(lookup_npz_path, combined_matmul_ints, lookup_batch_size)
        .map_err(|e| anyhow::anyhow!("create_lookup_proof failed: {:?}", e))?;

    // 길이 체크 : prod_bigints_flat과 cpinfo.f_bigints 길이 비교(둘이 같은 값이어야 함)
    let mut mismatches = 0usize;
    if cpinfo.f_bigints.len() != prod_bigints_flat.len() {
        println!("길이 불일치: cpinfo.f_bigints={} prod_bigints_flat={}",
                cpinfo.f_bigints.len(), prod_bigints_flat.len());
    } else {
        for (idx, (a, b)) in prod_bigints_flat.iter().zip(cpinfo.f_bigints.iter()).enumerate() {
            if a != b {
                println!("값 불일치 인덱스 {}: zkdiet={} lookup_pr={}", idx, a, b);
                mismatches += 1;
            }
        }
        if mismatches == 0 {
            println!("모든 BigInt 값이 일치합니다 (순서 동일 가정).");
        } else {
            println!("불일치 총 {}개 발견", mismatches);
        }
    }

    // 2. 필드 요소 변환 및 검증
    let prod_fields: Vec<E::ScalarField> = prod_bigints_flat
        .iter()
        .map(|b| bigint_to_fr(b.clone()))
        .collect();

    // 3. CP-Link를 위한 로컬 커밋(cm_prod) 생성
    let mut link_rng = StdRng::seed_from_u64(43u64);
    let o_prod = E::ScalarField::rand(&mut link_rng);

    let mut cm_prod_proj = E::G1::zero();
    cm_prod_proj += cpinfo.pp.ck[0].mul(o_prod);
    for (g_i, u_i) in cpinfo.pp.ck.iter().skip(1).zip(prod_fields.iter()) {
        cm_prod_proj += g_i.mul(*u_i);
    }
    let cm_prod_affine = cm_prod_proj.into_affine();

    // 4. Link Witness 준비
    let mut r_vec = Vec::new();
    r_vec.push(o_prod);      
    r_vec.push(cpinfo.o_f); 

    let link_witness = LinkSnark::<E>::generate_witness(
        r_vec, 
        prod_fields.clone(), // 증명하고자 하는 공통 값 u
        vec![] 
    );

    // 5. Link Instance 준비
    let mut link_instance = Vec::new();
    link_instance.push(cm_prod_affine);
    link_instance.push(cpinfo.cm_f);

    Ok(LookupLinkData {
        cpinfo,
        link_instance,
        link_witness,
        prod_fields,
        link_rng
    })
}