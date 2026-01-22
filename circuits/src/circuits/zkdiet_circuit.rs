use std::{fs::File, ops::{AddAssign, Mul},path::Path, str::FromStr};
use anyhow::{Context, Result};
use ark_ec::{CurveGroup, pairing::Pairing};
use ndarray::{ArrayD, IxDyn, Ix4, Array2, Ix2, IxDynImpl, arr0};
use ndarray_npy::{NpzReader, NpzWriter};
use ark_std::{marker::PhantomData, rand::{rngs::StdRng, SeedableRng, Rng}, UniformRand};
use ark_bn254::{Bn254, Fr as F};
use ark_ff::{PrimeField, One, Zero};
use ark_relations::r1cs::{ConstraintSystemRef, ConstraintSynthesizer, SynthesisError};
use ark_r1cs_std::{alloc::AllocVar, eq::EqGadget, fields::fp::FpVar};
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use num_bigint::BigInt;
use crate::{
    cc_snark::cc_gro::{CcGroth16, prepare_verifying_key},
    circuits::{
        freivalds_circuit::FreivaldsCircuit, freivalds_erase_circuit::FreivaldsMatMulCircuit, lookup_pr_return::{CPLinkInfo, create_lookup_proof}, 
        softmax_circuit::{SoftmaxCircuit, K_SEGMENTS}, // 파이썬 측 상수와 값을 맞춰주어야 함
    },
    gadgets::{
        matmul::constraints::{ MatrixVar, mat_vec_mul, new_matrix_witness},
        softmax::constraints::build_softmax_witnesses,
    }, 
    harisa::{type_conversion::bigint_to_fr, harisa::Harisa},
    linker::{snark::LinkSnark, Linker}, lookup::lookup::HarisaPlus, util::load::{read_array_u64, read_array_i64}, 
};

/// 모든 서브 서킷을 하나로 합친 통합 서킷
#[derive(Clone)]
pub struct ZKDietCircuit {
    pub qkfreivalds: FreivaldsCircuit<F>,
    pub softmax: SoftmaxCircuit<F>,
    pub qkvfreivalds: FreivaldsMatMulCircuit<F>,
}

impl ConstraintSynthesizer<F> for ZKDietCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        self.qkfreivalds.clone().generate_constraints(cs.clone())?;
        self.softmax.clone().generate_constraints(cs.clone())?;
        self.qkvfreivalds.clone().generate_constraints(cs.clone())?;
        Ok(())
    }
}

/// npz파일을 로드하고, 필요한 값들을 준비하는 함수
pub fn load_and_build_combined<E: Pairing>(path: &Path) -> Result<ZKDietCircuit> 
where
    <<E as Pairing>::ScalarField as FromStr>::Err: core::fmt::Debug,
    {
    let f = File::open(path).context("opening integrated npz")?;
    let mut npz = NpzReader::new(f)?;

    // npz에 저장된 값들이 음수가 없으므로 unsigned 64비트로 읽어옴
    let a_arr = read_array_u64(&mut npz, "a").context("reading 'a'")?;
    let b_arr = read_array_u64(&mut npz, "b").context("reading 'b'")?;
    let c_arr = read_array_u64(&mut npz, "c").context("reading 'c'")?;
    let div_arr = read_array_u64(&mut npz, "div").context("reading 'div'")?;
    let rem_arr = read_array_u64(&mut npz, "rem").context("reading 'rem'")?;
    // 서킷에 넣기 위해 4차원으로 강제함
    let a4 = a_arr.into_dimensionality::<Ix4>().context("'a' must be 4D")?;
    let b4 = b_arr.into_dimensionality::<Ix4>().context("'b' must be 4D")?;
    let c4 = c_arr.into_dimensionality::<Ix4>().context("'c' must be 4D")?;
    let div4 = div_arr.into_dimensionality::<Ix4>().context("'div' must be 4D")?;
    let rem4 = rem_arr.into_dimensionality::<Ix4>().context("'rem' must be 4D")?;

    let shape = a4.shape();
    let (batch, num_head, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);

    // 가변 배열을 위해 ndarrays를 nested Vec<Vec<Vec<Vec<F>>>> 형태로 변환
    let mut a_f = vec![vec![vec![vec![F::zero(); head_dim]; seq_len]; num_head]; batch];
    let mut b_f = vec![vec![vec![vec![F::zero(); seq_len]; head_dim]; num_head]; batch];
    let mut c_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch];
    let mut div_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch];
    let mut rem_f = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch];
    // 채워넣기
    for ((n,h,i,k), val) in a4.indexed_iter() {
        a_f[n][h][i][k] = F::from(*val as u64);
    }
    for ((n,h,k,i), val) in b4.indexed_iter() {
        b_f[n][h][k][i] = F::from(*val as u64);
    }
    for ((n,h,i,j), val) in c4.indexed_iter() {
        c_f[n][h][i][j] = F::from(*val as u64);
    }
    for ((n,h,i,j), val) in div4.indexed_iter() {
        div_f[n][h][i][j] = F::from(*val as u64);
    }
    for ((n,h,i,j), val) in rem4.indexed_iter() {
        rem_f[n][h][i][j] = F::from(*val as u64);
    }

    // Scalars: r, input_0, c_0_converted, multiplier
    let r_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("r.npy")?;
    let r = F::from(r_arr.iter().next().copied().unwrap_or(0u64));
    let input0_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("input_0.npy")?;
    let input_0 = F::from(input0_arr.iter().next().copied().unwrap_or(0u64));
    let c0_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("c_0_converted.npy")?;
    let c0 = F::from(c0_arr.iter().next().copied().unwrap_or(0u64));
    let mult_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("multiplier.npy")?;
    let multiplier = F::from(mult_arr.iter().next().copied().unwrap_or(1u64));

    // 프리발드 서킷 생성
    let qkfreivalds = FreivaldsCircuit {
        a: a_f.clone(),
        b: b_f.clone(),
        c: c_f.clone(),
        r,
        remainder: rem_f.clone(),
        div: div_f.clone(),
        input_0,
        c_0_converted: c0,
        multiplier,
    };

    // 소프트맥스 테이블 로드
    let tables_arr = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("softmax_tables.npy")
        .context("softmax_tables not found in integrated npz; ensure whole_pipeline embedded it")?;
    // 테이블값을 2차원으로 강제함
    let tables_2d = tables_arr.into_dimensionality::<Ix2>()
        .context("softmax_tables must be 2D (K, table_width)")?;

    // softmax metadata
    let soft_k_segments = npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::IxDyn>("softmax_k_segments.npy")
        .map(|arr| arr.into_dimensionality::<IxDyn>().ok().and_then(|a| a.iter().next().cloned()).unwrap_or(0))
        .unwrap_or(0) as usize;
    let soft_limb_bits = npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::IxDyn>("softmax_limb_bits.npy")
        .map(|arr| arr.into_dimensionality::<IxDyn>().ok().and_then(|a| a.iter().next().cloned()).unwrap_or(0))
        .unwrap_or(0) as usize;
    let soft_frac_bits = npz.by_name::<ndarray::OwnedRepr<i32>, ndarray::IxDyn>("softmax_tables_frac_bits.npy")
        .map(|arr| arr.into_dimensionality::<IxDyn>().ok().and_then(|a| a.iter().next().cloned()).unwrap_or(32))
        .unwrap_or(32) as i32;

    // 서킷에서의 설정값과 파이썬 파일에서의 설정값이 맞지 않을 때
    if soft_k_segments != K_SEGMENTS {
        return Err(anyhow::anyhow!(
            "softmax table k_segments={} does not match circuit K_SEGMENTS={} ; recompile or regenerate tables",
            soft_k_segments,
            K_SEGMENTS
        ));
    }

    // 테이블 또한 벡터로 변환
    let mut tables_field: Vec<Vec<F>> = Vec::with_capacity(tables_2d.shape()[0]);
    for k in 0..tables_2d.shape()[0] {
        let row = tables_2d.row(k);
        let mut v = Vec::with_capacity(row.len());
        for val in row.iter() {
            v.push(F::from(*val as u64));
        }
        tables_field.push(v);
    }

    // (추가) softmax lookup 테이블과 그 메타데이터를 .npz 파일로 임시 저장하는 코드
    let lookup_npz_path = "tmp_tables/embedded_softmax_for_lookup.npz";
    {
        let _ = std::fs::create_dir_all("tmp_tables");
        let f = File::create(lookup_npz_path).context("creating temporary lookup npz file")?;
        let mut writer = NpzWriter::new(f);
        writer.add_array("tables.npy", &tables_2d).context("writing tables to temp npz")?;
        let frac = arr0(soft_frac_bits as i32);
        let limb = arr0(soft_limb_bits as i32);
        writer.add_array("frac_bits.npy", &frac).context("writing frac_bits")?;
        writer.add_array("limb_bits.npy", &limb).context("writing limb_bits")?;
        writer.finish().context("finalizing temp npz write")?;
    }

    // 첫번째 행렬곱 이후의 타겟값 로드 
    // TODO : 최적화 > x가 아니라 e^x값만 넣어도 되려나?
    let targets_arr = read_array_u64(&mut npz, "matmul_targets")
        .context("matmul_targets not found or not u64; ensure whole_pipeline saved it as uint64")?;
    let t4 = targets_arr.into_dimensionality::<Ix4>().context("matmul_targets must be 4D")?;

    // 필요한 배열들 정의
    let mut matmul_outputs_all = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch];
    let mut limbs_all: Vec<Vec<Vec<Vec<[F; K_SEGMENTS]>>>> =
        vec![vec![vec![vec![[F::zero(); K_SEGMENTS]; seq_len]; seq_len]; num_head]; batch];
    let mut exp_limb_vals_all: Vec<Vec<Vec<Vec<[F; K_SEGMENTS]>>>> =
        vec![vec![vec![vec![[F::one(); K_SEGMENTS]; seq_len]; seq_len]; num_head]; batch];
    let mut softmax_outputs_all = vec![vec![vec![vec![F::zero(); seq_len]; seq_len]; num_head]; batch];
    let mut exp_prod_field_all = vec![vec![vec![vec![F::one(); seq_len]; seq_len]; num_head]; batch];

    // (추가)
    let mut prod_bigints_flat: Vec<BigInt> = Vec::new();

    for n in 0..batch {
        for h in 0..num_head {
            // t4가 ndarray이므로, 마찬가지로 Vec에 담기
            let mut matmul_ints: Vec<Vec<i128>> = vec![vec![0i128; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let v = t4[[n,h,i,j]]; // t4[n][h][i][j]
                    matmul_ints[i][j] = v as i128;
                }
            }

            // matmul_outputs_field : 이게 생 값. limb단위로 decompose하기 전
            // limbs_field : 각 k segment번째 테이블 인덱스, 2^?까지 적용한 값임
            // exp_limb_vals_field : 하나의 e^x에 대하여 세그먼트 나누어서 저장, e^한 상태임.
            // softmax_outputs_field : 소프트 맥스 결과 e^i / sum 모든 e
            let (matmul_outputs_field, limbs_field_dyn, exp_limb_vals_dyn, softmax_outputs_field) =
                build_softmax_witnesses::<F>(matmul_ints.clone(), &tables_field, soft_limb_bits);

            // batch랑 head깐 형태
            let mo = &matmul_outputs_field[0][0];
            let lf = &limbs_field_dyn[0][0];
            let ef = &exp_limb_vals_dyn[0][0];
            let so = &softmax_outputs_field[0][0];

            for i in 0..seq_len {
                for j in 0..seq_len {
                    matmul_outputs_all[n][h][i][j] = mo[i][j]; // 생값
                    let mut limb_fixed: [F; K_SEGMENTS] = [F::zero(); K_SEGMENTS];
                    let mut exp_fixed: [F; K_SEGMENTS] = [F::one(); K_SEGMENTS];
                    for k in 0..K_SEGMENTS {
                        limb_fixed[k] = lf[i][j][k]; // 각 k segment번째 테이블 인덱스, 세그먼트 나눈 상태, 2^?까지 적용한 값임
                        exp_fixed[k] = ef[i][j][k]; // 하나의 e^x에 대하여 세그먼트 나누어서 저장, e^한 상태임.
                    }
                    limbs_all[n][h][i][j] = limb_fixed;
                    exp_limb_vals_all[n][h][i][j] = exp_fixed;
                    softmax_outputs_all[n][h][i][j] = so[i][j]; // 소프트 맥스 결과 e^i / sum 모든 e

                    
                    let mut prod_field = F::one();
                    for k in 0..soft_k_segments {
                        prod_field *= ef[i][j][k]; // 하나의 e^x에 대하여 세그먼트 나누어서 저장, e^한 상태임.
                    }
                    exp_prod_field_all[n][h][i][j] = prod_field; // 각 세그먼트 모두모아 하나의 e^x 값으로 복원
                }
            }
        }
    }

    {
        // 서킷안에 들어가기 전에 맞는 값일지 잠깐 디버그
        println!("DEBUG: running numeric consistency checks before building circuits...");

        // 5차원 맞는지 모양 확인
        let limb_idx_arr_dyn = read_array_u64(&mut npz, "limb_indices")
            .context("limb_indices not present in npz")?;
        let limb_idx_dyn = limb_idx_arr_dyn; // ArrayD<u64>
        let limb_shape = limb_idx_dyn.shape().to_vec();
        if limb_shape.len() != 5 {
            println!("DEBUG: unexpected limb_indices shape: {:?}", limb_shape);
            return Err(anyhow::anyhow!("limb_indices has unexpected dimensionality"));
        } else {
           
            let k_dim = limb_shape[4];
            println!("DEBUG: limb_indices shape: {:?}", limb_shape);
            for n in 0..batch {
                for h in 0..num_head {
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let mut recomposed: u128 = 0;
                            for k in 0..k_dim {
                                let index = IxDyn(&[n, h, i, j, k]);
                                let idx_val = *limb_idx_dyn.get(index)
                                    .ok_or_else(|| anyhow::anyhow!("index out of bounds reading limb_indices"))?;
                                recomposed |= (idx_val as u128) << (k * soft_limb_bits);
                            }
                            let target = t4[[n,h,i,j]] as u128; // 결과값 
                            // [중요] 원래 정수를 복원한 것이 결과값과 일치하는지(아직 테이블 값과는 상관이 없음)
                            if recomposed != target {
                                println!("DIAG MISMATCH: recomposed limb != matmul_target at n{} h{} i{} j{}: recomposed={} target={}", n,h,i,j, recomposed, target);
                                return Err(anyhow::anyhow!("recomposition mismatch detected"));
                            }
                        }
                    }
                }
            }
            println!("DEBUG: recomposition check PASSED (limb indices -> targets)");
        }

        // 테이블 모양 확인한번 더
        let k_segments_u = tables_2d.shape()[0];
        let table_width = tables_2d.shape()[1];
        println!("DEBUG: tables shape K={}, W={}", k_segments_u, table_width);

        // 값 확인
        for n in 0..batch {
            for h in 0..num_head {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        // compute product in u128
                        let mut prod_u128: u128 = 1u128;
                        // For lookup compatibility: push per-segment table entries (n,h,i,j,k order)
                        for k in 0..limb_shape[4] {
                            let index = IxDyn(&[n, h, i, j, k]);
                            let idx = *limb_idx_dyn.get(index)
                                .ok_or_else(|| anyhow::anyhow!("index out of bounds reading limb_indices for exp_prod"))? as usize;
                            if idx >= table_width {
                                println!("DIAG: limb idx out of table bounds at n{} h{} i{} j{} k{} idx{} table_width{}", n,h,i,j,k,idx,table_width);
                                return Err(anyhow::anyhow!("limb idx out of bounds"));
                            }
                            let v = tables_2d[[k, idx]];
                            prod_u128 = prod_u128.saturating_mul(v as u128); // 오버플로우가 나면 값을 포화시키는 함수
                            // (추가) 룩업 증명을 하고 싶은 값 준비
                            prod_bigints_flat.push(BigInt::from(v as u64));
                        }
                        // convert prod_u128 -> field element
                        let prod_field = F::from(prod_u128);
                        let got = exp_prod_field_all[n][h][i][j]; // 각 세그먼트 모두모아 하나의 e^x 값으로 복원(테이블 참조한 값)
                        if prod_field != got { // npz파일로 읽어온 타겟값이 테이블을 참조한 e^x 값과 일치하는지 확인
                            println!("DIAG: exp_prod_field mismatch at n{} h{} i{} j{}: recomputed={:?} witness={:?}", n,h,i,j, prod_u128, got);
                            return Err(anyhow::anyhow!("exp_prod_field mismatch"));
                        }
                    }
                }
            }
        }
        println!("DEBUG: exp_prod recompute check PASSED");

        // 분수 확인 e^x/sum e = softmax_outputs
        for n in 0..batch {
            for h in 0..num_head {
                for i in 0..seq_len {
                    // denom = sum_j exp_prod_field_all[n,h,i,j]
                    let mut denom = F::zero();
                    for j in 0..seq_len {
                        denom += exp_prod_field_all[n][h][i][j];
                    }
                    for j in 0..seq_len {
                        let s = softmax_outputs_all[n][h][i][j];
                        let lhs = s * denom;
                        let rhs = exp_prod_field_all[n][h][i][j];
                        if lhs != rhs {
                            println!("DIAG: softmax normalization mismatch at n{} h{} i{} j{}: s*denom={:?} exp_prod={:?}", n,h,i,j, lhs, rhs);
                            return Err(anyhow::anyhow!("softmax normalization mismatch"));
                        }
                    }
                }
            }
        }
        println!("DEBUG: softmax normalization check PASSED (s*denom == exp_prod_field)");
    } 

    // (추가) 여기서부터 룩업 값준비
    let mut combined_matmul_ints: Vec<Vec<i128>> = Vec::with_capacity(batch * num_head * seq_len);
    for n in 0..batch {
        for h in 0..num_head {
            for i in 0..seq_len {
                let mut row: Vec<i128> = Vec::with_capacity(seq_len);
                for j in 0..seq_len {
                    row.push(t4[[n, h, i, j]] as i128); // t4 : 첫번째 행렬곱 이후의 타겟값
                }
                combined_matmul_ints.push(row); // e^x의 세그먼트가 아니라, x의 세그먼트임.
            }
        }
    }
    let rows = combined_matmul_ints.len();
    let lookup_batch_size = rows * seq_len * soft_k_segments;

    println!("DEBUG: invoking create_lookup_proof for entire dataset: rows={}, cols={}, k_segments={}, batch_size={}", rows, seq_len, soft_k_segments, lookup_batch_size);

    let cpinfo = create_lookup_proof::<E>(lookup_npz_path, combined_matmul_ints, lookup_batch_size)
        .map_err(|e| anyhow::anyhow!("create_lookup_proof failed: {:?}", e))?;
    println!("DEBUG: create_lookup_proof completed successfully for entire dataset");

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

    // ------------------- 링크 증명 생성 -------------------
    // 1) prod_bigints_flat -> E::ScalarField 벡터로 변환 (f 부분)
    let prod_fields: Vec<<E as Pairing>::ScalarField> = prod_bigints_flat
        .iter()
        .map(|b| bigint_to_fr(b.clone()))
        .collect();

    // cpinfo에서 반환받은 f_hat_fields, z_fields 사용
    //let cp_f_hat_fields = cpinfo.f_hat_fields.clone();
    //let cp_z_fields = cpinfo.z_fields.clone();

    // Convert cpinfo.f_bigints -> field vector (this corresponds to the plain values used to form cm_f if f_bigints is proven t)
    let cp_f_fields_from_bigints: Vec<<E as Pairing>::ScalarField> = cpinfo
        .f_bigints
        .iter()
        .map(|b| bigint_to_fr(b.clone()))
        .collect();

    // Debug logging: lengths and some sample values
    // print first few elements for quick inspection
    let show_n = |n: usize| -> usize { if n < 4 { n } else { 4 } };
    let nshow = 4usize;
    println!("DEBUG: sample prod_fields (first {}): {:?}", show_n(nshow), &prod_fields.iter().take(show_n(nshow)).collect::<Vec<_>>());
    println!("DEBUG: sample cp_f_fields_from_bigints (first {}): {:?}", show_n(nshow), &cp_f_fields_from_bigints.iter().take(show_n(nshow)).collect::<Vec<_>>());
    //println!("DEBUG: sample cp_f_hat_fields (first {}): {:?}", show_n(nshow), &cp_f_hat_fields.iter().take(show_n(nshow)).collect::<Vec<_>>());
    //println!("DEBUG: sample cp_z_fields (first {}): {:?}", show_n(nshow), &cp_z_fields.iter().take(show_n(nshow)).collect::<Vec<_>>());

    // 2) 오프닝 생성
    let mut link_rng = ark_std::rand::rngs::StdRng::seed_from_u64(43u64);
    // Use a distinct random opening for the local committed cm_prod (we want to prove equality even when openings differ)
    let o_prod: <E as Pairing>::ScalarField = <E as Pairing>::ScalarField::rand(&mut link_rng);

    // 3) cm_prod를 Projective로 안전하게 누적한 뒤 affinity로 변환
    let mut cm_prod_proj: <E as Pairing>::G1 = <E as Pairing>::G1::zero();
    cm_prod_proj.add_assign(&cpinfo.pp.ck[0].mul(o_prod));
    for (g_i, u_i) in cpinfo.pp.ck.iter().skip(1).zip(prod_fields.iter()) {
        cm_prod_proj.add_assign(g_i.mul(*u_i));
    }
    let cm_prod_affine: <E as Pairing>::G1Affine = cm_prod_proj.into_affine();

    // ---------------------------------------------------------------------------
    // 4) CP-Link 증명 생성 로직 (cm_prod == cm_f 증명)
    // ---------------------------------------------------------------------------

    // 1. CP-Link Setup (전용 모드 "simple_equal" 사용)
    // ck1: cm_prod를 위한 로컬 키, ck2: Harisa에서 전달받은 cpinfo.pp.ck
    let (link_pp, link_crs) = LinkSnark::<E>::setup(
        prod_fields.len(), 
        cpinfo.pp.ck.clone(), // ck1 (사용자 정의 ck를 쓰려면 여기에 입력)
        cpinfo.pp.ck.clone(), // ck2 (Harisa의 ck)
        "equal", 
        &mut link_rng
    );

    // 2. Keygen
    let (link_ek, link_vk) = LinkSnark::<E>::keygen(&link_pp, link_crs, &mut link_rng);

    // 3. Witness 생성
    // Witness 구조: [o_prod, o_f, m_1, ..., m_l]
    // generate_cp_witness는 r 벡터와 u 벡터를 순서대로 결합함
    let mut r_vec = Vec::new();
    r_vec.push(o_prod);      // cm_prod의 오프닝
    r_vec.push(cpinfo.o_f);  // cm_f의 오프닝

    let link_witness = LinkSnark::<E>::generate_witness(
        r_vec, 
        prod_fields.clone(), // 증명하고자 하는 공통 값 u
        vec![] 
    );

    // 4. Proof 생성
    let (link_proof, _) = LinkSnark::<E>::prove(&link_pp, &link_ek, link_witness, &mut link_rng);

    // 5. Instance 생성
    // Instance 구조: [cm_prod, cm_f]
    let mut instance_vec = Vec::new();
    instance_vec.push(cm_prod_affine);
    instance_vec.push(cpinfo.cm_f);

    // generate_instance가 내부적으로 push를 수행하므로 구조에 맞춰 전달
    let link_instance = instance_vec; 

    // 6. Verify
    let is_link_valid = LinkSnark::<E>::verify(&link_pp, &link_vk, &link_instance, &link_proof);

    println!("DEBUG: CP-Link equality verification result: {}", is_link_valid);
    if !is_link_valid {
        return Err(anyhow::anyhow!("CP-Link Equality check failed: cm_prod and cm_f differ"));
    }

    // 외부에서 Harisa lookup proof를 검증
    let flag = HarisaPlus::<E, Harisa<E, LinkSnark<E>>, LinkSnark<E>>::verify_lookup(
        cpinfo.pp,
        cpinfo.accum,
        cpinfo.cm_f_hat,
        cpinfo.cm_f,
        cpinfo.cm_z,
        cpinfo.lookup_proof,
    )?;
    if !flag {
        return Err(anyhow::anyhow!("Harisa lookup proof failed verification"));
    }

    // 최종 소프트맥스 서킷 구성
    let softmax_circuit = SoftmaxCircuit {
        matmul_outputs: matmul_outputs_all.clone(),
        limbs: limbs_all.clone(),
        exp_limb_vals: exp_limb_vals_all.clone(),
        softmax_outputs: softmax_outputs_all.clone(),
        frac_bits: soft_frac_bits as u32,
    };

    // 마지막 행렬곱도 프리발드로 검증
    let v_q_arr = read_array_i64(&mut npz, "V_q").context("reading V_q")?;
    let x_q_arr = read_array_i64(&mut npz, "X_q").context("reading X_q")?;
    let v4 = v_q_arr.into_dimensionality::<Ix4>().context("V_q must be 4D")?;
    let x4 = x_q_arr.into_dimensionality::<Ix4>().context("X_q must be 4D")?;

    let mut b2_f = vec![vec![vec![vec![F::zero(); head_dim]; seq_len]; num_head]; batch]; // n x p -> seq_len x head_dim
    let mut c2_f = vec![vec![vec![vec![F::zero(); head_dim]; seq_len]; num_head]; batch]; // m x p -> seq_len x head_dim

    for ((n,h,i,k), val) in v4.indexed_iter() {
        let v_i64 = *val;
        if v_i64 >= 0 {
            b2_f[n][h][i][k] = F::from(v_i64 as u128);
        } else {
            let abs = (-(v_i64 as i128)) as u128;
            b2_f[n][h][i][k] = F::zero() - F::from(abs);
        }
    }
    for ((n,h,i,k), val) in x4.indexed_iter() {
        let x_i64 = *val;
        if x_i64 >= 0 {
            c2_f[n][h][i][k] = F::from(x_i64 as u128);
        } else {
            let abs = (-(x_i64 as i128)) as u128;
            c2_f[n][h][i][k] = F::zero() - F::from(abs);
        }
    }

    let a2_f = exp_prod_field_all.clone();

    // 마지막 프리발드 서킷 구성
    let qkvfreivalds = FreivaldsMatMulCircuit {
        a: a2_f,
        b: b2_f,
        c: c2_f,
        r, 
    };

    // 최종 서킷 구성
    let combined = ZKDietCircuit {
            qkfreivalds: qkfreivalds.clone(),
            softmax: softmax_circuit,
            qkvfreivalds: qkvfreivalds.clone(),
        };

    // 디버그용 컨스트레인트 개수 출력
    {
        use ark_relations::r1cs::ConstraintSystem;
        // qkfreivalds
        let cs1 = ConstraintSystem::<F>::new_ref();
        println!("DEBUG: running qkfreivalds.generate_constraints on a fresh CS...");
        qkfreivalds.clone().generate_constraints(cs1.clone()).context("failed generating qkfreivalds constraints")?;
        println!("DEBUG: qkfreivalds CS satisfied?: {}", cs1.is_satisfied().unwrap_or(false));
        println!("DEBUG: qkfreivalds num_constraints: {}", cs1.num_constraints());

        // softmax
        let cs2 = ConstraintSystem::<F>::new_ref();
        println!("DEBUG: running softmax.generate_constraints on a fresh CS...");
        combined.softmax.clone().generate_constraints(cs2.clone()).context("failed generating softmax constraints")?;
        println!("DEBUG: softmax CS satisfied?: {}", cs2.is_satisfied().unwrap_or(false));
        println!("DEBUG: softmax num_constraints: {}", cs2.num_constraints());

        // qkvfreivalds
        let cs3 = ConstraintSystem::<F>::new_ref();
        println!("DEBUG: running qkvfreivalds.generate_constraints on a fresh CS...");
        qkvfreivalds.clone().generate_constraints(cs3.clone()).context("failed generating qkvfreivalds constraints")?;
        println!("DEBUG: qkvfreivalds CS satisfied?: {}", cs3.is_satisfied().unwrap_or(false));
        println!("DEBUG: qkvfreivalds num_constraints: {}", cs3.num_constraints());
    }

    Ok(combined)
}

pub fn run(path: &Path) -> Result<()> {
    let combined = load_and_build_combined::<Bn254>(path)?;

    {
        use ark_relations::r1cs::ConstraintSystem;
        let cs = ConstraintSystem::<F>::new_ref();
        combined.clone().generate_constraints(cs.clone())?;
        let satisfied = cs.is_satisfied().unwrap_or(false);
        println!("Constraint system satisfied?: {}", satisfied);
        println!("num constraints: {}", cs.num_constraints());
        if !satisfied {
            return Err(anyhow::anyhow!("CS not satisfied on provided witnesses; aborting prove"));
        }
    }

    let mut rng = StdRng::seed_from_u64(42u64);
    let (pk, vk) = CcGroth16::<Bn254>::setup(combined.clone(), &mut rng)?;
    let pvk = prepare_verifying_key::<Bn254>(&vk);

    let proof = CcGroth16::<Bn254>::prove(&pk, combined.clone(), &mut rng)?;
    let ok = CcGroth16::<Bn254>::verify_with_processed_vk(&pvk, &[combined.qkfreivalds.r], &proof)?;
    println!("Integrated verification result: {}", ok);
    if !ok {
        return Err(anyhow::anyhow!("verification failed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn zkdiet_proof_test() {
        let path = Path::new("tmp/integrated_inputs.npz");
        run(path).expect("Integrated proof runner failed in test");
    }
}