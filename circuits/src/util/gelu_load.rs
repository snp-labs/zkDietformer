use anyhow::{Context, Result};
use ndarray::{Dimension, Ix0, Ix4, IxDyn};
use ndarray_npy::NpzReader;
use std::fs::File;
use ark_ff::PrimeField;
use crate::util::load::read_array_u64;
use ndarray::{ArrayD, Ix2};
use crate::gadgets::{activation::gelu::GeluBundle, layernorm::constraints::LayerNormBundle};

// GeLU 메타데이터 로드 함수
pub fn load_gelu_metadata(reader: &mut NpzReader<File>) -> (u32, usize, usize, i64, i64, i32, u64) {
    let frac_bits = reader.by_name::<ndarray::OwnedRepr<i32>, IxDyn>("gelu_frac_bits.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(16) as u32; // Python에서 int32로 저장하므로 i32로 읽음
    let limb_bits = reader.by_name::<ndarray::OwnedRepr<i32>, IxDyn>("gelu_limb_bits.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(16) as usize;
    let k_segments = reader.by_name::<ndarray::OwnedRepr<i32>, IxDyn>("gelu_k_segments.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(5) as usize;
    let scale_input = reader.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("gelu_scale_input.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(256i64);
    let multiplier = reader.by_name::<ndarray::OwnedRepr<i64>, IxDyn>("gelu_multiplier.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(1i64);
    let base_scale_exp = reader.by_name::<ndarray::OwnedRepr<i32>, IxDyn>("gelu_base_scale_exp.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(16i32);
    let input_0 = reader.by_name::<ndarray::OwnedRepr<u64>, IxDyn>("gelu_input_0.npy")
        .ok()
        .and_then(|a| a.into_dimensionality::<Ix0>().ok().and_then(|s| s.iter().next().cloned()))
        .unwrap_or(0u64);
    
    (frac_bits, limb_bits, k_segments, scale_input, multiplier, base_scale_exp, input_0)
}

pub fn load_gelu_bundle<F: PrimeField>(
    reader: &mut NpzReader<File>,
    prefix: &str, // 보통 "gelu_"
) -> Result<GeluBundle<F>> {
    // 1. 메타데이터 로드
    let (frac_bits, limb_bits, k_segments, scale_input, multiplier, _base_scale_exp, input_0) =
        load_gelu_metadata(reader);

    // 2. 양자화된 입력/출력 로드
    let in_q_arr = read_array_u64(reader, &format!("{}in_q", prefix))?;
    let out_q_arr = read_array_u64(reader, &format!("{}out_q", prefix))?;
    
    // [중요] GeLU 및 시그모이드 나머지 값 로드 추가
    let gelu_rem_arr = read_array_u64(reader, &format!("{}remainders", prefix))
        .context("gelu_remainders missing")?;
    let sig_q_arr = read_array_u64(reader, &format!("{}sigmoid_q", prefix))
        .context("gelu_sigmoid_q missing")?;
    let sig_rem_arr = read_array_u64(reader, &format!("{}sigmoid_rem", prefix))
        .context("gelu_sigmoid_rem missing")?;

    // 3. 룩업 테이블 로드
    let tables_arr = reader.by_name::<ndarray::OwnedRepr<u64>, Ix2>("gelu_tables.npy")?;
    let table_width = tables_arr.shape()[1];

    // 4. 림브 데이터 로드 (Python에서 저장된 gelu_limbs 사용)
    let limbs_arr = reader.by_name::<ndarray::OwnedRepr<u64>, Ix2>("gelu_limbs.npy")
        .context("gelu_limbs missing")?;

    let n = in_q_arr.len();
    let mut inputs_q_field = Vec::with_capacity(n);
    let mut gelu_outputs_field = Vec::with_capacity(n);
    let mut gelu_rem_field = Vec::with_capacity(n);
    let mut limbs_field = Vec::with_capacity(n);
    let mut exp_limb_vals_field = Vec::with_capacity(n);
    let mut sigmoid_field = Vec::with_capacity(n);
    let mut sigmoid_rem_field = Vec::with_capacity(n);

    // 5. 필드 요소로 변환 및 번들 구성
    for i in 0..n {
        inputs_q_field.push(F::from(in_q_arr[i]));
        gelu_outputs_field.push(F::from(out_q_arr[i]));
        gelu_rem_field.push(F::from(gelu_rem_arr[i])); // GeLU 나머지 추가
        
        sigmoid_field.push(F::from(sig_q_arr[i]));
        sigmoid_rem_field.push(F::from(sig_rem_arr[i])); // 시그모이드 나머지 추가

        let mut limb_row = Vec::with_capacity(k_segments);
        let mut exp_row = Vec::with_capacity(k_segments);
        for k in 0..k_segments {
            let limb_idx = limbs_arr[[i, k]] as usize;
            limb_row.push(F::from(limb_idx as u64));
            exp_row.push(F::from(tables_arr[[k, limb_idx]]));
        }
        limbs_field.push(limb_row);
        exp_limb_vals_field.push(exp_row);
    }

    // 6. 최종 번들 생성
    let bundle = GeluBundle {
        inputs_q: inputs_q_field,
        limbs: limbs_field,
        exp_limb_vals: exp_limb_vals_field,
        sigmoid_outputs: sigmoid_field,
        sigmoid_remainders: sigmoid_rem_field,
        gelu_outputs: gelu_outputs_field,
        gelu_remainders: gelu_rem_field, // 추가됨
        input_0: F::from(input_0),
        scale_input: F::from(scale_input as u64),
        frac_bits: frac_bits,
        multiplier: F::from(multiplier as u64),
    };

    Ok(bundle)
}