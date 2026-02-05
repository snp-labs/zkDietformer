/*use crate::{util::load::{read_array_u64, to_field_4d}};
use ark_ff::PrimeField;
use ark_std::fs::File;
use ndarray_npy::NpzReader;
use ndarray::{Ix0, Ix4, ArrayD, IxDyn};
use std::error::Error;

pub fn read_array_i64(npz: &mut NpzReader<File>, name: &str) -> Result<ArrayD<i64>, Box<dyn Error>> {
    let key = format!("{}.npy", name);
    let arr = npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>(&key)?;
    Ok(arr.into_owned().into_dimensionality().unwrap())
}

pub fn load_freivalds_circuit<F: PrimeField>(
    npz: &mut NpzReader<File>,
    r_challenge: F
) -> Result<MatmulBundle<F>, Box<dyn std::error::Error>> {
    // 1. 행렬 데이터 로드 (whole_pipeline.py에서 저장한 키값 기준)
    let q_q = read_array_i64(npz, "Q_q")?.into_dimensionality::<Ix4>()?;
    let q_k = read_array_i64(npz, "Q_k")?.into_dimensionality::<Ix4>()?;
    let q_y_sim = read_array_i64(npz, "Q_y1_sim")?.into_dimensionality::<Ix4>()?;
    let r_mean_k = read_array_i64(npz, "R_mean_k")?.into_dimensionality::<Ix4>()?;
    let r_matmul = read_array_i64(npz, "R_matmul_k1x")?.into_dimensionality::<Ix4>()?;

    // 2. 양자화 파라미터 및 챌린지 로드
    // int/float 타입을 필드 요소(F)로 변환
    let z_q = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("z_q.npy")?.into_scalar();
    let z_k = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("z_k.npy")?.into_scalar();
    let z_y = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("z_y1.npy")?.into_scalar();
    let m_k1 = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("M_k1.npy")?.into_scalar();
    let m_prime = npz.by_name::<ndarray::OwnedRepr<i64>, Ix0>("M_prime_k1.npy")?.into_scalar();
    
    Ok(MatmulBundle {
        q_q: to_field_4d(&q_q),
        q_k: to_field_4d(&q_k),
        q_y_sim: to_field_4d(&q_y_sim),
        r_mean_k: to_field_4d(&r_mean_k),
        r_matmul: to_field_4d(&r_matmul),
        z_q: F::from(z_q),
        z_k: F::from(z_k),
        z_y: F::from(z_y),
        m_k1: F::from(m_k1),
        m_prime: F::from(m_prime),
        r_challenge
    })
}*/