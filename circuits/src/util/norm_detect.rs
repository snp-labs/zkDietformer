//! Normalization 종류 자동 감지 (Phase 5-B B3).
//!
//! Python pipeline 의 두 출력 (자체학습 RMSNorm vs HF LayerNorm) 을 npz 메타로
//! 구분. 회로 측 loader 가 적절한 분기 (RMSNorm 보존 / LayerNorm 신규) 를 선택.
//!
//! 시그널: `var_scale.npy` 의 존재 여부.
//! - 자체학습 RMSNorm npz (`normalization_pipeline.py`): var_scale 부재
//! - HF LayerNorm npz (`bert_normalization_pipeline.py`, Phase 4 후): var_scale 존재 (= 1000)
//!
//! 단일 책임 모듈 — gadget/loader 코드와 분리. 다른 위치에서도 재사용 가능.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray_npy::NpzReader;

/// Normalization 종류 — npz 메타로 자동 결정.
///
/// 추후 다른 normalization (RMSNorm, LayerNorm 외) 추가 시 enum variant 확장.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// 자체학습 모델 시절 (RMSNorm + 정수 인덱스 lookup). prefix=rms1/rms2.
    /// `enforce_rmsnorm` + `NormBundle` 사용.
    RMSNorm,
    /// HF 사전학습 모델 + Phase 4 lookup (LayerNorm + var_scale=1000). prefix=ln1/ln2.
    /// `enforce_layernorm` + `LayerNormBundle` 사용 (B4 신설).
    LayerNorm,
}

impl NormType {
    /// 블록 인덱스 (1 또는 2) 에 해당하는 npz 키 prefix.
    /// 예) NormType::LayerNorm.block_prefix(1) → "ln1"
    pub fn block_prefix(&self, idx: usize) -> &'static str {
        match (self, idx) {
            (NormType::RMSNorm, 1) => "rms1",
            (NormType::RMSNorm, 2) => "rms2",
            (NormType::LayerNorm, 1) => "ln1",
            (NormType::LayerNorm, 2) => "ln2",
            _ => panic!("invalid block index {} (expected 1 or 2)", idx),
        }
    }
}

/// npz 파일 메타로부터 NormType 결정.
///
/// 시그널: `var_scale.npy` 존재 여부 (Phase 4 후 LayerNorm 측에만 작성됨).
pub fn detect_norm_type<P: AsRef<Path>>(path: P) -> Result<NormType> {
    let path_ref = path.as_ref();
    let f = File::open(path_ref)
        .with_context(|| format!("opening npz for norm-type detection: {:?}", path_ref))?;
    let mut npz = NpzReader::new(f)?;

    // var_scale.npy 존재 → LayerNorm. 부재 → RMSNorm.
    // ndarray-npy 의 by_name 은 키 부재 시 Err 반환 — Result::is_ok() 로 분기.
    let has_var_scale = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("var_scale.npy")
        .is_ok();

    if has_var_scale {
        Ok(NormType::LayerNorm)
    } else {
        Ok(NormType::RMSNorm)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::arr0;
    use ndarray_npy::NpzWriter;
    use std::fs;

    /// helper: 테스트용 npz 작성
    fn write_test_npz(path: &str, with_var_scale: bool) {
        let dir = Path::new(path).parent().unwrap();
        fs::create_dir_all(dir).unwrap();
        if Path::new(path).exists() {
            fs::remove_file(path).unwrap();
        }

        let file = File::create(path).unwrap();
        let mut writer = NpzWriter::new(file);

        // 최소 필수 키 (k_bits, frac_bits)
        writer.add_array("k_bits.npy", &arr0(22u32)).unwrap();
        writer.add_array("frac_bits.npy", &arr0(8u32)).unwrap();

        if with_var_scale {
            writer.add_array("var_scale.npy", &arr0(1000i32)).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn detect_layernorm_via_var_scale_present() {
        let path = "outputs/test_detect_ln.npz";
        write_test_npz(path, true);

        let nt = detect_norm_type(path).unwrap();
        assert_eq!(nt, NormType::LayerNorm);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn detect_rmsnorm_via_var_scale_absent() {
        let path = "outputs/test_detect_rms.npz";
        write_test_npz(path, false);

        let nt = detect_norm_type(path).unwrap();
        assert_eq!(nt, NormType::RMSNorm);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn block_prefix_layernorm() {
        assert_eq!(NormType::LayerNorm.block_prefix(1), "ln1");
        assert_eq!(NormType::LayerNorm.block_prefix(2), "ln2");
    }

    #[test]
    fn block_prefix_rmsnorm() {
        assert_eq!(NormType::RMSNorm.block_prefix(1), "rms1");
        assert_eq!(NormType::RMSNorm.block_prefix(2), "rms2");
    }

    #[test]
    #[should_panic(expected = "invalid block index")]
    fn block_prefix_invalid_idx() {
        let _ = NormType::LayerNorm.block_prefix(3);
    }
}
