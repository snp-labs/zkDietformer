use ark_ff::PrimeField;
use ndarray_npy::NpzReader;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use ndarray::{Array1, Ix1};
use crate::lasso::subtables::SubtableStrategy;

pub enum GeLUSubtableStrategy {}

static GELU_CACHE: OnceLock<Array1<u64>> = OnceLock::new();
static GELU_TABLE_PATH: OnceLock<String> = OnceLock::new();

/// witness npz 와 같은 디렉터리에 있는 `gelu_tables.npz` 경로를 설정.
///
/// Usage:
///     set_gelu_table_dir("outputs_m64");
///     set_gelu_table_dir("outputs_bert");
///
/// OnceLock 의미상 두 번째 호출은 silently 무시됨 (first-write-wins).
pub fn set_gelu_table_dir(dir: impl AsRef<Path>) {
    let path: PathBuf = dir.as_ref().join("gelu_tables.npz");
    let path_str = path.to_string_lossy().into_owned();
    let _ = GELU_TABLE_PATH.set(path_str);
}

/// 현재 적용될 표 파일 경로 결정. setter > env var > default 순.
fn get_gelu_table_path() -> String {
    if let Some(p) = GELU_TABLE_PATH.get() {
        return p.clone();
    }
    if let Ok(env_path) = env::var("GELU_TABLE_PATH") {
        return env_path;
    }
    "outputs/gelu_tables.npz".to_string()
}

fn get_or_init_gelu_cache(m: usize) -> &'static Array1<u64> {
    GELU_CACHE.get_or_init(|| {
        let lookup_npz_path = get_gelu_table_path();
        let file = File::open(Path::new(&lookup_npz_path)).unwrap_or_else(|e| {
            panic!(
                "GeLU table file not found at '{}': {}.\n\
                 Set the path via `set_gelu_table_dir()` (call before first lookup),\n\
                 via env var `GELU_TABLE_PATH=path/to/gelu_tables.npz`,\n\
                 or place the file at default path `outputs/gelu_tables.npz`.",
                lookup_npz_path, e,
            )
        });
        let mut npz = NpzReader::new(file).expect("failed to create NpzReader");

        let tables_arr_dyn = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("table.npy")
            .expect("npz does not contain table");

        let tables_1d = tables_arr_dyn
            .into_dimensionality::<Ix1>()
            .expect("gelu tables must be 1D");

        assert_eq!(tables_1d.len(), m, "GeLU table size mismatch");
        tables_1d
    })
}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M> 
    for GeLUSubtableStrategy
{
    const NUM_SUBTABLES: usize = C;
    const NUM_MEMORIES: usize = C;

    fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
        let tables_1d = get_or_init_gelu_cache(M);
        std::array::from_fn(|_| {
            tables_1d.iter().map(|&val| F::from(val)).collect()
        })
    }

    fn evaluate_subtable_mle(_subtable_index: usize, point: &Vec<F>) -> F {
        let table = get_or_init_gelu_cache(M);
        let b = point.len();
        let mut evals = vec![F::one(); 1 << b];
        let mut size = 1;
        for j in 0..b {
            let r = point[b - 1 - j];
            for i in 0..size {
                evals[i + size] = evals[i] * r;
                evals[i] = evals[i] * (F::one() - r);
            }
            size *= 2;
        }
        let mut acc = F::zero();
        for (idx, &basis) in evals.iter().enumerate() {
            acc += F::from(table[idx]) * basis;
        }
        acc
    }

    fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C, M>>::NUM_MEMORIES]) -> F { 
        vals[0] 
    }
    fn g_poly_degree() -> usize { C }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ndarray_npy::{NpzReader, NpzWriter};
    use ndarray::{arr0, Ix1, Array1};
    use std::fs::{self, File};
    use std::path::Path;
    use crate::lasso::utils::index_to_field_bitvector;
    use super::*;

    // GeLU 테이블 스펙 정의
    const TABLE_BITS: usize = 16;
    const WIDTH: usize = 1 << TABLE_BITS; // 65536
    const C_TEST: usize = 1; 

    /// 테스트용 임시 gelu_tables.npz 파일을 생성합니다.
    fn setup_gelu_test_npz() {
        let dir = "outputs";
        let path = "outputs/gelu_tables.npz";
        fs::create_dir_all(dir).unwrap();

        if !Path::new(path).exists() {
            let file = File::create(path).unwrap();
            let mut writer = NpzWriter::new(file);
            
            // 테스트용 데이터: index i에 대하여 (i + 500) 값을 저장
            let data = Array1::<u64>::from_shape_fn(WIDTH, |i| (i + 500) as u64);
            writer.add_array("table.npy", &data).unwrap();
            writer.finish().unwrap();
        }
    }

    #[test]
    fn gelu_materialization_parity() {
        setup_gelu_test_npz();
        
        // 1. materialize_subtables 호출하여 필드 원소 벡터 획득
        let materialized: [Vec<Fr>; C_TEST] =
            <GeLUSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::materialize_subtables();

        // 2. 캐시된 원본 데이터와 비교
        let cache = get_or_init_gelu_cache(WIDTH);
        
        assert_eq!(materialized[0].len(), WIDTH);
        for i in 0..WIDTH {
            assert_eq!(materialized[0][i], Fr::from(cache[i]));
        }
        println!("✅ GeLU Materialization Parity: OK");
    }

    #[test]
    fn gelu_mle_eval_test() {
        setup_gelu_test_npz();
        
        let materialized = <GeLUSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::materialize_subtables();
        
        // 3. 주요 지점(Boolean Hypercube의 정점)에서 MLE 값이 테이블 값과 일치하는지 확인
        let test_indices = [0, 1, 1024, 32768, WIDTH - 1];
        
        for &idx in test_indices.iter() {
            // 인덱스를 비트 벡터(Lasso 지점)로 변환
            let point = index_to_field_bitvector(idx, TABLE_BITS);
            
            // MLE 평가 실행
            let mle_val = <GeLUSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::evaluate_subtable_mle(0, &point);
            
            let expected_val = materialized[0][idx];
            assert_eq!(mle_val, expected_val, "MLE mismatch at index {}", idx);
        }
        println!("✅ GeLU MLE Evaluation Test: OK");
    }

    #[test]
    fn gelu_combine_is_identity() {
        // GeLU는 단일 서브테이블을 사용하므로 combine_lookups가 입력 값을 그대로 반환해야 함
        let val = Fr::from(98765u64);
        let vals: [Fr; C_TEST] = [val];
        
        let combined = <GeLUSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::combine_lookups(&vals);
        assert_eq!(combined, val);
        println!("✅ GeLU Combine Lookups (Identity): OK");
    }
}