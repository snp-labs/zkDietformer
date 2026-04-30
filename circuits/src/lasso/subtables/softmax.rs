use ark_ff::PrimeField;
use ark_std::log2;
use ndarray_npy::NpzReader;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock; // 메모리 캐싱을 위해 추가
use ndarray::{Array2, Ix2};
use super::SubtableStrategy;

pub enum SoftmaxSubtableStrategy {}

/// 필드 독립적인 u64 데이터를 캐싱하기 위한 정적 변수
static SOFTMAX_CACHE: OnceLock<Array2<u64>> = OnceLock::new();

// ── Phase 5-B B6-A1: Softmax 표 path 동적화 (LayerNorm B1 평행 패턴) ─────────
//
// LayerNorm B1 (`subtables/layernorm.rs`) 와 정확히 동일 디자인 — 자체학습
// 호환을 위해 default fallback `"outputs/softmax_tables.npz"` 보존.
//
// 우선순위 (3-tier):
//   1. `set_softmax_table_dir(dir)` 로 설정된 경로 (OnceLock first-write-wins)
//   2. 환경 변수 `SOFTMAX_TABLE_PATH`
//   3. default `"outputs/softmax_tables.npz"` (자체학습 호환)
//
// Python `softmax_table.py:generate_softmax_tables(out_dir, ...)` 로 만든 파일을
// witness 와 같은 dir 에 두고 setter 로 지정. 또는 env var 로 직접 path 명시.
static SOFTMAX_TABLE_PATH: OnceLock<String> = OnceLock::new();

/// 프로그램 측 setter — Python pipeline 이 작성한 `softmax_tables.npz` 가
/// 들어 있는 디렉터리를 지정. 보통 attention witness npz 와 같은 dir.
///
/// Usage:
///     set_softmax_table_dir("outputs_m64");   // Nyström attention
///     set_softmax_table_dir("outputs_bert");  // BERT attention
///
/// OnceLock 의미상 두 번째 호출은 silently 무시됨 (first-write-wins).
pub fn set_softmax_table_dir(dir: impl AsRef<Path>) {
    let path: PathBuf = dir.as_ref().join("softmax_tables.npz");
    let path_str = path.to_string_lossy().into_owned();
    let _ = SOFTMAX_TABLE_PATH.set(path_str);
}

/// 현재 적용될 표 파일 경로 결정. setter > env var > default 순.
fn get_softmax_table_path() -> String {
    if let Some(p) = SOFTMAX_TABLE_PATH.get() {
        return p.clone();
    }
    if let Ok(env_path) = env::var("SOFTMAX_TABLE_PATH") {
        return env_path;
    }
    "outputs/softmax_tables.npz".to_string()
}

/// 파일에서 테이블을 한 번만 로드하는 헬퍼 함수
fn get_or_init_softmax_cache(c: usize, m: usize) -> &'static Array2<u64> {
    SOFTMAX_CACHE.get_or_init(|| {
        let lookup_npz_path = get_softmax_table_path();
        let path = Path::new(&lookup_npz_path);

        let file = File::open(path).unwrap_or_else(|e| {
            panic!(
                "Softmax table file not found at '{}': {}.\n\
                 Set the path via `set_softmax_table_dir()` (call before first lookup),\n\
                 via env var `SOFTMAX_TABLE_PATH=path/to/softmax_tables.npz`,\n\
                 or place the file at default path `outputs/softmax_tables.npz`.\n\
                 Generate via Python: `from softmax_table import generate_softmax_tables; \
                 generate_softmax_tables('<out_dir>', k_bits=22, frac_bits=16)`.",
                lookup_npz_path, e,
            )
        });
        let mut npz = NpzReader::new(file).expect("failed to create NpzReader");

        let tables_arr_dyn = npz
            .by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("tables.npy")
            .or_else(|_| npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("softmax_tables.npy"))
            .or_else(|_| npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("tables"))
            .expect("npz does not contain tables");

        let tables_2d = tables_arr_dyn
            .into_dimensionality::<Ix2>()
            .expect("softmax tables must be 2D");

        // 컴파일 타임 C, M과 실제 로드된 데이터 규격 확인
        let shape = tables_2d.shape();
        assert_eq!(shape[0], c, "k_segments mismatch");
        assert_eq!(shape[1], m, "table width mismatch");

        tables_2d
    })
}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
    for SoftmaxSubtableStrategy
{
    const NUM_SUBTABLES: usize = C; // 서브테이블과 메모리 1:1 매핑
    const NUM_MEMORIES: usize = C;

    /// 메모리 인덱스(0 ~ C-1)를 서브테이블 인덱스(0 ~ C-1)로 1:1 매핑
    fn memory_to_subtable_index(memory_index: usize) -> usize {
        debug_assert!(memory_index < C);
        memory_index
    }

    /// 메모리 인덱스가 어느 차원(Dimension)에 대응하는지 정의
    /// 각 세그먼트가 입력의 각 파트에 대응하므로 memory_index를 그대로 사용
    fn memory_to_dimension_index(memory_index: usize) -> usize {
        debug_assert!(memory_index < C);
        memory_index
    }

    fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
        let tables_2d = get_or_init_softmax_cache(C, M);
        
        // 각 메모리가 어떤 서브테이블을 참조하는지 1:1로 매핑
        std::array::from_fn(|i| {
            tables_2d.row(i).iter().map(|&val| F::from(val)).collect()
        })
    }

    fn evaluate_subtable_mle(subtable_index: usize, point: &Vec<F>) -> F {
        let bits_per_operand = (log2(M) as usize).max(1);
        let tables_2d = get_or_init_softmax_cache(C, M);

        // MLE 평가를 위한 기저 다항식(Lagrange basis) 계산 최적화 (O(M))
        // f(r) = \sum V_i * \chi_i(r) 에서 모든 \chi_i(r)을 DP로 미리 계산
        let compute_eq_evals = |point_slice: &[F]| -> Vec<F> {
            let b = point_slice.len();
            let mut evals = vec![F::one(); 1 << b];
            let mut size = 1;
            for j in 0..b {
                // MSB-first 매핑 (Lasso 표준에 맞춰 point_slice[b-1-j] 사용)
                let r = point_slice[b - 1 - j];
                for i in 0..size {
                    evals[i + size] = evals[i] * r;
                    evals[i] = evals[i] * (F::one() - r);
                }
                size *= 2;
            }
            evals
        };

        let eval_row_mle = |seg: usize, point_slice: &[F]| -> F {
            let chi = compute_eq_evals(point_slice);
            let mut acc = F::zero();
            for (idx, &basis) in chi.iter().enumerate() {
                let entry_u64 = tables_2d[[seg, idx]];
                acc += F::from(entry_u64) * basis;
            }
            acc
        };

        eval_row_mle(subtable_index, point)
        
    }

    fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C, M>>::NUM_MEMORIES]) -> F {
        vals.iter().product()
    }

    fn g_poly_degree() -> usize {
         C 
    }
}


#[cfg(test)]
mod test {
    use ark_curve25519::Fr;
    use ndarray_npy::{NpzReader, NpzWriter};
    use ndarray::{arr0, Ix2, Array2};
    use std::fs::{self, File};
    use std::path::Path;
    use crate::lasso::utils::index_to_field_bitvector;
    use super::*;

    //[주의] 내가 만든 테이블의 SPEC과 동일하게 맞춰 주어야 함.
    const SEG_LIMB_BITS: usize = 16;
    const C_TEST: usize = 5;
    const WIDTH: usize = 1 << SEG_LIMB_BITS;

    fn setup_test_npz() {
        let dir = "outputs";
        let path = "outputs/softmax_tables.npz";
        fs::create_dir_all(dir).unwrap();

        if !Path::new(path).exists() {
            let file = File::create(path).unwrap();
            let mut writer = NpzWriter::new(file);
            
            let data = Array2::<u64>::from_shape_fn((C_TEST, WIDTH), |(i, j)| (i * WIDTH + j) as u64);
            writer.add_array("tables.npy", &data).unwrap();
            writer.add_array("frac_bits.npy", &arr0(32i32)).unwrap();
            writer.add_array("limb_bits.npy", &arr0(SEG_LIMB_BITS as i32)).unwrap();
            writer.finish().unwrap();
        }
    }

    #[test]
    fn softmax_materialization_matches_npz() {
        setup_test_npz();
        
        let materialized: [Vec<Fr>; C_TEST] =
            <SoftmaxSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::materialize_subtables();

        let cache = get_or_init_softmax_cache(C_TEST, WIDTH);
        
        // 각 서브테이블이 WIDTH 크기를 가지는지 확인
        for s in 0..C_TEST {
            assert_eq!(materialized[s].len(), WIDTH);
            for i in 0..WIDTH {
                assert_eq!(materialized[s][i], Fr::from(cache[[s, i]]));
            }
        }
    }

    #[test]
    fn softmax_mle_parity() {
        setup_test_npz();
        
        let materialized = <SoftmaxSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::materialize_subtables();
        
        for s in 0..C_TEST {
            for i in 0..WIDTH {
                let point = index_to_field_bitvector(i, SEG_LIMB_BITS);
                let mle_val = <SoftmaxSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::evaluate_subtable_mle(s, &point);
                
                // 2차원 배열 형태인 materialized[s][i]와 비교
                let expected_val = materialized[s][i];
                
                assert_eq!(
                    mle_val, 
                    expected_val, 
                    "Mismatch at segment {} index {}", s, i
                );
            }
        }
    }

    #[test]
    fn combine_lookups_is_product() {
        setup_test_npz();
        
        let v1 = Fr::from(10u64);
        let v2 = Fr::from(20u64);
        let v3 = Fr::from(30u64);
        let v4 = Fr::from(40u64);
        let v5 = Fr::from(50u64);
        let vals: [Fr; C_TEST] = [v1, v2, v3, v4, v5];
        
        let combined = <SoftmaxSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::combine_lookups(&vals);
        assert_eq!(combined, v1 * v2 * v3 * v4 * v5);
    }
}