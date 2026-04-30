use ark_ff::PrimeField;
use ark_std::log2;
use ndarray_npy::NpzReader;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use ndarray::{Array1, Array2, Ix2, Ix1};
use super::SubtableStrategy;

// ── Subtable strategy ───────────────────────────────────────────────────────
//
// 본 strategy 는 `1/√(i + ε)·2^FRAC_BITS` (Phase 4 이전) 또는
// `1/√(i/VAR_SCALE + ε)·2^FRAC_BITS` (Phase 4 후, VAR_SCALE=1000) 형태의
// 정수-인덱스 lookup 표를 사용. 함수형 자체는 Rust 코드에 없고, npz 파일에서
// 로드한 표 값을 그대로 사용. (즉 SCALE 변경은 Python 측 표 재생성으로 완결.)
//
// History:
//   - 자체학습 RMSNorm 시절 작성. type name `RMSNormSubtableStrategy` 보존
//     (`mlp_lookup.rs::run_lasso_for_rmsnorm` backward compat).
//   - Phase 5-B (LayerNorm 마이그레이션): 같은 strategy 를 LayerNorm 가젯도
//     공유 — 표 파일명을 `layernorm_tables.npz` 로 통일 (D3: a 결정),
//     path 를 동적으로 (D2: A 결정).

pub enum RMSNormSubtableStrategy {}

/// LayerNorm 측 사용을 위한 type alias (의미 정합).
/// 내부 구현은 `RMSNormSubtableStrategy` 와 동일 — 표 값이 LayerNorm 의
/// `1/√(var/SCALE+ε)` 으로 채워졌다는 의미만 다름.
pub type LayerNormSubtableStrategy = RMSNormSubtableStrategy;

// ── Phase 5-B B2 (D5: var_scale sanity check) ───────────────────────────────
//
// 회로 측이 기대하는 VAR_SCALE 값. Python pipeline (layernorm_table.py:
// VAR_SCALE) 과 정확히 일치해야 함. Phase 4 결정 (1000) 기반.
//
// 변경 시 양 측 동시 갱신 필수:
//   - circuits/src/lasso/subtables/layernorm.rs (이 파일)
//   - models/Nystromformer/code/dietformer/layernorm_table.py
pub const EXPECTED_VAR_SCALE: i32 = 1000;

static LAYERNORM_CACHE: OnceLock<Array1<u64>> = OnceLock::new();

/// 표 파일 경로 override. 첫 번째 lookup 전에 한 번 호출.
/// OnceLock 의미상 두 번째 호출은 silently 무시됨 (first-write-wins).
///
/// fallback 우선순위:
/// 1. `set_layernorm_table_dir(dir)` 호출로 설정된 경로
/// 2. 환경 변수 `LAYERNORM_TABLE_PATH`
/// 3. default `"outputs/layernorm_tables.npz"`
static LAYERNORM_TABLE_PATH: OnceLock<String> = OnceLock::new();

/// 프로그램 측 setter — Python pipeline 이 작성한 `layernorm_tables.npz` 가
/// 들어 있는 디렉터리를 지정. 보통 witness npz 와 같은 dir.
///
/// Usage:
///     set_layernorm_table_dir("outputs_m64");           // Nystromformer
///     set_layernorm_table_dir("outputs_bert");          // BERT baseline
pub fn set_layernorm_table_dir(dir: impl AsRef<Path>) {
    let path: PathBuf = dir.as_ref().join("layernorm_tables.npz");
    let path_str = path.to_string_lossy().into_owned();
    // OnceLock::set 은 already-set 시 Err 반환 — silently 받아들임
    let _ = LAYERNORM_TABLE_PATH.set(path_str);
}

/// 현재 적용될 표 파일 경로 결정. setter > env var > default 순.
fn get_layernorm_table_path() -> String {
    if let Some(p) = LAYERNORM_TABLE_PATH.get() {
        return p.clone();
    }
    if let Ok(env_path) = env::var("LAYERNORM_TABLE_PATH") {
        return env_path;
    }
    "outputs/layernorm_tables.npz".to_string()
}

/// LayerNorm 표 lazy load — 첫 호출 시 npz 에서 로드 + 캐싱.
///
/// Phase 5-B B2 추가: var_scale 메타 검증. Python 측 SCALE 과 회로 측
/// 기대값 (`EXPECTED_VAR_SCALE`) 미스매치 시 즉시 panic.
fn get_or_init_layernorm_cache(m: usize) -> &'static Array1<u64> {
    LAYERNORM_CACHE.get_or_init(|| {
        let lookup_npz_path = get_layernorm_table_path();
        let file = File::open(Path::new(&lookup_npz_path)).unwrap_or_else(|e| {
            panic!(
                "LayerNorm table file not found at '{}': {}.\n\
                 Set the path via `set_layernorm_table_dir()` (call before first lookup)\n\
                 or via env var `LAYERNORM_TABLE_PATH=path/to/layernorm_tables.npz`.",
                lookup_npz_path, e,
            )
        });
        let mut npz = NpzReader::new(file).expect("failed to create NpzReader");

        // ── B2 sanity check: var_scale 메타 일치 검증 ───────────────────────
        // Python `layernorm_table.py` 는 `np.int32(var_scale)` 으로 저장.
        let actual_var_scale: i32 = npz
            .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix0>("var_scale.npy")
            .unwrap_or_else(|e| panic!(
                "var_scale.npy not found in '{}': {}.\n\
                 Python pipeline must include var_scale in layernorm_tables.npz \
                 (see layernorm_table.py:generate_layernorm_tables, post-Phase 4).",
                lookup_npz_path, e,
            ))
            .into_scalar();
        if actual_var_scale != EXPECTED_VAR_SCALE {
            panic!(
                "var_scale mismatch in '{}': expected {}, got {}.\n\
                 Either regenerate the npz with VAR_SCALE={} \
                 (Python: layernorm_table.py:VAR_SCALE)\n\
                 or update EXPECTED_VAR_SCALE in \
                 circuits/src/lasso/subtables/layernorm.rs.",
                lookup_npz_path, EXPECTED_VAR_SCALE, actual_var_scale, EXPECTED_VAR_SCALE,
            );
        }

        // ── 표 데이터 로드 (B1 부터 동일) ─────────────────────────────────
        let tables_arr_dyn = npz
            .by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("table.npy")
            .expect("npz does not contain table.npy");

        let tables_1d = tables_arr_dyn
            .into_dimensionality::<Ix1>()
            .expect("layernorm tables must be 1D");

        assert_eq!(tables_1d.len(), m, "LayerNorm table size mismatch");
        tables_1d
    })
}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
    for RMSNormSubtableStrategy
{
    const NUM_SUBTABLES: usize = C;
    const NUM_MEMORIES: usize = C;

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
        let tables_1d = get_or_init_layernorm_cache(M);

        // 각 메모리가 어떤 서브테이블을 참조하는지 1:1로 매핑
        std::array::from_fn(|i| {
            tables_1d.iter().map(|&val| F::from(val)).collect()
        })
    }


    fn evaluate_subtable_mle(_subtable_index: usize, point: &Vec<F>) -> F {
        let table = get_or_init_layernorm_cache(M);
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

    fn g_poly_degree() -> usize {
         C
    }
}

#[cfg(test)]
mod test {
    use ark_curve25519::Fr;
    use ndarray_npy::{NpzReader, NpzWriter};
    use ndarray::{arr0, Ix1, Array1};
    use std::fs::{self, File};
    use std::path::Path;
    use crate::lasso::utils::index_to_field_bitvector;
    use super::*;

    const TABLE_BITS: usize = 16;
    const WIDTH: usize = 1 << TABLE_BITS;
    const C_TEST: usize = 1;

    fn setup_layernorm_test_npz() {
        let dir = "outputs";
        let path = "outputs/layernorm_tables.npz";
        fs::create_dir_all(dir).unwrap();

        // 항상 재생성 — Phase 5-B B2 에서 var_scale.npy 추가됨. 이전 cargo test
        // 실행에서 작성된 stale 파일 (var_scale 없음) 을 회피하기 위해
        // `if !exists` 분기 제거.
        let file = File::create(path).unwrap();
        let mut writer = NpzWriter::new(file);

        let data = Array1::<u64>::from_shape_fn(WIDTH, |i| (i + 100) as u64);
        writer.add_array("table.npy", &data).unwrap();

        // B2: var_scale 메타 (회로 측 sanity check 가 검증)
        let var_scale_arr = arr0(EXPECTED_VAR_SCALE);
        writer.add_array("var_scale.npy", &var_scale_arr).unwrap();

        writer.finish().unwrap();
    }

    #[test]
    fn layernorm_materialization_parity() {
        setup_layernorm_test_npz();

        // materialize_subtables 호출
        let materialized: [Vec<Fr>; C_TEST] =
            <RMSNormSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::materialize_subtables();

        let cache = get_or_init_layernorm_cache(WIDTH);

        assert_eq!(materialized[0].len(), WIDTH);
        for i in 0..WIDTH {
            assert_eq!(materialized[0][i], Fr::from(cache[i]));
        }
    }

    #[test]
    fn layernorm_mle_eval_test() {
        setup_layernorm_test_npz();

        let materialized = <RMSNormSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::materialize_subtables();

        // 무작위 지점이 아닌, 각 인덱스(정점)에서의 MLE 값이 테이블 값과 일치하는지 확인
        // 인덱스 0, 1, 100, WIDTH-1 등 주요 지점 테스트
        let test_indices = [0, 1, 100, 500, WIDTH - 1];

        for &idx in test_indices.iter() {
            let point = index_to_field_bitvector(idx, TABLE_BITS);
            let mle_val = <RMSNormSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::evaluate_subtable_mle(0, &point);

            let expected_val = materialized[0][idx];
            assert_eq!(mle_val, expected_val, "MLE mismatch at index {}", idx);
        }
    }

    #[test]
    fn layernorm_combine_is_identity() {
        let val = Fr::from(12345u64);
        let vals: [Fr; C_TEST] = [val];

        let combined = <RMSNormSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::combine_lookups(&vals);
        assert_eq!(combined, val);
    }
}
