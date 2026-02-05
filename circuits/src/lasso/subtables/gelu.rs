use ark_ff::PrimeField;
use ark_std::log2;
use ndarray_npy::NpzReader;
use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;
use ndarray::{Array2, Ix2};

use super::SubtableStrategy;

pub enum GeluSubtableStrategy {}

/// GeLU 양자화 테이블 캐싱 (필드 독립적 u64 데이터)
static GELU_CACHE: OnceLock<Array2<u64>> = OnceLock::new();

/// 파일에서 GeLU 테이블을 로드하는 헬퍼 함수
fn get_or_init_gelu_cache(c: usize, m: usize) -> &'static Array2<u64> {
    GELU_CACHE.get_or_init(|| {
        // gelu_table.py에서 생성된 경로와 일치해야 함
        let lookup_npz_path = "tmp_tables/gelu_tables_quantized.npz".to_string();
        let path = Path::new(&lookup_npz_path);

        let file = File::open(path)
            .expect("GeluSubtable: failed to open gelu lookup npz");
        let mut npz = NpzReader::new(file).expect("failed to create NpzReader");

        let tables_arr_dyn = npz
            .by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("tables.npy")
            .expect("npz does not contain gelu tables");

        let tables_2d = tables_arr_dyn
            .into_dimensionality::<Ix2>()
            .expect("gelu tables must be 2D");

        let shape = tables_2d.shape();
        assert_eq!(shape[0], c, "k_segments mismatch for GeLU");
        assert_eq!(shape[1], m, "table width mismatch for GeLU");

        tables_2d
    })
}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
    for GeluSubtableStrategy
{
    const NUM_SUBTABLES: usize = C; 
    const NUM_MEMORIES: usize = C;

    fn memory_to_subtable_index(memory_index: usize) -> usize {
        memory_index
    }

    fn memory_to_dimension_index(memory_index: usize) -> usize {
        memory_index
    }

    /// 테이블 실체화: NPZ의 각 행(Segment)을 필드 요소 벡터로 변환
    fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
        let tables_2d = get_or_init_gelu_cache(C, M);
        std::array::from_fn(|i| {
            tables_2d.row(i).iter().map(|&val| F::from(val)).collect()
        })
    }

    /// MLE 평가: 다항식 확장을 통해 특정 지점의 값을 계산
    fn evaluate_subtable_mle(subtable_index: usize, point: &Vec<F>) -> F {
        let tables_2d = get_or_init_gelu_cache(C, M);

        // MLE 평가를 위한 기저 다항식 계산 (O(M))
        let compute_eq_evals = |point_slice: &[F]| -> Vec<F> {
            let b = point_slice.len();
            let mut evals = vec![F::one(); 1 << b];
            let mut size = 1;
            for j in 0..b {
                let r = point_slice[b - 1 - j];
                for i in 0..size {
                    evals[i + size] = evals[i] * r;
                    evals[i] = evals[i] * (F::one() - r);
                }
                size *= 2;
            }
            evals
        };

        let chi = compute_eq_evals(point);
        let mut acc = F::zero();
        for (idx, &basis) in chi.iter().enumerate() {
            let entry_u64 = tables_2d[[subtable_index, idx]];
            acc += F::from(entry_u64) * basis;
        }
        acc
    }

    /// GeLU의 exp(sum(contribs))는 product(exp(contrib))와 같으므로 곱셈 사용
    fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C, M>>::NUM_MEMORIES]) -> F {
        vals.iter().product()
    }

    /// g 다항식의 차수는 세그먼트 개수 C와 동일함
    fn g_poly_degree() -> usize {
        C
    }
}