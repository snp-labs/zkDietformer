use ark_ff::PrimeField;
use ark_std::log2;
use ndarray_npy::NpzReader;
use std::fs::File;
use std::path::Path;
use std::sync::OnceLock; 
use ndarray::{Array1, Array2, Ix2, Ix1};
use super::SubtableStrategy;

pub enum RMSNormSubtableStrategy {}

static RMSNORM_CACHE: OnceLock<Array1<u64>> = OnceLock::new();

// K_SEGENTS와 LIMBS로 바꾸지 않아 1차원 고정
fn get_or_init_rmsnorm_cache(m: usize) -> &'static Array1<u64> {
    RMSNORM_CACHE.get_or_init(|| {
        let lookup_npz_path = "outputs/rmsnorm_tables.npz".to_string();
        let file = File::open(Path::new(&lookup_npz_path))
            .expect("RMSNorm table file not found");
        let mut npz = NpzReader::new(file).expect("failed to create NpzReader");

        let tables_arr_dyn = npz.by_name::<ndarray::OwnedRepr<u64>, ndarray::IxDyn>("table.npy")
            .expect("npz does not contain table");

        let tables_1d = tables_arr_dyn
            .into_dimensionality::<Ix1>()
            .expect("rmsnorm tables must be 1D");

        assert_eq!(tables_1d.len(), m, "RMSNorm table size mismatch");
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
        let tables_1d = get_or_init_rmsnorm_cache(M);
        
        // 각 메모리가 어떤 서브테이블을 참조하는지 1:1로 매핑
        std::array::from_fn(|i| {
            tables_1d.iter().map(|&val| F::from(val)).collect()
        })
    }


    fn evaluate_subtable_mle(_subtable_index: usize, point: &Vec<F>) -> F {
        let table = get_or_init_rmsnorm_cache(M);
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

    fn setup_rmsnorm_test_npz() {
        let dir = "outputs";
        let path = "outputs/rmsnorm_tables.npz";
        fs::create_dir_all(dir).unwrap();

        if !Path::new(path).exists() {
            let file = File::create(path).unwrap();
            let mut writer = NpzWriter::new(file);
            
            let data = Array1::<u64>::from_shape_fn(WIDTH, |i| (i + 100) as u64);
            writer.add_array("table.npy", &data).unwrap();
            writer.finish().unwrap();
        }
    }

    #[test]
    fn rmsnorm_materialization_parity() {
        setup_rmsnorm_test_npz();
        
        // materialize_subtables 호출
        let materialized: [Vec<Fr>; C_TEST] =
            <RMSNormSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::materialize_subtables();

        let cache = get_or_init_rmsnorm_cache(WIDTH);
        
        assert_eq!(materialized[0].len(), WIDTH);
        for i in 0..WIDTH {
            assert_eq!(materialized[0][i], Fr::from(cache[i]));
        }
    }

    #[test]
    fn rmsnorm_mle_eval_test() {
        setup_rmsnorm_test_npz();
        
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
    fn rmsnorm_combine_is_identity() {
        let val = Fr::from(12345u64);
        let vals: [Fr; C_TEST] = [val];
        
        let combined = <RMSNormSubtableStrategy as SubtableStrategy<Fr, C_TEST, WIDTH>>::combine_lookups(&vals);
        assert_eq!(combined, val);
    }
}