use crate::cc_snark::linker::{
    matrix::SparseMatrix,
    snark::{LinkSnark, PP},
};

use ark_ec::pairing::Pairing;
use ark_relations::r1cs::SynthesisError;

use super::Linker;

pub fn generate_cp_arithm_relation<E: Pairing>(
    msg_len: usize,
    ck: Vec<E::G1Affine>,
    snark_ck: Vec<E::G1Affine>,
) -> SparseMatrix<E::G1Affine> {
    let l = 3;
    let t = msg_len + 5;

    let mut snark_ck = snark_ck.clone();
    snark_ck.truncate(msg_len + 3);

    let mut crs = SparseMatrix::new(l, t);

    // cm_u
    crs.insert_row_slice(0, 0, &vec![ck[0]]);
    for i in 0..msg_len {
        crs.insert_row_slice(0, i + 3, &vec![ck[i + 1]]);
    }

    // cm_sr
    crs.insert_row_slice(1, 1, &vec![ck[0]]);
    crs.insert_row_slice(1, msg_len + 3, &vec![ck[1], ck[2]]);

    // snark_ck
    crs.insert_row_slice(l - 1, 2, &snark_ck);

    crs
}

pub fn generate_cp_bound_relation<E: Pairing>(
    msg_len: usize,
    ck: Vec<E::G1Affine>,
    snark_ck: Vec<E::G1Affine>,
) -> SparseMatrix<E::G1Affine> {
    let l = 2;
    let t = msg_len + 2;

    let mut snark_ck = snark_ck.clone();
    snark_ck.truncate(msg_len + 1);

    let mut crs = SparseMatrix::new(l, t);

    // cm_u
    crs.insert_row_slice(0, 0, &vec![ck[0]]);
    for i in 0..msg_len {
        crs.insert_row_slice(0, i + 2, &vec![ck[i + 1]]);
    }

    // snark_ck
    crs.insert_row_slice(l - 1, 1, &snark_ck);

    crs
}

pub fn generate_cp_ctt_relation<E: Pairing>(
    msg_len: usize,
    ck: Vec<E::G1Affine>,
    snark_ck: Vec<E::G1Affine>,
) -> SparseMatrix<E::G1Affine> {
    let l = 3;
    let t = 2 * msg_len + 3;

    let mut crs = SparseMatrix::new(l, t);

    // cm_u
    crs.insert_row_slice(0, 0, &vec![ck[0]]);
    for i in 0..msg_len {
        crs.insert_row_slice(0, i + 3, &vec![ck[i + 1]]);
    }

    // cm_a
    crs.insert_row_slice(1, 1, &vec![ck[0]]);
    for i in 0..msg_len {
        crs.insert_row_slice(1, i + 3 + msg_len, &vec![ck[i + 1]]);
    }

    // snark_ck
    crs.insert_row_slice(l - 1, 2, &snark_ck);

    crs
}

pub fn generate_cp_wt_relation<E: Pairing>(
    msg_len: usize,
    ck: Vec<E::G1Affine>,
    snark_ck: Vec<E::G1Affine>,
) -> SparseMatrix<E::G1Affine> {
    let l = 4;
    let t = 3 * msg_len + 4;

    let mut crs = SparseMatrix::new(l, t);

    // cm_u
    crs.insert_row_slice(0, 0, &vec![ck[0]]);
    for i in 0..msg_len {
        crs.insert_row_slice(0, i + 4, &vec![ck[i + 1]]);
    }

    // cm_a
    crs.insert_row_slice(1, 1, &vec![ck[0]]);
    for i in 0..msg_len {
        crs.insert_row_slice(1, i + 4 + msg_len, &vec![ck[i + 1]]);
    }

    // cm_z
    crs.insert_row_slice(2, 2, &vec![ck[0]]);
    for i in 0..msg_len {
        crs.insert_row_slice(2, i + 4 + 2 * msg_len, &vec![ck[i + 1]]);
    }

    // snark_ck
    crs.insert_row_slice(l - 1, 3, &snark_ck);

    crs
}

pub fn generate_cp_witness<E: Pairing>(
    r: Vec<E::ScalarField>,
    u: Vec<E::ScalarField>,
    snark_witness: Vec<E::ScalarField>,
) -> Result<<LinkSnark<E> as Linker<E>>::Witness, SynthesisError> {
    let mut witness_vec = Vec::new();
    witness_vec.extend_from_slice(r.as_slice());
    witness_vec.extend_from_slice(snark_witness.as_slice());
    witness_vec.extend_from_slice(u.as_slice());

    Ok(witness_vec)
}

pub fn generate_cp_instance<E: Pairing>(
    cm: Vec<E::G1Affine>,
    snark_cm: E::G1Affine,
) -> Result<<LinkSnark<E> as Linker<E>>::Instance, SynthesisError> {
    let mut instance_vec = cm;

    instance_vec.push(snark_cm);

    Ok(instance_vec)
}


pub fn generate_cp_1_to_n_vector_relation<E: Pairing>(
    n_indiv: usize,                    
    msg_lens: Vec<usize>,              
    ck_joint: Vec<E::G1Affine>,        
    cks_indiv: Vec<Vec<E::G1Affine>>,  
) -> SparseMatrix<E::G1Affine> {
    let total_msg_len: usize = msg_lens.iter().sum();
    let l = 1 + n_indiv;                
    let t = (1 + n_indiv) + total_msg_len; 
    
    let mut crs = SparseMatrix::new(l, t);

    crs.insert_val(0, 0, &ck_joint[0]); 
    for i in 0..total_msg_len {
        crs.insert_val(0, (1 + n_indiv) + i, &ck_joint[i + 1]);
    }

    let mut col_offset = 0;
    for j in 0..n_indiv {
        let row = 1 + j;
        let r_col = 1 + j; 
        
        crs.insert_val(row, r_col, &cks_indiv[j][0]);
        
        for i in 0..msg_lens[j] {
            let v_col = (1 + n_indiv) + col_offset + i;
            crs.insert_val(row, v_col, &cks_indiv[j][1 + i]);
        }
        col_offset += msg_lens[j];
    }

    crs
}


pub fn generate_cp_factored_relation<E: Pairing>(
    n_lasso: usize,                     
    l_rows_vec: Vec<usize>,             
    r_cols_vec: Vec<usize>,             
    ck_joint: Vec<E::G1Affine>,         
    cks_lasso_vec: Vec<Vec<E::G1Affine>>, 
) -> SparseMatrix<E::G1Affine> {
    
    let total_msg_len: usize = l_rows_vec.iter().zip(r_cols_vec.iter()).map(|(l, r)| l * r).sum();
    let total_lasso_points: usize = l_rows_vec.iter().sum();
    
    let l_rows = 1 + total_lasso_points;                
    let t_cols = (1 + total_lasso_points) + total_msg_len; 
    
    let mut crs = SparseMatrix::new(l_rows, t_cols);

    // 1. 통합 커밋먼트 정의 (Row 0): LegoGroth16
    crs.insert_val(0, 0, &ck_joint[0]); 
    for i in 0..total_msg_len {
        crs.insert_val(0, (1 + total_lasso_points) + i, &ck_joint[i + 1]); 
    }

    // 2. Lasso Row 커미션들 정의 (Row 1 ~ l-1)
    let mut current_row_offset = 1;
    let mut current_data_col_offset = 0;
    
    for j in 0..n_lasso {
        let L = l_rows_vec[j]; 
        let R = r_cols_vec[j]; 
        let ck_lasso = &cks_lasso_vec[j]; 
        
        for row_idx in 0..L {
            let row = current_row_offset + row_idx;
            let r_col = current_row_offset + row_idx; 
            
            crs.insert_val(row, r_col, &ck_lasso[0]);
            
            for k in 0..R {
                let data_idx = row_idx * R + k;
                let v_col = (1 + total_lasso_points) + current_data_col_offset + data_idx;
                crs.insert_val(row, v_col, &ck_lasso[k + 1]);
            }
        }
        
        current_row_offset += L;
        current_data_col_offset += L * R;
    }

    crs
}


pub fn generate_cp_factored_witness<E: Pairing>(
    rs: Vec<E::ScalarField>, 
    u: Vec<E::ScalarField>,  
) -> Vec<E::ScalarField> {
    let mut witness = rs;
    witness.extend(u);
    witness
}


pub fn generate_cp_factored_instance<E: Pairing>(
    cm_joint: E::G1Affine,
    lasso_cms_all: Vec<E::G1Affine>, 
) -> Vec<E::G1Affine> {
    let mut instance = vec![cm_joint];
    instance.extend(lasso_cms_all);
    instance
}