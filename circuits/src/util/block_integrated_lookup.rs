//! Lasso lookup helpers for the block-integrated circuit (Version B).
//!
//! Re-exports the shared lookup functions from `model_integrated_lookup`,
//! which are used by both Version A and Version B circuits.

pub use crate::util::model_integrated_lookup::{
    make_lasso_label,
    run_layer_attn_lasso,
    run_layer_mlp_lasso,
};
