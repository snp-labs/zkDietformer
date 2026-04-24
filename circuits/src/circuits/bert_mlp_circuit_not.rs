/// Baseline (non-Freivalds) version of the BERT MLP circuit.
///
/// The BERT MLP circuit (`bert_mlp_circuit.rs`) is a type alias for
/// `NystromMLPCircuit` which uses `enforce_linear` — already naive
/// element-wise matmul with constant weights. No Freivalds is used.
///
/// Difference from original: NONE — this file re-exports the original
/// circuit unchanged because it already uses naive matmul.
pub use crate::circuits::bert_mlp_circuit::*;
