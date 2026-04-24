/// Baseline (non-Freivalds) version of the MLP circuit.
///
/// The MLP circuit (`mlp_circuit.rs`) does NOT use Freivalds-based matrix
/// multiplication. It uses `enforce_linear`, which already performs naive
/// element-wise matmul with constant weights (Q_w_diff). Therefore this
/// baseline version is structurally identical to the original.
///
/// Difference from original: NONE — this file re-exports the original
/// circuit unchanged because it already uses naive matmul.
pub use crate::circuits::mlp_circuit::*;
