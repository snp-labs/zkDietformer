//! zkSNARK circuits and gadgets for Transformer-style neural networks
//! components for building zero-knowledge proofs of Transformer models,
//! including matrix multiplication, linear layers, and activation functions.
#![deny(missing_docs)]

/// High-level zkSNARK circuits for Transformer models.
pub mod circuits;

/// Reusable low-level constraint gadgets used by circuits.
pub mod gadgets;
