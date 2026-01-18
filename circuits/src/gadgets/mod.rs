//! Constraint gadgets for Transformer-style neural network components.
//!
//! This module contains reusable low-level gadgets that implement
//! core linear algebra and non-linear operations used in Transformer models
//! such as matrix multiplication and activation functions.

/// Activation function gadgets (e.g., ReLU).
pub mod activation;

/// Dot product constraint gadgets.
pub mod dot_prod;

/// Matrix multiplication and linear layer gadgets.
pub mod matmul;

/// Softmax function gadgets.
pub mod softmax;