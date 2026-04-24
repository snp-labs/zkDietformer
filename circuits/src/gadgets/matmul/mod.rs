#![allow(missing_docs)]
//! Matrix multiplication and linear layer gadgets.
//!
//! This module provides matrix and vector level gadgets

/// Constraint implementations for matrix multiplication gadgets.
pub mod constraints;
pub mod quantized;
pub mod naive_quantized;