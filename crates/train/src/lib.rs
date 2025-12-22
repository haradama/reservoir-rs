#![cfg_attr(not(feature = "std"), no_std)]

//! Training utilities for Echo State Networks (ESN).
//!
//! `reservoir-train` provides training-time components for Echo State Networks:
//! - ridge / lasso solvers for linear readouts
//! - an [`ESNBuilder`] for quickly assembling common ESN configurations
//! - (optional, `std`) static code generation for embedded inference models
//!
//! This crate pairs with [`reservoir-infer`] (inference-time reservoirs/readouts).
//!
//! # Features
//! - `std` (default): Enables `std` and code generation utilities.
//! - `libm`: Enables `libm`-backed math for `no_std` targets (also affects dependencies).
//!
//! # High-level workflow
//! 1. Build an ESN (dense or sparse) with [`ESNBuilder`]
//! 2. Fit the readout using [`ESNFitRidge::fit`] or [`ESNFitLasso::fit_lasso`]
//! 3. Run inference using the same model (`predict`), or export weights for deployment.
//!
//! # Examples
//! See the `examples/` directory:
//! - `esn_henon.rs`
//! - `esn_mackey_glass.rs`
//! - `esn_narma.rs`

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[cfg(feature = "std")]
pub mod codegen;
pub mod esn;
pub mod float;
pub mod trainer;

#[cfg(feature = "std")]
pub use codegen::StaticModelGenerator;
pub use esn::{ESNBuilder, ESNFitLasso, ESNFitRidge};

/// Re-export `reservoir-infer` so downstream users can refer to a single crate.
///
/// This is convenient when `reservoir-train` is used as the training entry point.
pub use reservoir_infer;

pub use reservoir_infer::{DenseReservoir, EchoStateNetwork, LassoReadout, RidgeReadout};
pub use trainer::RidgeTrainer;

/// RNG type used internally by [`ESNBuilder`].
///
/// - With `std`: [`rand::rngs::StdRng`]
/// - Without `std`: [`rand::rngs::SmallRng`]
#[cfg(feature = "std")]
pub type RngType = rand::rngs::StdRng;

#[cfg(not(feature = "std"))]
pub type RngType = rand::rngs::SmallRng;
