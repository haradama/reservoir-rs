#![cfg_attr(not(feature = "std"), no_std)]
//! Core traits and utilities for reservoir computing / Echo State Networks.
//!
//! # Design
//! This crate defines small, reusable building blocks:
//! - [`Reservoir`]: state transition (input -> internal state)
//! - [`Readout`]: mapping from state -> output
//! - [`Trainer`]: fitting readout parameters from sequences
//!
//! # Feature flags
//! - `std` (default): enables standard library support (implies `alloc`).
//! - `alloc`: enables heap-backed vectors/matrices via `nalgebra`.
//! - `libm`: provides the floating-point math (`tanh`, `powi`, ...) used by
//!   [`Scalar`] and [`metrics`] on `no_std` targets.
//!
//! A floating-point backend is **always required**: enable either `std`
//! (default) or `libm`. The `alloc` feature only adds heap-backed types and does
//! **not** provide math functions, so `no_std` builds must enable `libm`
//! (optionally together with `alloc`). Selecting neither `std` nor `libm`
//! triggers a compile error with this guidance.
//!
//! Note: Many concrete implementations live in `reservoir-infer` / `reservoir-train`.
//!
//! # State layout convention
//! Implementations in this workspace commonly expose an **extended state**
//! `[1, input..., reservoir_state...]` (bias + input + reservoir).
//! See `reservoir-infer` reservoirs for details.

// A floating-point backend is mandatory: `Scalar::activation` and the `metrics`
// module rely on `num_traits::Float`, which is only available when `num-traits`
// is built with either `std` or `libm`. Fail early with an actionable message
// instead of the cryptic "unresolved import `num_traits::Float`".
#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!(
    "reservoir-core needs a floating-point backend: enable the default `std` feature, \
     or the `libm` feature for `no_std` targets. The `alloc` feature only adds heap-backed \
     matrix/vector types and does not provide the math functions used by `Scalar`/`metrics`."
);

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod metrics;
pub mod prelude;
pub mod readout;
pub mod reservoir;
pub mod trainer;
pub mod types;

pub use metrics::*;
pub use readout::*;
pub use reservoir::*;
pub use trainer::*;
pub use types::*;
