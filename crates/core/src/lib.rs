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
//! - `std` (default): enables standard library support.
//! - `alloc`: enables heap-backed vectors/matrices via `nalgebra`.
//! - `libm`: enables `libm` math for targets without `std` floating-point support.
//!
//! Note: Many concrete implementations live in `reservoir-infer` / `reservoir-train`.
//!
//! # State layout convention
//! Implementations in this workspace commonly expose an **extended state**
//! `[1, input..., reservoir_state...]` (bias + input + reservoir).
//! See `reservoir-infer` reservoirs for details.

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
