//! Reservoir implementations for ESN inference.
//!
//! This module provides:
//! - Dynamic reservoirs (behind `alloc`) using `nalgebra::DVector/DMatrix`
//! - Static reservoirs designed for `no_std` use, using `nalgebra::SVector/SMatrix`
//!
//! Many reservoirs here return an **extended state** vector typically laid out as:
//! `[bias(1), input(input_dim), reservoir_state(units)]`.
//! Readouts are expected to be sized accordingly.

#[cfg(feature = "alloc")]
pub mod dense;
#[cfg(feature = "alloc")]
pub mod sparse;

pub mod static_reservoir;
pub mod static_sparse_reservoir;

#[cfg(feature = "alloc")]
pub use dense::DenseReservoir;
#[cfg(feature = "alloc")]
pub use sparse::{CsrMatrix, SparseReservoir};

pub use static_reservoir::StaticReservoir;
pub use static_sparse_reservoir::{StaticCsrMatrix, StaticSparseReservoir};
