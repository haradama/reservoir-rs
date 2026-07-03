//! Readout implementations for ESN inference.
//!
//! Readouts map the (extended) reservoir state into an output vector.
//! In this crate, dynamic readouts are linear maps using `nalgebra::DMatrix`,
//! while the static readout uses `nalgebra::SMatrix` for `no_std`.
//!
//! Note: These types do not train. You provide the weight matrix.

#[cfg(feature = "alloc")]
pub mod lasso;
#[cfg(feature = "alloc")]
pub mod linear;
#[cfg(feature = "alloc")]
pub mod ridge;
pub mod static_readout;

#[cfg(feature = "alloc")]
pub use linear::LinearReadout;
// `RidgeReadout` / `LassoReadout` are backwards-compatible aliases for `LinearReadout`.
#[cfg(feature = "alloc")]
pub use lasso::LassoReadout;
#[cfg(feature = "alloc")]
pub use ridge::RidgeReadout;
pub use static_readout::StaticReadout;
