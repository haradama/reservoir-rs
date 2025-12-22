//! Readout training algorithms.
//!
//! Trainers implement [`reservoir_core::trainer::Trainer`] and update
//! readout weights given a reservoir and training sequences.
//!
//! - [`RidgeTrainer`]: closed-form ridge regression via `LU` solve
//! - [`LassoTrainer`]: coordinate descent with soft-thresholding

pub mod lasso;
pub mod ridge;

pub use lasso::LassoTrainer;
pub use ridge::RidgeTrainer;
