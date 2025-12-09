#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod esn;
pub mod float;
pub mod trainer;

pub use esn::{ESNBuilder, ESNFitLasso, ESNFitRidge};
pub use reservoir_infer;
pub use reservoir_infer::{DenseReservoir, EchoStateNetwork, LassoReadout, RidgeReadout};
pub use trainer::RidgeTrainer;

#[cfg(feature = "std")]
pub type RngType = rand::rngs::StdRng;

#[cfg(not(feature = "std"))]
pub type RngType = rand::rngs::SmallRng;
