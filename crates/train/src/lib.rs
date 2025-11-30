#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod esn;
pub mod float;
pub mod input;
pub mod readout;
pub mod reservoir;
pub mod trainer;

pub use esn::{ESNBuilder, EchoStateNetwork};
pub use readout::RidgeReadout;
pub use reservoir::DenseReservoir;
pub use trainer::RidgeTrainer;

#[cfg(feature = "std")]
pub type RngType = rand::rngs::StdRng;

#[cfg(not(feature = "std"))]
pub type RngType = rand::rngs::SmallRng;
