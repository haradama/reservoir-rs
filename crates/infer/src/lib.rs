#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod esn;
pub mod readout;
pub mod reservoir;

pub use esn::EchoStateNetwork;
pub use readout::{LassoReadout, RidgeReadout};
pub use reservoir::DenseReservoir;
