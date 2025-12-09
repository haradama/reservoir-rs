#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod esn;
pub mod readout;
pub mod reservoir;

#[cfg(feature = "alloc")]
pub use esn::EchoStateNetwork;

#[cfg(feature = "alloc")]
pub use readout::{LassoReadout, RidgeReadout};

#[cfg(feature = "alloc")]
pub use reservoir::DenseReservoir;

pub use esn::StaticESN;
pub use readout::StaticReadout;
pub use reservoir::StaticReservoir;
