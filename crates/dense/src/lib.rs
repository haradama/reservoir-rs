mod esn;
mod readout;
mod reservoir;
pub use esn::{ESNBuilder, EchoStateNetwork};
pub use readout::RidgeReadout;
pub use reservoir::DenseReservoir;
pub mod trainer;
pub use trainer::RidgeTrainer;
