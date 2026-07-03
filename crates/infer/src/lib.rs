#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

//! Inference-time building blocks for Echo State Networks (ESN).
//!
//! `reservoir-infer` provides inference-oriented components for Echo State Networks:
//! reservoirs (dense/sparse, dynamic/static) and readouts.
//! Training utilities live in other crates; this crate focuses on **forward/predict**.
//!
//! # Features
//! - `std` (default): Enables `std` support.
//! - `alloc` (default): Enables heap allocations and dynamic-sized types (`nalgebra::DVector/DMatrix`).
//! - `libm`: Enables `libm`-backed math (useful for `no_std` targets).
//!
//! By default, `std` + `alloc` are enabled.
//!
//! # Module overview
//! - [`esn`]: ESN wrappers (dynamic ESN behind `alloc`, and static ESN for `no_std`).
//! - [`reservoir`]: Dense/Sparse reservoirs (dynamic behind `alloc`) and static reservoirs.
//! - [`readout`]: Linear readouts (dynamic behind `alloc`) and static readout.
//!
//! # Quick start (dynamic ESN, requires `alloc`)
//! ```rust,ignore
//! use reservoir_infer::{EchoStateNetwork, DenseReservoir, RidgeReadout};
//! use nalgebra::{DMatrix, DVector};
//!
//! // Construct your reservoir weights and readout weights (training is out of scope here).
//! let w_in = DMatrix::<f64>::zeros(2, 2);
//! let w = DMatrix::<f64>::identity(2, 2);
//! let reservoir = DenseReservoir::create(w_in, w, 1.0, /*input_dim*/ 2, /*units*/ 2);
//!
//! let w_out = DMatrix::<f64>::zeros(1, 1 + 2 + 2); // output_dim x extended_state_dim
//! let readout = RidgeReadout::create(w_out);
//!
//! let mut esn = EchoStateNetwork::new(reservoir, readout);
//! let y = esn.predict(vec![0.1, -0.2]);
//! # let _ = y;
//! ```
//!
//! # Quick start (static ESN, `no_std` friendly)
//! ```rust,ignore
//! use reservoir_infer::{StaticESN, StaticReservoir, StaticReadout};
//! use nalgebra::{SMatrix, SVector};
//!
//! // Example dimensions
//! type S = f32;
//! const IN: usize = 2;
//! const N: usize = 2;
//! const EXT: usize = 1 + IN + N;
//! const OUT: usize = 1;
//!
//! let w_in: SMatrix<S, N, IN> = SMatrix::identity();
//! let w: SMatrix<S, N, N> = SMatrix::zeros();
//! let reservoir = reservoir_infer::reservoir::StaticReservoir::<S, IN, N, EXT>::create(w_in, w, 1.0);
//!
//! let w_out: SMatrix<S, OUT, EXT> = SMatrix::zeros();
//! let readout = reservoir_infer::readout::StaticReadout::<S, EXT, OUT>::create(w_out);
//!
//! let mut esn = StaticESN::new(reservoir, readout);
//! let x = SVector::<S, IN>::new(0.1, -0.2);
//! let y: SVector<S, OUT> = esn.predict::<IN, OUT, EXT>(&x);
//! # let _ = y;
//! ```

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod esn;
pub mod readout;
pub mod reservoir;

#[cfg(feature = "alloc")]
pub use esn::EchoStateNetwork;

#[cfg(feature = "alloc")]
pub use readout::{LassoReadout, LinearReadout, RidgeReadout};

#[cfg(feature = "alloc")]
pub use reservoir::{DenseReservoir, SparseReservoir};

pub use esn::StaticESN;
pub use readout::StaticReadout;
pub use reservoir::{StaticReservoir, StaticSparseReservoir};
