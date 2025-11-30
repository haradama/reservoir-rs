#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod henon;
pub mod mackey_glass;
pub mod narma;

#[cfg(feature = "std")]
pub type RngType = rand::rngs::StdRng;

#[cfg(not(feature = "std"))]
pub type RngType = rand::rngs::SmallRng;
