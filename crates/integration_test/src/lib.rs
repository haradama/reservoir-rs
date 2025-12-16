#![no_std]

pub mod weights {
    include!(concat!(env!("OUT_DIR"), "/weights.rs"));
}

pub mod common;
