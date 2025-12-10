#[cfg(feature = "alloc")]
pub mod dense;
pub mod static_reservoir;

#[cfg(feature = "alloc")]
pub use dense::DenseReservoir;
pub use static_reservoir::StaticReservoir;
