#[cfg(feature = "alloc")]
pub mod dense;
#[cfg(feature = "alloc")]
pub mod sparse;

pub mod static_reservoir;
pub mod static_sparse_reservoir;

#[cfg(feature = "alloc")]
pub use dense::DenseReservoir;
#[cfg(feature = "alloc")]
pub use sparse::{CsrMatrix, SparseReservoir};

pub use static_reservoir::StaticReservoir;
pub use static_sparse_reservoir::{StaticCsrMatrix, StaticSparseReservoir};
