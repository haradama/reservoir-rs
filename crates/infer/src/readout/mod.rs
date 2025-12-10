#[cfg(feature = "alloc")]
pub mod lasso;
#[cfg(feature = "alloc")]
pub mod ridge;
pub mod static_readout;

#[cfg(feature = "alloc")]
pub use lasso::LassoReadout;
#[cfg(feature = "alloc")]
pub use ridge::RidgeReadout;
pub use static_readout::StaticReadout;
