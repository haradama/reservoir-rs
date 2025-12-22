use nalgebra::RealField;
use rand::distributions::uniform::SampleUniform;

/// Scalar bounds used by training utilities.
///
/// Training needs:
/// - `reservoir_core::types::Scalar` for activation helpers / numeric ops
/// - `nalgebra::RealField` for linear algebra decompositions and norms
/// - `SampleUniform` for random initialization (builder)
pub trait RealScalar: reservoir_core::types::Scalar + RealField + SampleUniform {}

impl<T> RealScalar for T where T: reservoir_core::types::Scalar + nalgebra::RealField + SampleUniform
{}
