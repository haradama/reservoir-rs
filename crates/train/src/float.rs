use nalgebra::RealField;
use rand::distributions::uniform::SampleUniform;

pub trait RealScalar: reservoir_core::types::Scalar + RealField + SampleUniform {}

impl<T> RealScalar for T where T: reservoir_core::types::Scalar + nalgebra::RealField + SampleUniform
{}
