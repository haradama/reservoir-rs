use nalgebra::RealField;
use num_traits::Float;
use rand::distributions::uniform::SampleUniform;

pub trait RealScalar: reservoir_core::types::Scalar + RealField + Float + SampleUniform {}

impl<T> RealScalar for T where T: reservoir_core::types::Scalar + nalgebra::RealField + SampleUniform
{}