use ndarray::s;
use ndarray::{Array1, Array2, Axis};
use reservoir_core::{reservoir::Reservoir, trainer::Trainer};

use super::{DenseReservoir, RidgeReadout};

pub struct RidgeTrainer {
    pub ridge: f32,
}
impl Default for RidgeTrainer {
    fn default() -> Self {
        Self { ridge: 1e-6 }
    }
}

impl Trainer<DenseReservoir<f32>, RidgeReadout<f32>, f32> for RidgeTrainer {
    type Error = ();

    fn fit(
        &mut self,
        reservoir: &mut DenseReservoir<f32>,
        readout: &mut RidgeReadout<f32>,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> Result<(), Self::Error> {
        let n = inputs.len();
        let units = reservoir.dim();

        // collect states & teachers
        let mut x = Array2::<f32>::zeros((n, units));
        let mut y = Array1::<f32>::zeros(n);

        for (i, (u_raw, t)) in inputs.iter().zip(targets).enumerate() {
            let state = reservoir.step(&Array1::from_vec(u_raw.clone()));
            x.slice_mut(s![i, ..]).assign(state);
            y[i] = t[0];
        }

        let xt = x.view();
        let gram = xt.t().dot(&xt) + Array2::<f32>::eye(units) * self.ridge;
        let mut a = gram.to_owned();
        let mut inv = Array2::<f32>::eye(units);
        for i in 0..units {
            let pivot = a[[i, i]];
            for j in 0..units {
                a[[i, j]] /= pivot;
                inv[[i, j]] /= pivot;
            }
            for k in 0..units {
                if k == i {
                    continue;
                }
                let factor = a[[k, i]];
                for j in 0..units {
                    a[[k, j]] -= factor * a[[i, j]];
                    inv[[k, j]] -= factor * inv[[i, j]];
                }
            }
        }
        let yxt = y.view().insert_axis(Axis(0)).dot(&xt);
        let w = yxt.dot(&inv);

        readout.set_weights(w);
        Ok(())
    }
}
