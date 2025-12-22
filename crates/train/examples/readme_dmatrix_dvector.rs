use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{distributions::Uniform, Rng, SeedableRng};

use reservoir_core::{mse, rmse, rsquare, Reservoir, Scalar, Trainer};
use reservoir_infer::{DenseReservoir, EchoStateNetwork, RidgeReadout};
use reservoir_train::RidgeTrainer;

fn main() {
    type S = f32;

    const IN: usize = 1;
    const OUT: usize = 1;

    let units: usize = 50;
    let leaking_rate: S = 0.3;
    let ridge: S = 1e-6;
    let washout: usize = 50;

    // --- build an ESN manually with DMatrix/DVector ---
    let mut rng = StdRng::seed_from_u64(42);
    let uni = Uniform::new(-0.5f32, 0.5f32);

    let mut rand_mat = |r: usize, c: usize| DMatrix::<S>::from_fn(r, c, |_, _| rng.sample(uni));

    let w_in: DMatrix<S> = rand_mat(units, IN) * S::from_f64_val(0.5);

    let mut w: DMatrix<S> = rand_mat(units, units);

    let w_f64 = w.map(|x| x as f64);
    let ev = w_f64.complex_eigenvalues();
    let current_rho = ev.iter().map(|c| c.norm()).fold(0.0f64, f64::max);
    let target_rho = 0.9f64;
    if current_rho > 0.0 {
        let scale = (target_rho / current_rho) as S;
        w *= scale;
    }

    let reservoir = DenseReservoir::create(w_in, w, leaking_rate, IN, units);

    let w_out = DMatrix::<S>::zeros(OUT, reservoir.dim());
    let readout = RidgeReadout::create(w_out);

    let mut esn = EchoStateNetwork::new(reservoir, readout);

    // --- toy time-series: u_t = sin(0.02 t), y_t = u_{t+1} ---
    let steps = 2000usize;
    let series: Vec<S> = (0..=steps).map(|t| ((t as f32) * 0.02).sin()).collect();

    let inputs: Vec<DVector<S>> = series[..steps]
        .iter()
        .map(|&v| DVector::from_element(IN, v))
        .collect();

    let targets: Vec<DVector<S>> = series[1..]
        .iter()
        .map(|&v| DVector::from_element(OUT, v))
        .collect();

    let train_len = 1500usize;

    // --- train readout (ridge) ---
    let mut trainer = RidgeTrainer { ridge };
    trainer
        .fit(
            &mut esn.reservoir,
            &mut esn.readout,
            &inputs[..train_len],
            &targets[..train_len],
            washout,
        )
        .expect("training failed");

    // --- evaluate ---
    let preds: Vec<S> = inputs[train_len..]
        .iter()
        .map(|u| esn.predict(u.clone())[0])
        .collect();

    let y_true: Vec<S> = targets[train_len..].iter().map(|v| v[0]).collect();

    println!("--------------------------------------------------");
    println!("MSE  : {:.6}", mse(&y_true, &preds));
    println!("RMSE : {:.6}", rmse(&y_true, &preds));
    println!("R²   : {:.6}", rsquare(&y_true, &preds));
    println!("--------------------------------------------------");
}
