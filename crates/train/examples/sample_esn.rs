use reservoir_train::mackey_glass::{MackeyGlass, MackeyGlassParams};
use reservoir_train::ESNBuilder;
use reservoir_train::{rmse, rsquare};

fn main() {
    let mut mg = MackeyGlass::new(MackeyGlassParams {
        a: 0.2,
        b: 0.1,
        n: 10,
        tau: 17,
        x0: 1.2,
        h: 0.1,
        steps: 2000,
        seed: Some(42),
        history: None,
    });
    let data_f64 = mg.generate();
    let data: Vec<f32> = data_f64.into_iter().map(|v| v as f32).collect();

    let inputs: Vec<Vec<f32>> = data[..data.len() - 1].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<f32>> = data[1..].iter().map(|&v| vec![v]).collect();

    let mut esn = ESNBuilder::new(1)
        .units(200)
        .spectral_radius(0.9)
        .leaking_rate(0.8)
        .input_scaling(1.0)
        .seed(1)
        .build();
    esn.fit(&inputs, &targets, 1e-6);

    let preds: Vec<f32> = inputs
        .iter()
        .map(|u| {
            use nalgebra::DVector;
            esn.predict(&DVector::from_vec(u.clone()))[0]
        })
        .collect();

    let y_true: Vec<f32> = targets.iter().map(|v| v[0]).collect();
    println!("RMSE : {:.6}", rmse(&y_true, &preds));
    println!("R^2  : {:.6}", rsquare(&y_true, &preds));
}