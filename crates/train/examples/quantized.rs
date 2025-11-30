use fixed::types::I20F12;
use reservoir_core::types::Scalar;
use reservoir_train::mackey_glass::{MackeyGlass, MackeyGlassParams};
use reservoir_train::{rmse, rsquare, ESNBuilder};

type MyFixed = I20F12;

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
    let data: Vec<f64> = data_f64;

    let inputs: Vec<Vec<f64>> = data[..data.len() - 1].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<f64>> = data[1..].iter().map(|&v| vec![v]).collect();

    let mut esn_float = ESNBuilder::new(1, 1)
        .units(100)
        .spectral_radius(0.9)
        .leaking_rate(0.8)
        .seed(42)
        .build();

    esn_float.fit(&inputs, &targets, 1e-6, 100);

    let float_weights = esn_float.readout.weights();
    let fixed_weights = float_weights.map(|v| MyFixed::from_f64_val(v));

    let mut esn_fixed = ESNBuilder::<MyFixed>::new(1, 1)
        .units(100)
        .spectral_radius(MyFixed::from_num(0.9))
        .leaking_rate(MyFixed::from_num(0.8))
        .seed(42)
        .build();

    esn_fixed.readout.set_weights(fixed_weights);

    let inputs_fixed: Vec<Vec<MyFixed>> = inputs
        .iter()
        .map(|v| v.iter().map(|&x| MyFixed::from_num(x)).collect())
        .collect();

    let preds_fixed: Vec<f64> = inputs_fixed
        .iter()
        .map(|u| {
            let out = esn_fixed.predict(u.as_slice());
            out[0].to_num::<f64>()
        })
        .collect();

    let y_true: Vec<f64> = targets.iter().map(|v| v[0]).collect();
    println!("--------------------------------------------------");
    println!("Evaluation Results (Fixed)");
    println!("RMSE : {:.6}", rmse(&y_true, &preds_fixed));
    println!("R^2  : {:.6}", rsquare(&y_true, &preds_fixed));
    println!("--------------------------------------------------");

    let preds_float: Vec<f64> = inputs
        .iter()
        .map(|u| esn_float.predict(u.as_slice())[0])
        .collect();

    println!("Evaluation Results (Float)");
    println!("RMSE : {:.6}", rmse(&y_true, &preds_float));
    println!("R^2  : {:.6}", rsquare(&y_true, &preds_float));
    println!("--------------------------------------------------");
}
