use fixed::types::I20F12;
use reservoir_core::{rmse, rsquare};
use reservoir_datasets::mackey_glass::{MackeyGlass, MackeyGlassParams};
use reservoir_train::ESNBuilder;

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

    let inputs_raw: Vec<Vec<f64>> = data_f64[..data_f64.len() - 1]
        .iter()
        .map(|&v| vec![v])
        .collect();
    let targets_raw: Vec<Vec<f64>> = data_f64[1..].iter().map(|&v| vec![v]).collect();

    let inputs_fixed: Vec<Vec<MyFixed>> = inputs_raw
        .iter()
        .map(|v| v.iter().map(|&x| MyFixed::from_num(x)).collect())
        .collect();

    let targets_fixed: Vec<Vec<MyFixed>> = targets_raw
        .iter()
        .map(|v| v.iter().map(|&x| MyFixed::from_num(x)).collect())
        .collect();

    let mut esn = ESNBuilder::<MyFixed>::new(1, 1)
        .units(100)
        .spectral_radius(MyFixed::from_num(0.9))
        .input_scaling(MyFixed::from_num(1.0))
        .leaking_rate(MyFixed::from_num(0.8))
        .seed(42)
        .build_lasso();

    let alpha = MyFixed::from_num(0.0001);
    let tol = MyFixed::from_num(0.001);
    let max_iter = 1000;
    let washout = 100;

    esn.fit_lasso(&inputs_fixed, &targets_fixed, alpha, max_iter, tol, washout);

    let preds_fixed: Vec<f64> = inputs_fixed
        .iter()
        .map(|u| {
            let out = esn.predict(u.as_slice());
            out[0].to_num::<f64>()
        })
        .collect();

    let y_true: Vec<f64> = targets_raw.iter().map(|v| v[0]).collect();

    println!("--------------------------------------------------");
    println!("Evaluation Results (Quantized Training & Inference)");
    println!("RMSE : {:.6}", rmse(&y_true, &preds_fixed));
    println!("R^2  : {:.6}", rsquare(&y_true, &preds_fixed));
    println!("--------------------------------------------------");
}
