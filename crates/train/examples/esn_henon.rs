use reservoir_train::data::henon::{HenonMap, HenonParams};
use reservoir_train::ESNBuilder;
use reservoir_train::{rmse, rsquare};

fn main() {
    let params = HenonParams {
        a: 1.4,
        b: 0.3,
        steps: 3000,
        ..Default::default()
    };
    let mut henon = HenonMap::new(params);

    let data_raw = henon.generate();
    let data: Vec<f32> = data_raw.iter().map(|&v| v as f32).collect();

    let inputs: Vec<Vec<f32>> = data[..data.len() - 1].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<f32>> = data[1..].iter().map(|&v| vec![v]).collect();

    let train_len = 2000;
    let inputs_train = &inputs[0..train_len];
    let targets_train = &targets[0..train_len];

    let inputs_test = &inputs[train_len..];
    let targets_test = &targets[train_len..];

    let mut esn = ESNBuilder::new(1, 1)
        .units(200)
        .spectral_radius(1.5)
        .leaking_rate(0.8)
        .input_scaling(1.5)
        .seed(1)
        .build();
    esn.fit(inputs_train, targets_train, 1e-6, 50);

    let preds: Vec<f32> = inputs_test
        .iter()
        .map(|u| esn.predict(u.as_slice())[0])
        .collect();

    let y_true: Vec<f32> = targets_test.iter().map(|v| v[0]).collect();

    println!("--------------------------------------------------");
    println!("RMSE : {:.6}", rmse(&y_true, &preds));
    println!("R^2  : {:.6}", rsquare(&y_true, &preds));
    println!("--------------------------------------------------");
}
