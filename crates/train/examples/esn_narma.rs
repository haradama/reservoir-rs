use reservoir_train::narma::{Narma, NarmaParams};
use reservoir_train::ESNBuilder;
use reservoir_train::{rmse, rsquare};

fn main() {
    let params = NarmaParams {
        n: 10,
        steps: 5000, 
        seed: Some(42),
        ..Default::default()
    };
    let mut narma = Narma::new(params);
    let (raw_inputs, raw_targets) = narma.generate();

    let input_vec: Vec<f32> = raw_inputs.iter().map(|&v| v as f32).collect();
    let target_vec: Vec<f32> = raw_targets.iter().map(|&v| v as f32).collect();

    let inputs: Vec<Vec<f32>> = input_vec.iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<f32>> = target_vec.iter().map(|&v| vec![v]).collect();

    let train_len = 3000;
    let inputs_train = &inputs[0..train_len];
    let targets_train = &targets[0..train_len];
    
    let inputs_test = &inputs[train_len..];
    let targets_test = &targets[train_len..];

    let mut esn = ESNBuilder::new(1, 1) 
        .units(500)            
        .spectral_radius(0.95)
        .leaking_rate(0.3)
        .input_scaling(0.2)
        .seed(1234)
        .build();

    esn.fit(inputs_train, targets_train, 1e-6, 500);

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