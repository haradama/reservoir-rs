use reservoir_core::types::Input;
use reservoir_dense::ESNBuilder;

fn main() {
    let timesteps = 200;
    let data: Vec<f32> = (0..timesteps).map(|t| ((t as f32) * 0.05).sin()).collect();

    let mut esn = ESNBuilder::new(1)
        .units(50)
        .spectral_radius(0.9)
        .seed(0)
        .build();

    for u in data.iter() {
        let input = Input::from_vec(vec![*u]);
        let y = esn.predict(&input);
        println!("{:.3} -> {:.3}", u, y[0]);
    }
}
