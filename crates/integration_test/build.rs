use reservoir_datasets::mackey_glass::{MackeyGlass, MackeyGlassParams};
use reservoir_train::{ESNBuilder, ESNFitRidge, StaticModelGenerator};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let memory_x_filename = if target.starts_with("thumbv7m-") {
        "memory_lm3s6965.x"
    } else if target.starts_with("thumbv7em-") {
        "memory_mps2.x"
    } else if target.starts_with("thumbv6m-") {
        "memory_microbit.x"
    } else if target.starts_with("riscv32") {
        "memory_virt.x"
    } else {
        ""
    };

    if !memory_x_filename.is_empty() {
        let source_path = PathBuf::from("boards").join(memory_x_filename);
        let dest_path = out_dir.join("memory.x");
        fs::copy(&source_path, &dest_path).expect("failed to copy memory.x");
        println!("cargo:rustc-link-search={}", out_dir.display());
        println!("cargo:rerun-if-changed=boards/{}", memory_x_filename);
    }

    generate_weights(&out_dir);
    println!("cargo:rerun-if-changed=build.rs");
}

fn generate_weights(out_dir: &PathBuf) {
    const RESERVOIR_SIZE: usize = 50;
    const INPUT_DIM: usize = 1;
    const OUTPUT_DIM: usize = 1;
    const TRAIN_LEN: usize = 1000;
    const TEST_LEN: usize = 20;

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
    let data_raw = mg.generate();
    let data: Vec<f32> = data_raw.iter().map(|&v| v as f32).collect();

    let inputs: Vec<Vec<f32>> = data[..data.len() - 1].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<f32>> = data[1..].iter().map(|&v| vec![v]).collect();

    let inputs_train = &inputs[0..TRAIN_LEN];
    let targets_train = &targets[0..TRAIN_LEN];
    
    let inputs_test = &inputs[TRAIN_LEN..TRAIN_LEN + TEST_LEN];
    let targets_test = &targets[TRAIN_LEN..TRAIN_LEN + TEST_LEN];

    let mut esn = ESNBuilder::new(INPUT_DIM, OUTPUT_DIM)
        .units(RESERVOIR_SIZE)
        .spectral_radius(0.95)
        .leaking_rate(0.5)
        .seed(42)
        .connectivity(3)
        .input_connectivity(1)
        .build_sparse();

    esn.fit(inputs_train, targets_train, 1e-6, 100);

    let mut code = StaticModelGenerator::generate_sparse_code(&esn)
        .expect("Failed to generate model code");

    let test_inputs_flat: Vec<String> = inputs_test.iter().map(|v| format!("{:.8}", v[0])).collect();
    let test_targets_flat: Vec<String> = targets_test.iter().map(|v| format!("{:.8}", v[0])).collect();

    code.push_str(&format!(
        "pub const TEST_LEN: usize = {};\n",
        TEST_LEN
    ));
    code.push_str(&format!(
        "pub const TEST_INPUTS: [f32; {}] = [{}];\n",
        TEST_LEN, test_inputs_flat.join(", ")
    ));
    code.push_str(&format!(
        "pub const TEST_TARGETS: [f32; {}] = [{}];\n",
        TEST_LEN, test_targets_flat.join(", ")
    ));

    let dest_path = out_dir.join("weights.rs");
    fs::write(&dest_path, code).expect("Failed to write weights.rs");
}
