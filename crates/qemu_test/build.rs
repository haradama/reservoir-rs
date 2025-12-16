use reservoir_datasets::mackey_glass::{MackeyGlass, MackeyGlassParams};
use reservoir_train::{ESNBuilder, ESNFitRidge};
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
    const RESERVOIR_SIZE: usize = 10;
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
        .spectral_radius(0.9)
        .leaking_rate(0.5)
        .seed(42)
        .connectivity(2) 
        .input_connectivity(1)
        .build_sparse(); 

    esn.fit(inputs_train, targets_train, 1e-6, 100);

    
    
    let w_in = &esn.reservoir.w_in;
    let w_res = &esn.reservoir.w;
    
    
    let w_out = &esn.readout.w_out;
    let final_state = &esn.reservoir.res_state;
    let leaking_rate = esn.reservoir.leaking_rate;

    
    fn to_u16_vec(v: &[usize], name: &str) -> Vec<u16> {
        v.iter().map(|&x| {
            if x > u16::MAX as usize {
                panic!("{} exceeds u16::MAX. Decrease reservoir size or sparsity.", name);
            }
            x as u16
        }).collect()
    }

    let w_in_row_ptr = to_u16_vec(&w_in.row_ptr, "W_in row_ptr");
    let w_in_col_idx = to_u16_vec(&w_in.col_idx, "W_in col_idx");
    let w_in_values = &w_in.values; 

    let w_res_row_ptr = to_u16_vec(&w_res.row_ptr, "W_res row_ptr");
    let w_res_col_idx = to_u16_vec(&w_res.col_idx, "W_res col_idx");
    let w_res_values = &w_res.values;

    

    let mut code = String::new();

    code.push_str("
    code.push_str(&format!("pub const INPUT_DIM: usize = {};\n", INPUT_DIM));
    code.push_str(&format!("pub const RESERVOIR_SIZE: usize = {};\n", RESERVOIR_SIZE));
    code.push_str(&format!("pub const OUTPUT_DIM: usize = {};\n", OUTPUT_DIM));

    let ext_size = 1 + INPUT_DIM + RESERVOIR_SIZE;
    code.push_str(&format!("pub const EXTENDED_SIZE: usize = {};\n", ext_size));
    code.push_str(&format!("pub const TEST_LEN: usize = {};\n", TEST_LEN));
    code.push_str(&format!("pub const LEAKING_RATE: f32 = {:.8};\n\n", leaking_rate));

    
    code.push_str(&format!("pub const W_IN_NROWS: usize = {};\n", w_in.nrows)); 
    code.push_str(&format!("pub const W_IN_NCOLS: usize = {};\n", w_in.ncols));
    code.push_str(&format!("pub const W_RES_NROWS: usize = {};\n", w_res.nrows));
    code.push_str(&format!("pub const W_RES_NCOLS: usize = {};\n", w_res.ncols));

    
    code.push_str(&format!("pub const W_IN_NNZ: usize = {};\n", w_in_values.len()));
    code.push_str(&format!("pub const W_RES_NNZ: usize = {};\n\n", w_res_values.len()));

    fn fmt_u16_array(v: &[u16]) -> String {
        let elements: Vec<String> = v.iter().map(|x| format!("{}", x)).collect();
        format!("[{}]", elements.join(", "))
    }
    fn fmt_f32_array(v: &[f32]) -> String {
        let elements: Vec<String> = v.iter().map(|x| format!("{:.8}", x)).collect();
        format!("[{}]", elements.join(", "))
    }
    fn fmt_matrix_f32_data(mat: &nalgebra::DMatrix<f32>) -> String {
        let elements: Vec<String> = mat.iter().map(|v| format!("{:.8}", v)).collect();
        format!("[{}]", elements.join(", "))
    }
    fn fmt_vector_f32_data(vec: &nalgebra::DVector<f32>) -> String {
        let elements: Vec<String> = vec.iter().map(|v| format!("{:.8}", v)).collect();
        format!("[{}]", elements.join(", "))
    }

    
    code.push_str(&format!(
        "pub const W_IN_ROW_PTR: [u16; {}] = {};\n",
        w_in_row_ptr.len(), fmt_u16_array(&w_in_row_ptr)
    ));
    code.push_str(&format!(
        "pub const W_IN_COL_IDX: [u16; {}] = {};\n",
        w_in_col_idx.len(), fmt_u16_array(&w_in_col_idx)
    ));
    code.push_str(&format!(
        "pub const W_IN_VALUES: [f32; {}] = {};\n\n",
        w_in_values.len(), fmt_f32_array(w_in_values)
    ));

    
    code.push_str(&format!(
        "pub const W_RES_ROW_PTR: [u16; {}] = {};\n",
        w_res_row_ptr.len(), fmt_u16_array(&w_res_row_ptr)
    ));
    code.push_str(&format!(
        "pub const W_RES_COL_IDX: [u16; {}] = {};\n",
        w_res_col_idx.len(), fmt_u16_array(&w_res_col_idx)
    ));
    code.push_str(&format!(
        "pub const W_RES_VALUES: [f32; {}] = {};\n\n",
        w_res_values.len(), fmt_f32_array(w_res_values)
    ));

    
    code.push_str(&format!(
        "pub const W_OUT_DATA: [f32; {}] = {};\n",
        w_out.len(), fmt_matrix_f32_data(w_out)
    ));

    
    code.push_str(&format!(
        "pub const INITIAL_STATE_DATA: [f32; {}] = {};\n\n",
        final_state.len(), fmt_vector_f32_data(final_state)
    ));

    
    let test_inputs_flat: Vec<String> = inputs_test.iter().map(|v| format!("{:.8}", v[0])).collect();
    code.push_str(&format!(
        "pub const TEST_INPUTS: [f32; {}] = [{}];\n",
        TEST_LEN, test_inputs_flat.join(", ")
    ));

    let test_targets_flat: Vec<String> = targets_test.iter().map(|v| format!("{:.8}", v[0])).collect();
    code.push_str(&format!(
        "pub const TEST_TARGETS: [f32; {}] = [{}];\n",
        TEST_LEN, test_targets_flat.join(", ")
    ));

    let dest_path = out_dir.join("weights.rs");
    fs::write(&dest_path, code).expect("Failed to write weights.rs");
}