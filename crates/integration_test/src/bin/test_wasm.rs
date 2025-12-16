#![no_main]

#[cfg(not(target_arch = "wasm32"))]
compile_error!("This binary must be compiled for wasm32 architecture");

use integration_test::common::{run_inference_test, TestLogger};
use wasm_bindgen::prelude::*;

struct WasmLogger;

impl TestLogger for WasmLogger {
    fn log_info(&mut self, msg: &str) {
        web_sys::console::log_1(&JsValue::from_str(msg));
    }

    fn log_step(&mut self, step: usize, input: f32, target: f32, pred: f32) {
        let msg = format!(
            "Step {}: Input={:.4}, Target={:.4}, Pred={:.4}",
            step, input, target, pred
        );
        web_sys::console::log_1(&JsValue::from_str(&msg));
    }

    fn log_metrics(&mut self, mse: f32, rmse: f32, r2: f32) {
        let sep = "--------------------------------------------------";
        web_sys::console::log_1(&JsValue::from_str(sep));
        
        let msg_mse = format!("MSE  : {:.6}", mse);
        web_sys::console::log_1(&JsValue::from_str(&msg_mse));
        
        let msg_rmse = format!("RMSE : {:.6}", rmse);
        web_sys::console::log_1(&JsValue::from_str(&msg_rmse));
        
        let msg_r2 = format!("R^2  : {:.6}", r2);
        web_sys::console::log_1(&JsValue::from_str(&msg_r2));
        
        web_sys::console::log_1(&JsValue::from_str(sep));
    }
}

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();

    let mut logger = WasmLogger;
    
    web_sys::console::log_1(&JsValue::from_str("--- WASM Inference Test Start ---"));
    run_inference_test(&mut logger);
    web_sys::console::log_1(&JsValue::from_str("--- WASM Inference Test Finished ---"));
}