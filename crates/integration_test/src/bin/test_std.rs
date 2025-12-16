#[cfg(not(feature = "std"))]
compile_error!("This binary requires the 'std' feature");

use integration_test::common::{run_inference_test, TestLogger};

struct StdLogger;

impl TestLogger for StdLogger {
    fn log_info(&mut self, msg: &str) {
        println!("{}", msg);
    }

    fn log_step(&mut self, step: usize, input: f32, target: f32, pred: f32) {
        println!(
            "Step {}: Input={:.4}, Target={:.4}, Pred={:.4}",
            step, input, target, pred
        );
    }

    fn log_metrics(&mut self, mse: f32, rmse: f32, r2: f32) {
        println!("--------------------------------------------------");
        println!("MSE  : {:.6}", mse);
        println!("RMSE : {:.6}", rmse);
        println!("R^2  : {:.6}", r2);
        println!("--------------------------------------------------");
    }
}

fn main() {
    let mut logger = StdLogger;
    run_inference_test(&mut logger);
}