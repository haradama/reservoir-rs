#![no_std]
#![no_main]

#[cfg(not(target_arch = "arm"))]
compile_error!("This binary must be compiled for ARM architecture");

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use panic_semihosting as _;
use integration_test::common::{run_inference_test, TestLogger};

struct ArmLogger;

impl TestLogger for ArmLogger {
    fn log_info(&mut self, msg: &str) {
        hprintln!("{}", msg);
    }

    fn log_step(&mut self, step: usize, input: f32, target: f32, pred: f32) {
        hprintln!(
            "Step {}: Input={:.4}, Target={:.4}, Pred={:.4}",
            step,
            input,
            target,
            pred
        );
    }

    fn log_metrics(&mut self, mse: f32, rmse: f32, r2: f32) {
        hprintln!("--------------------------------------------------");
        hprintln!("MSE  : {:.6}", mse);
        hprintln!("RMSE : {:.6}", rmse);
        hprintln!("R^2  : {:.6}", r2);
        hprintln!("--------------------------------------------------");
    }
}

#[entry]
fn main() -> ! {
    let mut logger = ArmLogger;

    run_inference_test(&mut logger);

    hprintln!("EMULATOR_EXIT");
    debug::exit(debug::EXIT_SUCCESS);

    loop {}
}
