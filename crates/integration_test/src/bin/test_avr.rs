#![no_std]
#![no_main]

#[cfg(not(target_arch = "avr"))]
compile_error!("This binary must be compiled for AVR architecture");

use panic_halt as _;
use integration_test::common::{run_inference_test, TestLogger};

const EMU_EXIT: &str = "EMULATOR_EXIT";

struct AvrLogger<S> {
    serial: S,
}

impl<S> AvrLogger<S>
where
    S: ufmt::uWrite,
{
    fn print_float(&mut self, val: f32) {
        let int_part = val as i32;

        if val < 0.0 && int_part == 0 {
            let _ = ufmt::uwrite!(&mut self.serial, "-");
        }

        let frac_part = ((val.abs() - int_part.abs() as f32) * 1000000.0) as i32;

        let _ = ufmt::uwrite!(&mut self.serial, "{}.{}", int_part, frac_part);
    }
}

impl<S> TestLogger for AvrLogger<S>
where
    S: ufmt::uWrite,
{
    fn log_info(&mut self, msg: &str) {
        ufmt::uwriteln!(&mut self.serial, "{}", msg).ok();
    }

    fn log_step(&mut self, step: usize, input: f32, target: f32, pred: f32) {
        let _ = ufmt::uwrite!(&mut self.serial, "Step {}: In=", step);
        self.print_float(input);

        let _ = ufmt::uwrite!(&mut self.serial, ", Tgt=");
        self.print_float(target);

        let _ = ufmt::uwrite!(&mut self.serial, ", Pred=");
        self.print_float(pred);

        let _ = ufmt::uwriteln!(&mut self.serial, "");
    }

    fn log_metrics(&mut self, mse: f32, rmse: f32, r2: f32) {
        let _ = ufmt::uwriteln!(&mut self.serial, "--- Metrics ---");

        let _ = ufmt::uwrite!(&mut self.serial, "MSE : ");
        self.print_float(mse);
        let _ = ufmt::uwriteln!(&mut self.serial, "");

        let _ = ufmt::uwrite!(&mut self.serial, "RMSE: ");
        self.print_float(rmse);
        let _ = ufmt::uwriteln!(&mut self.serial, "");

        let _ = ufmt::uwrite!(&mut self.serial, "R2  : ");
        self.print_float(r2);
        let _ = ufmt::uwriteln!(&mut self.serial, "");
    }
}

#[arduino_hal::entry]
fn main() -> ! {
    let dp = arduino_hal::Peripherals::take().unwrap();
    let pins = arduino_hal::pins!(dp);
    let serial = arduino_hal::default_serial!(dp, pins, 57600);

    let mut logger = AvrLogger { serial };

    run_inference_test(&mut logger);

    logger.log_info(EMU_EXIT);
    loop {}
}
