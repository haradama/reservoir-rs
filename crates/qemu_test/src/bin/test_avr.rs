#![no_std]
#![no_main]

#[cfg(not(target_arch = "avr"))]
compile_error!("This binary must be compiled for AVR architecture");

use panic_halt as _;
use qemu_test::common::{run_inference_test, TestLogger};

const EMU_EXIT: &str = "EMULATOR_EXIT";

struct AvrLogger<S> {
    serial: S,
}

impl<S> TestLogger for AvrLogger<S>
where
    S: ufmt::uWrite,
{
    fn log_info(&mut self, msg: &str) {
        ufmt::uwriteln!(&mut self.serial, "{}", msg).ok();
    }

    fn log_step(&mut self, step: usize, input: f32, target: f32, pred: f32) {
        let i_int = input as i32;
        let i_frac = ((input.abs() - i_int.abs() as f32) * 1000.0) as i32;
        let p_int = pred as i32;
        let p_frac = ((pred.abs() - p_int.abs() as f32) * 1000.0) as i32;

        ufmt::uwriteln!(
            &mut self.serial,
            "Step {}: In={}.{}, Tgt=..., Pred={}.{}",
            step,
            i_int,
            i_frac,
            p_int,
            p_frac
        )
        .ok();
    }

    fn log_mse(&mut self, mse: f32) {
        let m_int = mse as i32;
        let m_frac = ((mse - m_int as f32) * 1000000.0) as i32;
        ufmt::uwriteln!(&mut self.serial, "MSE: {}.{}", m_int, m_frac).ok();
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
