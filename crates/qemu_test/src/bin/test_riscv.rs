#![no_std]
#![no_main]

#[cfg(not(target_arch = "riscv32"))]
compile_error!("This binary must be compiled for RISC-V architecture");

use panic_halt as _;
use qemu_test::common::{run_inference_test, TestLogger};
use riscv_rt::entry;

const UART0: *mut u8 = 0x1000_0000 as *mut u8;
const EMU_EXIT: &str = "EMULATOR_EXIT";

struct RiscvLogger;

impl RiscvLogger {
    fn write_str(&self, s: &str) {
        for c in s.bytes() {
            unsafe {
                UART0.write_volatile(c);
            }
        }
    }

    fn write_float(&self, val: f32) {
        let int_part = val as i32;
        let frac_part = ((val.abs() - int_part.abs() as f32) * 10000.0) as i32;

        use ufmt::uWrite;
        struct UartWriter;
        impl uWrite for UartWriter {
            type Error = ();
            fn write_str(&mut self, s: &str) -> Result<(), Self::Error> {
                for c in s.bytes() {
                    unsafe {
                        UART0.write_volatile(c);
                    }
                }
                Ok(())
            }
        }
        let mut w = UartWriter;

        if val < 0.0 && int_part == 0 {
            let _ = ufmt::uwrite!(&mut w, "-");
        }
        let _ = ufmt::uwrite!(&mut w, "{}.{}", int_part, frac_part);
    }
}

impl TestLogger for RiscvLogger {
    fn log_info(&mut self, msg: &str) {
        use ufmt::uWrite;
        struct UartWriter;
        impl uWrite for UartWriter {
            type Error = ();
            fn write_str(&mut self, s: &str) -> Result<(), Self::Error> {
                for c in s.bytes() {
                    unsafe {
                        UART0.write_volatile(c);
                    }
                }
                Ok(())
            }
        }
        let mut w = UartWriter;
        let _ = ufmt::uwriteln!(&mut w, "{}", msg);
    }

    fn log_step(&mut self, step: usize, input: f32, target: f32, pred: f32) {
        use ufmt::uWrite;
        struct UartWriter;
        impl uWrite for UartWriter {
            type Error = ();
            fn write_str(&mut self, s: &str) -> Result<(), Self::Error> {
                for c in s.bytes() {
                    unsafe {
                        UART0.write_volatile(c);
                    }
                }
                Ok(())
            }
        }
        let mut w = UartWriter;

        let _ = ufmt::uwrite!(&mut w, "Step {}: Input=", step);
        self.write_float(input);
        let _ = ufmt::uwrite!(&mut w, ", Target=");
        self.write_float(target);
        let _ = ufmt::uwrite!(&mut w, ", Pred=");
        self.write_float(pred);
        let _ = ufmt::uwriteln!(&mut w, "");
    }

    fn log_metrics(&mut self, mse: f32, rmse: f32, r2: f32) {
        use ufmt::uWrite;
        struct UartWriter;
        impl uWrite for UartWriter {
            type Error = ();
            fn write_str(&mut self, s: &str) -> Result<(), Self::Error> {
                for c in s.bytes() {
                    unsafe {
                        UART0.write_volatile(c);
                    }
                }
                Ok(())
            }
        }
        let mut w = UartWriter;

        let _ = ufmt::uwriteln!(&mut w, "--------------------------------------------------");

        let _ = ufmt::uwrite!(&mut w, "MSE  : ");
        self.write_float(mse);
        let _ = ufmt::uwriteln!(&mut w, "");

        let _ = ufmt::uwrite!(&mut w, "RMSE : ");
        self.write_float(rmse);
        let _ = ufmt::uwriteln!(&mut w, "");

        let _ = ufmt::uwrite!(&mut w, "R2   : ");
        self.write_float(r2);
        let _ = ufmt::uwriteln!(&mut w, "");
        let _ = ufmt::uwriteln!(&mut w, "--------------------------------------------------");
    }
}

#[entry]
fn main() -> ! {
    let mut logger = RiscvLogger;

    run_inference_test(&mut logger);

    let mut logger = RiscvLogger;
    logger.log_info(EMU_EXIT);

    loop {}
}
