#![no_std]
#![no_main]

#[cfg(not(target_arch = "riscv32"))]
compile_error!("This binary must be compiled for RISC-V architecture");

use panic_halt as _;
use riscv_rt::entry;

const UART0: *mut u8 = 0x1000_0000 as *mut u8;
const EMU_EXIT: &str = "EMULATOR_EXIT";

struct QemuSerial;

impl ufmt::uWrite for QemuSerial {
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

#[entry]
fn main() -> ! {
    let mut serial = QemuSerial;

    ufmt::uwriteln!(&mut serial, "Hello from RISC-V!").unwrap();
    ufmt::uwriteln!(&mut serial, "{}", EMU_EXIT).unwrap();

    loop {}
}
