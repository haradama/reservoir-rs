#![no_std]
#![no_main]

#[cfg(not(target_arch = "arm"))]
compile_error!("This binary must be compiled for ARM architecture");

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use panic_semihosting as _;

#[entry]
fn main() -> ! {
    hprintln!("Hello from ARM!");

    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
