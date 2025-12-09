#![no_std]
#![no_main]

#[cfg(not(target_arch = "avr"))]
compile_error!("This binary must be compiled for AVR architecture");

const EMU_EXIT: &str = "EMULATOR_EXIT";

#[cfg(target_arch = "avr")]
mod avr_impl {
    use panic_halt as _;

    use crate::EMU_EXIT;

    #[arduino_hal::entry]
    fn main() -> ! {
        let dp = arduino_hal::Peripherals::take().unwrap();
        let pins = arduino_hal::pins!(dp);

        let mut serial = arduino_hal::default_serial!(dp, pins, 57600);
        ufmt::uwriteln!(&mut serial, "Hello from AVR!").unwrap();
        ufmt::uwriteln!(&mut serial, "{}", EMU_EXIT).unwrap();
        loop {}
    }
}
