#![no_std]
#![no_main]

#[cfg(not(target_arch = "arm"))]
compile_error!("This binary must be compiled for ARM architecture");

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use panic_semihosting as _;

use heapless::Vec as HVec;
use nalgebra::{SMatrix, SVector};
use reservoir_infer::readout::static_readout::StaticReadout;
use reservoir_infer::reservoir::static_reservoir::StaticReservoir;

#[entry]
fn main() -> ! {
    hprintln!("Starting Static Reservoir (No Alloc)...");

    const INPUT_DIM: usize = 1;
    const RESERVOIR_SIZE: usize = 5;
    const EXTENDED_SIZE: usize = 1 + INPUT_DIM + RESERVOIR_SIZE;
    const OUTPUT_DIM: usize = 1;

    let w_in = SMatrix::<f32, RESERVOIR_SIZE, INPUT_DIM>::repeat(0.5);
    let w_res = SMatrix::<f32, RESERVOIR_SIZE, RESERVOIR_SIZE>::repeat(0.1);
    let w_out = SMatrix::<f32, OUTPUT_DIM, EXTENDED_SIZE>::repeat(0.2);

    let leaking_rate = 0.9f32;

    let mut reservoir = StaticReservoir::<f32, INPUT_DIM, RESERVOIR_SIZE, EXTENDED_SIZE>::create(
        w_in,
        w_res,
        leaking_rate,
    );

    let readout = StaticReadout::<f32, EXTENDED_SIZE, OUTPUT_DIM>::create(w_out);

    let mut input_buffer: HVec<f32, 10> = HVec::new();
    input_buffer.push(1.0).ok();

    let input = SVector::<f32, INPUT_DIM>::new(input_buffer[0]);

    let state = reservoir.step(&input);
    let output = readout.predict(state);

    hprintln!("Input: {:.4}", input[0]);
    hprintln!("Output: {:.4}", output[0]);

    hprintln!("EMULATOR_EXIT");
    debug::exit(debug::EXIT_SUCCESS);

    loop {}
}
