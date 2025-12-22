#![no_std]
#![no_main]

#[cfg(not(target_arch = "x86"))]
compile_error!("This binary must be compiled for x86 architecture");

use core::arch::global_asm;
use core::panic::PanicInfo;
use integration_test::common::{run_inference_test, TestLogger};
use x86::io::outb;

// Multiboot Header Definition
const MULTIBOOT_HEADER_MAGIC: u32 = 0x1BADB002;
const MULTIBOOT_HEADER_FLAGS: u32 = 0;
const MULTIBOOT_HEADER_CHECKSUM: u32 = -(MULTIBOOT_HEADER_MAGIC as i32 + MULTIBOOT_HEADER_FLAGS as i32) as u32;

global_asm!(
    r#"
    .intel_syntax noprefix

    /* --- Multiboot Header --- */
    .section .multiboot_header
    .align 4
    .long {magic}
    .long {flags}
    .long {checksum}

    /* --- Text Section (Entry Point) --- */
    .section .text
    .global _start
    .code32
    _start:
        /* Setup stack pointer using the label defined in .bss below */
        /* 'offset' keyword gets the address of the symbol */
        mov esp, offset stack_top
        
        /* Clear EFLAGS */
        push 0
        popfd

        /* Call Rust kernel main */
        call kmain

        /* Halt loop if kmain returns */
        cli
    1:  hlt
        jmp 1b

    /* --- BSS Section (Stack Allocation) --- */
    .section .bss
    .align 16
    stack_bottom:
        .skip 16384 /* 16KB Stack */
    stack_top:
    "#,
    magic = const MULTIBOOT_HEADER_MAGIC,
    flags = const MULTIBOOT_HEADER_FLAGS,
    checksum = const MULTIBOOT_HEADER_CHECKSUM,
);

const COM1: u16 = 0x3F8;
const EMU_EXIT: &str = "EMULATOR_EXIT";

struct X86Logger;

impl X86Logger {
    unsafe fn init() {
        // Initialize Serial Port (COM1) minimal setup
        outb(COM1 + 1, 0x00); // Disable all interrupts
        outb(COM1 + 3, 0x80); // Enable DLAB (set baud rate divisor)
        outb(COM1 + 0, 0x03); // Set divisor to 3 (lo byte) 38400 baud
        outb(COM1 + 1, 0x00); //                  (hi byte)
        outb(COM1 + 3, 0x03); // 8 bits, no parity, one stop bit
        outb(COM1 + 2, 0xC7); // Enable FIFO, clear them, with 14-byte threshold
        outb(COM1 + 4, 0x0B); // IRQs enabled, RTS/DSR set
    }

    fn write_byte(&self, byte: u8) {
        unsafe {
            // Wait for transmit empty
            while (x86::io::inb(COM1 + 5) & 0x20) == 0 {}
            outb(COM1, byte);
        }
    }

    fn write_str(&self, s: &str) {
        for byte in s.bytes() {
            self.write_byte(byte);
        }
        self.write_byte(b'\n');
    }
}

impl TestLogger for X86Logger {
    fn log_info(&mut self, msg: &str) {
        self.write_str(msg);
    }

    fn log_step(&mut self, step: usize, input: f32, target: f32, pred: f32) {
        use core::fmt::Write;
        struct Writer(X86Logger);
        impl Write for Writer {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                for byte in s.bytes() {
                    self.0.write_byte(byte);
                }
                Ok(())
            }
        }
        let mut w = Writer(X86Logger);
        writeln!(w, "Step {}: Input={:.4}, Target={:.4}, Pred={:.4}", step, input, target, pred).ok();
    }

    fn log_metrics(&mut self, mse: f32, rmse: f32, r2: f32) {
        use core::fmt::Write;
        struct Writer(X86Logger);
        impl Write for Writer {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                for byte in s.bytes() {
                    self.0.write_byte(byte);
                }
                Ok(())
            }
        }
        let mut w = Writer(X86Logger);
        writeln!(w, "--------------------------------------------------").ok();
        writeln!(w, "MSE  : {:.6}", mse).ok();
        writeln!(w, "RMSE : {:.6}", rmse).ok();
        writeln!(w, "R^2  : {:.6}", r2).ok();
        writeln!(w, "--------------------------------------------------").ok();
    }
}

#[no_mangle]
pub extern "C" fn kmain() -> ! {
    unsafe { X86Logger::init() };
    
    let mut logger = X86Logger;
    
    // FPU initialization for float operations
    unsafe {
        // Enable SSE if available
        // CR4.OSFXSR (bit 9) = 1, CR4.UNMASKED_SSE (bit 10) = 1
        let mut cr4 = x86::controlregs::cr4();
        cr4 |= x86::controlregs::Cr4::CR4_ENABLE_SSE | x86::controlregs::Cr4::CR4_UNMASKED_SSE;
        x86::controlregs::cr4_write(cr4);
    }

    run_inference_test(&mut logger);

    logger.log_info(EMU_EXIT);

    loop {
        unsafe { x86::halt() };
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let logger = X86Logger;
    logger.write_str("PANIC OCCURRED");
    if let Some(location) = info.location() {
        use core::fmt::Write;
        struct Writer(X86Logger);
        impl Write for Writer {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                for byte in s.bytes() {
                    self.0.write_byte(byte);
                }
                Ok(())
            }
        }
        writeln!(Writer(logger), "File: {}, Line: {}", location.file(), location.line()).ok();
    }
    
    loop {
        unsafe { x86::halt() };
    }
}