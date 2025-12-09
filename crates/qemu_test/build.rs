use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let memory_x_filename = if target.starts_with("thumbv7m-") {
        "memory_lm3s6965.x" // Cortex-M3 (QEMU: lm3s6965evb)
    } else if target.starts_with("thumbv7em-") {
        "memory_mps2.x" // Cortex-M4F (QEMU: mps2-an386)
    } else if target.starts_with("thumbv6m-") {
        "memory_microbit.x" // Cortex-M0 (QEMU: microbit)
    } else {
        return;
    };

    let source_path = PathBuf::from("boards").join(memory_x_filename);
    let dest_path = out_dir.join("memory.x");

    fs::copy(&source_path, &dest_path).expect("failed to copy memory.x");

    println!("cargo:rustc-link-search={}", out_dir.display());

    println!("cargo:rerun-if-changed=boards/{}", memory_x_filename);
    println!("cargo:rerun-if-changed=build.rs");
}
