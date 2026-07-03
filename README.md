# reservoir-rs

| Crate                | crates.io                                                                                                           | docs.rs                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `reservoir-core`     | [![crates.io](https://img.shields.io/crates/v/reservoir-core.svg)](https://crates.io/crates/reservoir-core)         | [![docs.rs](https://docs.rs/reservoir-core/badge.svg)](https://docs.rs/reservoir-core)         |
| `reservoir-infer`    | [![crates.io](https://img.shields.io/crates/v/reservoir-infer.svg)](https://crates.io/crates/reservoir-infer)       | [![docs.rs](https://docs.rs/reservoir-infer/badge.svg)](https://docs.rs/reservoir-infer)       |
| `reservoir-train`    | [![crates.io](https://img.shields.io/crates/v/reservoir-train.svg)](https://crates.io/crates/reservoir-train)       | [![docs.rs](https://docs.rs/reservoir-train/badge.svg)](https://docs.rs/reservoir-train)       |
| `reservoir-datasets` | [![crates.io](https://img.shields.io/crates/v/reservoir-datasets.svg)](https://crates.io/crates/reservoir-datasets) | [![docs.rs](https://docs.rs/reservoir-datasets/badge.svg)](https://docs.rs/reservoir-datasets) |

A minimal, experimental Rust workspace for reservoir computing / Echo State Networks (ESNs).

This project focuses on:

- reusable **core traits** (reservoir / readout / trainer)
- optional **no_std** support via feature flags
- simple **datasets** for time-series experiments
- lightweight **training + inference** components

> Note: This is early-stage software. APIs, module layout, and crate names may change.

## What’s inside (high level)

- **Core**: traits, scalar/type aliases, basic metrics (MSE/RMSE/NRMSE/R²)
- **Inference**: ESN-style reservoir + readout building blocks (dynamic and/or const-generic variants depending on features)
- **Training**: simple readout training (e.g., ridge / lasso-style solvers)
- **Datasets**: synthetic time-series generators (for quick experiments)
- **Embedded/QEMU tests**: optional no_std inference checks on emulated targets

## Feature flags & `no_std`

Every crate defaults to `std`. For `no_std` / embedded targets you must select a
floating-point backend explicitly, because `Scalar::activation`, the `metrics`
module, and the reservoir updates all rely on `tanh` / `powi` / `sqrt`:

| Feature | Effect |
| ------- | ------ |
| `std` (default) | Standard library + heap types (implies `alloc`). |
| `libm` | Provides the float math used on `no_std` targets. **Required** for any `no_std` build. |
| `alloc` | Adds heap-backed `nalgebra` types. Does **not** provide math on its own — pair it with `libm`. |

Typical `no_std` configurations:

```bash
# Static-only inference (no heap):
cargo build -p reservoir-infer --no-default-features --features libm
# Dynamic inference / training on no_std (heap + math):
cargo build -p reservoir-train --no-default-features --features libm
```

Selecting neither `std` nor `libm` fails fast with a clear compile error rather
than an obscure "unresolved import `num_traits::Float`". Run `make features` to
check every supported combination locally.

## Examples

This workspace includes two small README-oriented examples that demonstrate the same ESN idea in two styles:

- **Dynamic (DMatrix/DVector) – training + inference**

Trains a Ridge readout on a toy 1-step-ahead sine-wave prediction task using heap-backed `nalgebra::DMatrix/DVector`, then evaluates MSE/RMSE/R².

```bash
cargo run -p reservoir-train --example readme_dmatrix_dvector
````

- **Static (SMatrix/SVector) – inference only (weights pre-baked)**

Runs inference only using const-generic `nalgebra::SMatrix/SVector` and fixed (pretrained) weights embedded in the example. This is intended as a “static model” style that maps well to `no_std` / embedded scenarios.

```bash
cargo run -p reservoir-infer --example readme_smatrix_svector
```

## Verified Environments

This repository includes an `integration_test` crate to validate **static inference** (pretrained weights + no_std-friendly execution) across multiple targets.

### Targets covered

- **Host (std)**: native execution with `std` enabled  
- **WASM (wasm32-wasip1)**: runs with Wasmtime  
- **Embedded / bare-metal (no_std)** via QEMU:
  - **ARM Cortex-M3** (`thumbv7m-none-eabi`)
  - **ARM Cortex-M4F** (`thumbv7em-none-eabihf`)
  - **ARM Cortex-M0** (`thumbv6m-none-eabi`)
  - **RISC-V 32-bit** (`riscv32imac-unknown-none-elf`)
  - **x86 32-bit** (`i686-unknown-none` via a custom JSON target)

### How to run

From the repository root:

```bash
cd integration_test
make smoke
````

This will:

1. Install required Rust targets (`make targets`)
2. Run the host smoke test (`make host`)
3. Build & run the WASM smoke test via Wasmtime (`make wasm`)
4. Run QEMU-based no_std tests for ARM / RISC-V / x86 (`make arm_m3`, `make arm_m4f`, `make arm_m0`, `make riscv`, `make x86`)

> Note: `integration_test` uses a pinned nightly toolchain (see `integration_test/rust-toolchain.toml`) and `-Z build-std=core` for bare-metal targets.

## License

Apache License 2.0
