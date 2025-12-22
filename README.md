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

## License

Apache License 2.0
