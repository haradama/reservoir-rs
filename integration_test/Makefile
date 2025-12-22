SHELL := bash
.ONESHELL:
.DEFAULT_GOAL := help

.PHONY: help targets host wasm arm_m3 arm_m4f arm_m0 riscv x86 smoke

help:
	@echo "Targets (integration_test):"
	@echo "  make targets  # rustup target add ..."
	@echo "  make host     # run test_std (std)"
	@echo "  make wasm     # build test_std for wasm32-wasip1 and run by wasmtime"
	@echo "  make arm_m3   # thumbv7m-none-eabi"
	@echo "  make arm_m4f  # thumbv7em-none-eabihf"
	@echo "  make arm_m0   # thumbv6m-none-eabi"
	@echo "  make riscv    # riscv32imac-unknown-none-elf"
	@echo "  make x86      # i686-unknown-none.json"
	@echo "  make smoke    # all of the above"

targets:
	rustup target add \
	  thumbv7m-none-eabi \
	  thumbv7em-none-eabihf \
	  thumbv6m-none-eabi \
	  wasm32-wasip1 \
	  riscv32imac-unknown-none-elf \
	  x86_64-unknown-none

host:
	@echo "Running Host Smoke Test..."
	cargo run --bin test_std --features std

wasm:
	@echo "Running WASM Smoke Test..."
	cargo build --bin test_std --target wasm32-wasip1 --features std
	wasmtime target/wasm32-wasip1/debug/test_std.wasm

arm_m3:
	@echo "Running ARM Cortex-M3 Smoke Test..."
	cargo run --bin test_arm --target thumbv7m-none-eabi --features qemu -Z build-std=core

arm_m4f:
	@echo "Running ARM Cortex-M4F Smoke Test..."
	cargo run --bin test_arm --target thumbv7em-none-eabihf --features qemu -Z build-std=core

arm_m0:
	@echo "Running ARM Cortex-M0 Smoke Test..."
	cargo run --bin test_arm --target thumbv6m-none-eabi --features qemu -Z build-std=core

riscv:
	@echo "Running RISC-V 32-bit Smoke Test..."
	cargo run --bin test_riscv --target riscv32imac-unknown-none-elf --features qemu -Z build-std=core

x86:
	@echo "Running x86 32-bit Smoke Test..."
	cargo run --bin test_x86 --target i686-unknown-none.json --features qemu -Z build-std=core

smoke: targets host wasm arm_m3 arm_m4f arm_m0 riscv x86
