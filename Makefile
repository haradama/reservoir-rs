SHELL := bash
.ONESHELL:
.DEFAULT_GOAL := help

.PHONY: help \
        build test fmt clippy licenses \
        check smoke ci

help:
	@echo "Targets:"
	@echo "  make fmt        # cargo fmt --check"
	@echo "  make clippy     # cargo clippy (deny warnings)"
	@echo "  make licenses   # cargo deny check licenses"
	@echo "  make build      # cargo build"
	@echo "  make test       # cargo test"
	@echo "  make check      # fmt + clippy + licenses + test"
	@echo "  make smoke      # integration_test smoke (all)"
	@echo "  make ci         # check + build + smoke"

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

licenses:
	cargo deny check licenses

build:
	cargo build --verbose

test:
	cargo test --verbose

check: fmt clippy licenses test

smoke:
	$$(MAKE) -C integration_test smoke

ci: check build smoke
