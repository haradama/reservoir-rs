SHELL := bash
.ONESHELL:
.DEFAULT_GOAL := help

.PHONY: help \
        build test fmt clippy licenses features \
        check smoke ci

help:
	@echo "Targets:"
	@echo "  make fmt        # cargo fmt --check"
	@echo "  make clippy     # cargo clippy (default + all features, deny warnings)"
	@echo "  make features   # verify supported no_std feature combinations build"
	@echo "  make licenses   # cargo deny check licenses"
	@echo "  make build      # cargo build"
	@echo "  make test       # cargo test"
	@echo "  make check      # fmt + clippy + features + licenses + test"
	@echo "  make smoke      # integration_test smoke (all)"
	@echo "  make ci         # check + build + smoke"

fmt:
	cargo fmt --all -- --check

clippy:
	# Default features: catches warnings that only surface under real feature
	# unification (e.g. a no_std dependency compiled next to std crates).
	cargo clippy --workspace --all-targets -- -D warnings
	# All features: broadest lint coverage.
	cargo clippy --workspace --all-targets --all-features -- -D warnings

# Build the library crates under the supported no_std / feature combinations.
#
# The workspace always needs a floating-point backend: `std` (default) or `libm`.
# `--all-features` clippy alone cannot catch breakage here because it turns every
# feature on at once and hides the no_std-only paths. This target exercises the
# individual combinations that embedded users actually build with, and asserts
# that a configuration without any float backend fails fast with a clear error.
features:
	set -e
	export RUSTFLAGS="-D warnings"
	echo ">> supported configurations must build"
	cargo check -p reservoir-core     --no-default-features --features libm
	cargo check -p reservoir-core     --no-default-features --features alloc,libm
	cargo check -p reservoir-infer    --no-default-features --features libm
	cargo check -p reservoir-infer    --no-default-features --features alloc,libm
	cargo check -p reservoir-train    --no-default-features --features libm
	cargo check -p reservoir-datasets --no-default-features
	echo ">> a build without a float backend must fail"
	if cargo check -p reservoir-core --no-default-features                   >/dev/null 2>&1; then \
		echo "ERROR: reservoir-core built without a float backend"; exit 1; fi
	if cargo check -p reservoir-core --no-default-features --features alloc  >/dev/null 2>&1; then \
		echo "ERROR: reservoir-core built with alloc but no float backend"; exit 1; fi
	echo "OK: feature matrix verified"

licenses:
	cargo deny check licenses

build:
	cargo build --verbose

test:
	cargo test --verbose

check: fmt clippy features licenses test

smoke:
	$(MAKE) -C integration_test smoke

ci: check build smoke
