RUSTUP ?= rustup

.PHONY: clean build develop setup-rust rust-test test

clean:
	cargo clean
	rm -rf *~ dist *.egg-info build target

setup-rust:
	$(RUSTUP) toolchain install --profile minimal

build:
	maturin build -F python  -i python3 --release

develop:
	maturin develop -F python --release

rust-test:
	cargo test

test:
	cargo test --features python
	python3 tests/test_decode.py
