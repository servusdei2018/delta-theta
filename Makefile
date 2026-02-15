.PHONY: build dev train test clean

## Build the Rust extension in release mode
build:
	maturin develop --release

## Build the Rust extension in debug mode (faster compile)
dev:
	maturin develop

## Train the RL agent with default hyperparameters
train:
	python -m delta_theta.train

## Run Rust unit tests
test:
	cargo test

## Remove build artifacts
clean:
	cargo clean
	rm -rf dist/ build/ *.egg-info __pycache__
