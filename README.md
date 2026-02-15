# Delta-Theta Matrix Integration

**Q-Learned Delta-Theta Matrix Integration** for algorithmic options trading.

A mixed **Rust + Python** project that combines a high-performance Rust core
(theta/delta surface math) with a Python reinforcement-learning environment
powered by Gymnasium and Stable-Baselines3.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Python Layer                    │
│  ┌────────────────┐  ┌───────────────────┐  │
│  │ DeltaThetaEnv  │  │   train.py (PPO)  │  │
│  │ (Gymnasium)    │  │ (Stable-Baselines)│  │
│  └───────┬────────┘  └────────┬──────────┘  │
│          │    PyO3 FFI Bridge │              │
├──────────┼────────────────────┼─────────────┤
│          ▼                    ▼              │
│  ┌──────────────────────────────────────┐   │
│  │         Rust Core (cdylib)           │   │
│  │  DeltaThetaMatrix · BacktestBridge   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

- **Rust core** (`src/lib.rs`): `DeltaThetaMatrix` for surface operations,
  `BacktestBridge` stub for NautilusTrader integration.
- **PyO3 / Maturin**: Compiles Rust into a native Python extension module.
- **Python RL** (`python/delta_theta/`): Gymnasium env + SB3 PPO training loop.

---

## Prerequisites

- **Rust** ≥ 1.75 (install via [rustup](https://rustup.rs))
- **Python** ≥ 3.10 (3.12 recommended)
- **uv** — fast Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **maturin** — Rust/Python build tool (installed automatically by uv)

---

## Setup

```bash
# 1. Create a virtual environment and install Python deps
uv venv
source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows

# 2. Install the project (builds Rust extension + installs Python deps)
uv pip install -e ".[dev]"

# 3. Or use maturin directly for development builds
maturin develop --release
```

---

## Usage

### Build the Rust extension

```bash
make build        # or: maturin develop --release
```

### Train the RL agent

```bash
make train        # or: python -m delta_theta.train --timesteps 100000
```

### Run Rust tests

```bash
make test         # or: cargo test
```

---

## Project Structure

```
.
├── Cargo.toml                 # Rust dependencies & lib config
├── pyproject.toml             # Python deps & Maturin build config
├── Makefile                   # Common dev commands
├── src/
│   └── lib.rs                 # Rust core: DeltaThetaMatrix, BacktestBridge
├── python/
│   └── delta_theta/
│       ├── __init__.py        # Package re-exports
│       ├── env.py             # Gymnasium RL environment
│       └── train.py           # SB3 PPO training script
└── README.md
```

---

## License

See [LICENSE](LICENSE).
