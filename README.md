# Q-Learned Delta-Theta Matrix Integration

A high-performance algorithmic trading harness that dynamically resizes multi-leg options structures to maximize theta decay while penalizing capital inefficiency. Built as a mixed Rust/Python system using PyO3 for zero-copy interop.

## Architecture

```mermaid
flowchart TB
    subgraph python ["Python Layer"]
        env["env.py<br/>Gymnasium Env"]
        train["train.py<br/>SB3 SAC/PPO"]
        visualize["visualize.py<br/>3D Plotly Surfaces"]
        live["live.py<br/>Pub/Sub Signals"]
    end

    subgraph interop ["Zero-Copy NumPy Arrays · PyO3"]
        boundary[" "]
    end

    subgraph rust ["Rust Core"]
        engine["engine.rs<br/>Backtest Engine"]
        orderbook["orderbook.rs<br/>L2 Depth"]
        options["options.rs<br/>Black-Scholes"]
        risk["risk.rs<br/>Margin Tracking"]
        state["state.rs<br/>NumPy Out"]
        lib["lib.rs<br/>PyO3 Entry"]
    end

    env --> boundary
    train --> boundary
    visualize --> boundary
    live --> boundary
    boundary --> engine
    boundary --> orderbook
    boundary --> options
    boundary --> risk
    boundary --> state
    boundary --> lib
```

## Components

### Rust Core (`src/`)

| Module | Description |
|--------|-------------|
| `lib.rs` | PyO3 module entry point — exposes all types and functions to Python |
| `engine.rs` | `BacktestEngine` with nanosecond-precision order book simulation, episodic resets, and configurable tick parameters |
| `orderbook.rs` | L2 order book with 10 levels of depth, mean-reverting random walks, and multi-strike option chains |
| `options.rs` | Black-Scholes put pricing, Greeks (δ, γ, θ, ν), Newton-Raphson IV solver, put credit spread structures |
| `risk.rs` | `RiskState` tracking buying power, margin requirements, early assignment detection, and catastrophic margin calls |
| `state.rs` | Zero-copy NumPy array emission via `get_state_vector()` — 20-dimensional observation space |

### Python Layer (`python/delta_theta_matrix/`)

| Module | Description |
|--------|-------------|
| `env.py` | `DeltaThetaEnv(gymnasium.Env)` — standard Gym interface with 2D continuous action space |
| `train.py` | CLI training script — SAC/PPO via Stable Baselines3 with TensorBoard logging |
| `visualize.py` | 3D Plotly surfaces (TTE × IV × Strike Width) and reward curves |
| `live.py` | Market data feed abstraction, `SignalBus` pub/sub, real-time inference loop |
| `feeds/` | Concrete feed implementations: `TradierFeed`, `AlpacaFeed` |

## Quick Start

### Prerequisites

- Rust toolchain (1.70+)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip (fallback)

### Build

```bash
# Create a virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"

# Build and install the Rust extension
uv run maturin develop --release
```

> **Note:** If you don't have `uv`, you can fall back to pip:
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install -e ".[dev]"
> maturin develop --release
> ```

### Train

```bash
# Train with SAC (default) for 100k timesteps
python -m delta_theta_matrix.train train --algo sac --timesteps 100000

# Train with PPO
python -m delta_theta_matrix.train train --algo ppo --timesteps 500000 --learning-rate 3e-4

# Full options
python -m delta_theta_matrix.train train \
    --algo sac \
    --timesteps 200000 \
    --learning-rate 3e-4 \
    --batch-size 256 \
    --gamma 0.99 \
    --penalty-scale 10.0 \
    --output-dir output \
    --seed 42
```

### Evaluate

```bash
# Run evaluation episodes
python -m delta_theta_matrix.train evaluate output/sac_*/final_model \
    --algo sac \
    --n-episodes 50 \
    --output output/eval_data.json
```

### Visualize

```bash
# Generate 3D surface and reward plots
python -m delta_theta_matrix.visualize --data output/eval_data.json --output output/plots/
```

### Live Inference

```python
from delta_theta_matrix.live import run_live, WebSocketFeed, SignalBus

feed = WebSocketFeed(url="wss://your-data-provider.com/market")
bus = SignalBus()
bus.register_callback(lambda signal: print(signal.to_json()))

run_live(
    model_path="output/sac_model/final_model",
    feed=feed,
    symbols=["MU", "AMD"],
    signal_bus=bus,
)
```

## Live Data Feeds

The project ships with two production-ready market data feed integrations that
implement the [`MarketDataFeed`](python/delta_theta_matrix/live.py:133) ABC and can be
plugged directly into [`run_live()`](python/delta_theta_matrix/live.py:250).

### TradierFeed

```python
from delta_theta_matrix.feeds import TradierFeed

# Reads TRADIER_API_KEY and TRADIER_SANDBOX env vars by default
feed = TradierFeed()
feed.connect()
feed.subscribe(["MU", "AMD"])

# Options chain with Greeks & IV
chain = feed.get_options_chain("MU", "2026-03-20")

# Real-time / delayed quotes
quotes = feed.get_quotes(["MU", "AMD"])

# Historical daily bars
history = feed.get_history("MU", interval="daily", start="2026-01-01", end="2026-02-01")

feed.disconnect()
```

| Env Variable | Description | Default |
|---|---|---|
| `TRADIER_API_KEY` | Tradier API bearer token | *(required)* |
| `TRADIER_SANDBOX` | `"true"` for sandbox, `"false"` for production | `"true"` |

### AlpacaFeed

```python
from delta_theta_matrix.feeds import AlpacaFeed

# Reads ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER env vars
feed = AlpacaFeed()
feed.connect()
feed.subscribe(["MU", "AMD"])

# Options chain snapshots
chain = feed.get_options_chain("MU", expiration_gte="2026-03-01")

# Latest stock quotes
quotes = feed.get_quotes(["MU", "AMD"])

# Historical bars
bars = feed.get_history("MU", timeframe="1Day", start="2026-01-01", end="2026-02-01")

feed.disconnect()
```

| Env Variable | Description | Default |
|---|---|---|
| `ALPACA_API_KEY` | Alpaca API key ID | *(required)* |
| `ALPACA_SECRET_KEY` | Alpaca secret key | *(required)* |
| `ALPACA_PAPER` | `"true"` for paper trading, `"false"` for live | `"true"` |

### Using feeds with `run_live()`

Both feeds are drop-in replacements for the default `WebSocketFeed`:

```python
from delta_theta_matrix.live import run_live, SignalBus
from delta_theta_matrix.feeds import TradierFeed

feed = TradierFeed()
bus = SignalBus()
bus.register_callback(lambda signal: print(signal.to_json()))

run_live(
    model_path="output/sac_model/final_model",
    feed=feed,
    symbols=["MU", "AMD"],
    signal_bus=bus,
)
```

## Observation Space (20-dimensional)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Underlying price | Normalized by /100 |
| 1 | ATM bid-ask spread | Spread at the money |
| 2–6 | OTM put deltas | Delta at 5 strike levels |
| 7–11 | IV surface | Implied vol at 5 strikes |
| 12 | Time-to-expiry | In years |
| 13 | Current IV | Overall implied volatility |
| 14 | Margin utilization | Ratio of margin used to buying power |
| 15 | Theta exposure | Net theta across positions |
| 16 | Buying power | Normalized to initial |
| 17 | Episode progress | Step / max_steps |
| 18 | Position count | Number of open spreads |
| 19 | Episode P&L | Normalized by /1000 |

## Action Space

| Dimension | Range | Maps To |
|-----------|-------|---------|
| strike_width | [-1, 1] | $2.50 – $15.00 spread width |
| delta_position | [-1, 1] | -0.50 – -0.05 short put delta |

## Reward Function

```
reward = θ_captured − (margin_utilization³) × penalty_scale + P&L × 0.01
```

- **Theta captured**: Absolute net theta exposure across positions
- **Margin penalty**: Cubic penalty on margin utilization discourages over-leveraging
- **P&L component**: Small weight on realized P&L for directional signal
- **Margin call**: −100 terminal penalty
- **Expiration**: +10% of episode P&L bonus

## Configuration

`EngineConfig` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tickers` | `["MU", "AMD"]` | Underlying symbols |
| `initial_prices` | `[85.0, 120.0]` | Starting prices |
| `initial_buying_power` | `100,000` | Starting capital |
| `risk_free_rate` | `0.05` | Annual risk-free rate |
| `initial_tte` | `0.0833` | ~30 days to expiry |
| `max_episode_steps` | `1000` | Steps before truncation |
| `base_iv` | `0.30` | Base implied volatility |
| `vol_spike_prob` | `0.02` | Probability of IV spike per tick |
| `vol_spike_magnitude` | `1.5` | IV spike multiplier |

## Development

```bash
# Run Rust tests
cargo test

# Build in debug mode
uv run maturin develop

# Build optimized release
uv run maturin develop --release

# Monitor training with TensorBoard
uv run tensorboard --logdir output/*/tensorboard
```

## License

MIT
