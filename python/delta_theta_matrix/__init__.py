"""
Q-Learned Delta-Theta Matrix Integration.

A high-performance algorithmic trading harness that dynamically resizes multi-leg
options structures to maximize theta decay while penalizing capital inefficiency.

The Rust core provides:
- BacktestEngine: Order book simulation with nanosecond precision
- EngineConfig: Configuration for the simulation engine
- StepResult: Result of a simulation step
- RiskState: Portfolio risk tracking
- Greeks: Option Greeks (delta, gamma, theta, vega)
- PutCreditSpread: Put credit spread structure
- get_state_vector(): Zero-copy NumPy state extraction
- get_state_dim(): State vector dimensionality

Usage:
    from delta_theta_matrix import BacktestEngine, EngineConfig

    config = EngineConfig()
    engine = BacktestEngine(config)
    obs = engine.reset()
    result = engine.step(strike_width=0.0, delta_position=0.0)
"""

__version__ = "0.1.0"

try:
    from delta_theta_matrix._native import (
        BacktestEngine,
        EngineConfig,
        Greeks,
        OptionLeg,
        PutCreditSpread,
        RiskState,
        StepResult,
        get_state_dim,
        get_state_vector,
    )

    __all__ = [
        "BacktestEngine",
        "EngineConfig",
        "StepResult",
        "RiskState",
        "Greeks",
        "PutCreditSpread",
        "OptionLeg",
        "get_state_vector",
        "get_state_dim",
    ]
except ImportError as e:
    import warnings

    warnings.warn(
        f"Native Rust module not found. Build with `maturin develop` first. Error: {e}",
        ImportWarning,
        stacklevel=2,
    )
