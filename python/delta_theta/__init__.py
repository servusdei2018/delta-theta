# Re-export Rust extension classes so they are available at package level.
# After `maturin develop`, the compiled .so is placed here by maturin.
from delta_theta.delta_theta import DeltaThetaMatrix, BacktestBridge  # noqa: F401

__all__ = ["DeltaThetaMatrix", "BacktestBridge"]
