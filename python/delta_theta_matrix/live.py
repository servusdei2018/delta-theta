"""
Live data ingestion, signal generation with pub/sub callbacks.

Provides:
- Abstract MarketDataFeed with connect/subscribe/on_tick
- WebSocketFeed stub implementation
- Signal dataclass for trade signals
- SignalBus pub/sub system
- run_live() function for real-time inference

Usage:
    from delta_theta_matrix.live import run_live, WebSocketFeed, SignalBus

    feed = WebSocketFeed(url="wss://example.com/market")
    bus = SignalBus()
    bus.register_callback(lambda signal: print(signal))
    run_live("output/sac_model/final_model", feed, ["MU", "AMD"], bus)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptionLegSpec:
    """Specification for a single option leg in a signal."""

    strike: float
    expiry: str  # ISO date string
    option_type: str  # "put" or "call"
    side: str  # "buy" or "sell"
    quantity: int = 1


@dataclass
class Signal:
    """Trade signal emitted by the live inference engine.

    Attributes:
        ticker: Underlying symbol.
        action_type: One of 'buy', 'sell', 'close'.
        legs: List of option legs comprising the trade.
        confidence: Model confidence score [0, 1].
        timestamp: UTC timestamp of signal generation.
        metadata: Additional signal metadata.
    """

    ticker: str
    action_type: str  # "buy", "sell", "close"
    legs: list[OptionLegSpec] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize signal to dictionary."""
        return {
            "ticker": self.ticker,
            "action_type": self.action_type,
            "legs": [
                {
                    "strike": leg.strike,
                    "expiry": leg.expiry,
                    "option_type": leg.option_type,
                    "side": leg.side,
                    "quantity": leg.quantity,
                }
                for leg in self.legs
            ],
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize signal to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class SignalBus:
    """Simple pub/sub system for trade signals.

    Register callbacks that will be invoked whenever a signal is emitted.

    Example:
        bus = SignalBus()
        bus.register_callback(lambda s: print(f"Signal: {s.ticker} {s.action_type}"))
        bus.emit(Signal(ticker="MU", action_type="buy"))
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable[[Signal], None]] = []

    def register_callback(self, callback: Callable[[Signal], None]) -> None:
        """Register a callback to receive signals.

        Args:
            callback: Function that accepts a Signal object.
        """
        self._callbacks.append(callback)

    def emit(self, signal: Signal) -> None:
        """Emit a signal to all registered callbacks.

        Args:
            signal: The trade signal to broadcast.
        """
        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    @property
    def callback_count(self) -> int:
        """Number of registered callbacks."""
        return len(self._callbacks)


class MarketDataFeed(ABC):
    """Abstract base class for market data feeds.

    Implementations should provide real-time tick data from a market data source.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        ...

    @abstractmethod
    def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data for the given symbols.

        Args:
            symbols: List of ticker symbols to subscribe to.
        """
        ...

    @abstractmethod
    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback for incoming tick data.

        Args:
            callback: Function called with tick data dict containing at minimum:
                      {'symbol': str, 'price': float, 'timestamp': str}
        """
        ...


class WebSocketFeed(MarketDataFeed):
    """WebSocket-based market data feed (stub implementation).

    This is a stub that simulates market data for development and testing.
    Replace with actual WebSocket connection for production use.

    Args:
        url: WebSocket URL to connect to.
        api_key: Optional API key for authentication.
    """

    def __init__(self, url: str = "wss://example.com/market", api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        self._connected = False
        self._subscribed_symbols: list[str] = []
        self._tick_callbacks: list[Callable[[dict[str, Any]], None]] = []

    def connect(self) -> None:
        """Establish WebSocket connection (stub)."""
        logger.info(f"Connecting to {self.url}...")
        # In production: establish actual WebSocket connection
        self._connected = True
        logger.info("Connected (stub mode)")

    def disconnect(self) -> None:
        """Disconnect from WebSocket (stub)."""
        self._connected = False
        logger.info("Disconnected")

    def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols (stub).

        Args:
            symbols: List of ticker symbols.
        """
        self._subscribed_symbols = symbols
        logger.info(f"Subscribed to: {symbols}")

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register tick callback.

        Args:
            callback: Function to call on each tick.
        """
        self._tick_callbacks.append(callback)

    def simulate_ticks(self, n_ticks: int = 100, interval: float = 0.1) -> None:
        """Generate synthetic tick data for testing.

        Args:
            n_ticks: Number of ticks to generate.
            interval: Seconds between ticks.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        rng = np.random.default_rng(42)
        prices = {sym: 100.0 for sym in self._subscribed_symbols}

        for i in range(n_ticks):
            for symbol in self._subscribed_symbols:
                # Random walk
                prices[symbol] *= 1.0 + rng.normal(0, 0.001)
                tick = {
                    "symbol": symbol,
                    "price": prices[symbol],
                    "bid": prices[symbol] * 0.999,
                    "ask": prices[symbol] * 1.001,
                    "volume": int(rng.integers(100, 10000)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tick_num": i,
                }
                for cb in self._tick_callbacks:
                    cb(tick)

            time.sleep(interval)


def run_live(
    model_path: str,
    feed: MarketDataFeed,
    symbols: list[str],
    signal_bus: SignalBus,
    algo: str = "sac",
    confidence_threshold: float = 0.5,
) -> None:
    """Run live inference: load model, connect feed, emit signals on each tick.

    Args:
        model_path: Path to the trained SB3 model.
        feed: Market data feed instance.
        symbols: List of symbols to trade.
        signal_bus: Signal bus for emitting trade signals.
        algo: Algorithm used ('sac' or 'ppo').
        confidence_threshold: Minimum confidence to emit a signal.
    """
    from stable_baselines3 import PPO, SAC

    from delta_theta_matrix import BacktestEngine, EngineConfig

    # Load the trained model
    logger.info(f"Loading {algo.upper()} model from {model_path}")
    if algo.lower() == "sac":
        model = SAC.load(model_path)
    elif algo.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Create engine for state computation
    config = EngineConfig()
    engine = BacktestEngine(config)
    engine.reset()

    # Track state
    tick_count = 0

    def on_tick(tick: dict[str, Any]) -> None:
        nonlocal tick_count
        tick_count += 1

        symbol = tick.get("symbol", "UNKNOWN")
        price = tick.get("price", 0.0)

        # Run engine tick to update state
        engine.run_ticks(1)

        # Get observation
        obs = np.array(engine.reset() if tick_count == 1 else engine.step(0.0, 0.0).observation)

        # Run inference
        action, _ = model.predict(obs, deterministic=True)
        strike_width_norm = float(action[0])
        delta_position_norm = float(action[1])

        # Denormalize actions
        strike_width = 2.5 + (strike_width_norm + 1.0) / 2.0 * 12.5
        target_delta = -0.05 - (delta_position_norm + 1.0) / 2.0 * 0.45

        # Compute confidence based on action magnitude and margin state
        action_magnitude = np.sqrt(strike_width_norm**2 + delta_position_norm**2) / np.sqrt(2)
        confidence = float(np.clip(action_magnitude, 0.0, 1.0))

        if confidence >= confidence_threshold:
            # Determine action type based on delta position
            if target_delta < -0.3:
                action_type = "sell"  # Aggressive short delta
            elif target_delta > -0.1:
                action_type = "close"  # Very low delta, close positions
            else:
                action_type = "buy"  # Open new spread

            # Create option legs for the signal
            short_strike = round(price * (1 + target_delta), 1)
            long_strike = round(short_strike - strike_width, 1)

            legs = [
                OptionLegSpec(
                    strike=short_strike,
                    expiry="",  # Would be set from actual expiry calendar
                    option_type="put",
                    side="sell",
                    quantity=1,
                ),
                OptionLegSpec(
                    strike=long_strike,
                    expiry="",
                    option_type="put",
                    side="buy",
                    quantity=1,
                ),
            ]

            signal = Signal(
                ticker=symbol,
                action_type=action_type,
                legs=legs,
                confidence=confidence,
                metadata={
                    "strike_width": strike_width,
                    "target_delta": target_delta,
                    "underlying_price": price,
                    "tick_count": tick_count,
                    "raw_action": action.tolist(),
                },
            )

            signal_bus.emit(signal)
            logger.info(
                f"Signal: {symbol} {action_type} | "
                f"width=${strike_width:.1f} delta={target_delta:.3f} "
                f"conf={confidence:.2f}"
            )

    # Connect and subscribe
    feed.connect()
    feed.subscribe(symbols)
    feed.on_tick(on_tick)

    logger.info(f"Live inference started for {symbols}")
    logger.info(f"Confidence threshold: {confidence_threshold}")

    # If using WebSocketFeed stub, start simulation
    if isinstance(feed, WebSocketFeed):
        logger.info("Running in simulation mode (WebSocketFeed stub)")
        try:
            feed.simulate_ticks(n_ticks=1000, interval=0.01)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            feed.disconnect()
