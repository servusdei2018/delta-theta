"""
Live data ingestion, signal generation with pub/sub callbacks.

Provides:
- Abstract MarketDataFeed with connect/subscribe/on_tick
- WebSocketFeed with reconnect and exponential backoff
- Signal dataclass for trade signals
- SignalBus pub/sub system
- run_live() function for real-time inference with graceful shutdown

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
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the application.

    Sets up a consistent log format with timestamps, levels, and module names.

    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
    )


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
    Thread-safe: callbacks are protected by a lock.

    Example:
        bus = SignalBus()
        bus.register_callback(lambda s: print(f"Signal: {s.ticker} {s.action_type}"))
        bus.emit(Signal(ticker="MU", action_type="buy"))
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable[[Signal], None]] = []
        self._lock = threading.Lock()

    def register_callback(self, callback: Callable[[Signal], None]) -> None:
        """Register a callback to receive signals.

        Args:
            callback: Function that accepts a Signal object.
        """
        with self._lock:
            self._callbacks.append(callback)

    def emit(self, signal: Signal) -> None:
        """Emit a signal to all registered callbacks.

        Args:
            signal: The trade signal to broadcast.
        """
        with self._lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(signal)
            except Exception:
                logger.exception("Signal callback error")

    @property
    def callback_count(self) -> int:
        """Number of registered callbacks."""
        with self._lock:
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
    """WebSocket-based market data feed with reconnect and exponential backoff.

    Supports automatic reconnection on disconnect with configurable backoff.

    Args:
        url: WebSocket URL to connect to.
        api_key: Optional API key for authentication.
        max_reconnect_attempts: Maximum reconnection attempts (0 = infinite).
        initial_backoff: Initial backoff delay in seconds.
        max_backoff: Maximum backoff delay in seconds.
        backoff_multiplier: Multiplier for exponential backoff.
    """

    def __init__(
        self,
        url: str = "wss://example.com/market",
        api_key: str | None = None,
        max_reconnect_attempts: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self._connected = False
        self._subscribed_symbols: list[str] = []
        self._tick_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._shutdown_event = threading.Event()

        # Reconnect configuration
        self._max_reconnect_attempts = max_reconnect_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff_multiplier = backoff_multiplier
        self._reconnect_count = 0

    def connect(self) -> None:
        """Establish WebSocket connection (stub).

        In production, this would establish an actual WebSocket connection.
        """
        logger.info("Connecting to %s...", self.url)
        self._connected = True
        self._reconnect_count = 0
        logger.info("Connected (stub mode)")

    def disconnect(self) -> None:
        """Disconnect from WebSocket (stub)."""
        self._connected = False
        self._shutdown_event.set()
        logger.info("Disconnected")

    def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols (stub).

        Args:
            symbols: List of ticker symbols.
        """
        self._subscribed_symbols = symbols
        logger.info("Subscribed to: %s", symbols)

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register tick callback.

        Args:
            callback: Function to call on each tick.
        """
        self._tick_callbacks.append(callback)

    def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection succeeded, False if max attempts exceeded.
        """
        backoff = self._initial_backoff

        while (
            self._max_reconnect_attempts == 0
            or self._reconnect_count < self._max_reconnect_attempts
        ):
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested, aborting reconnect")
                return False

            self._reconnect_count += 1
            logger.warning(
                "Reconnect attempt %d/%s (backoff=%.1fs)",
                self._reconnect_count,
                self._max_reconnect_attempts or "∞",
                backoff,
            )

            # Wait with backoff (interruptible by shutdown event)
            if self._shutdown_event.wait(timeout=backoff):
                logger.info("Shutdown requested during backoff")
                return False

            try:
                self.connect()
                if self._subscribed_symbols:
                    self.subscribe(self._subscribed_symbols)
                logger.info("Reconnected successfully after %d attempts", self._reconnect_count)
                return True
            except Exception:
                logger.exception("Reconnect attempt %d failed", self._reconnect_count)

            # Exponential backoff with cap
            backoff = min(backoff * self._backoff_multiplier, self._max_backoff)

        logger.error(
            "Max reconnect attempts (%d) exceeded", self._max_reconnect_attempts
        )
        return False

    def simulate_ticks(self, n_ticks: int = 100, interval: float = 0.1) -> None:
        """Generate synthetic tick data for testing.

        Supports graceful shutdown via the shutdown event.

        Args:
            n_ticks: Number of ticks to generate.
            interval: Seconds between ticks.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        rng = np.random.default_rng(42)
        prices = {sym: 100.0 for sym in self._subscribed_symbols}

        for i in range(n_ticks):
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested, stopping tick simulation at tick %d", i)
                break

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
                    try:
                        cb(tick)
                    except Exception:
                        logger.exception("Tick callback error at tick %d", i)

            time.sleep(interval)

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the feed."""
        self._shutdown_event.set()


class GracefulShutdown:
    """Context manager for graceful SIGINT/SIGTERM handling.

    Usage:
        with GracefulShutdown() as shutdown:
            while not shutdown.is_set():
                do_work()
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._original_sigint = None
        self._original_sigterm = None

    def _handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, initiating graceful shutdown...", sig_name)
        self._event.set()

    def __enter__(self) -> threading.Event:
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
        return self._event

    def __exit__(self, *args: Any) -> None:
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def is_set(self) -> bool:
        """Check if shutdown has been requested."""
        return self._event.is_set()


def run_live(
    model_path: str,
    feed: MarketDataFeed,
    symbols: list[str],
    signal_bus: SignalBus,
    algo: str = "sac",
    confidence_threshold: float = 0.5,
) -> None:
    """Run live inference: load model, connect feed, emit signals on each tick.

    Includes graceful shutdown handling for SIGINT/SIGTERM.

    Args:
        model_path: Path to the trained SB3 model.
        feed: Market data feed instance.
        symbols: List of symbols to trade.
        signal_bus: Signal bus for emitting trade signals.
        algo: Algorithm used ('sac' or 'ppo').
        confidence_threshold: Minimum confidence to emit a signal.

    Raises:
        ValueError: If algo is not 'sac' or 'ppo'.
        FileNotFoundError: If model_path does not exist.
        RuntimeError: If model loading fails.
    """
    from stable_baselines3 import PPO, SAC

    from delta_theta_matrix import BacktestEngine, EngineConfig

    # Load the trained model with error handling
    logger.info("Loading %s model from %s", algo.upper(), model_path)
    try:
        if algo.lower() == "sac":
            model = SAC.load(model_path)
        elif algo.lower() == "ppo":
            model = PPO.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
    except FileNotFoundError:
        logger.error("Model file not found: %s", model_path)
        raise
    except Exception:
        logger.exception("Failed to load model from %s", model_path)
        raise RuntimeError(f"Failed to load model from {model_path}")

    # Create engine for state computation
    try:
        config = EngineConfig()
        engine = BacktestEngine(config)
        engine.reset()
    except Exception:
        logger.exception("Failed to initialize BacktestEngine")
        raise RuntimeError("Failed to initialize BacktestEngine")

    # Track state
    tick_count = 0

    def on_tick(tick: dict[str, Any]) -> None:
        nonlocal tick_count
        tick_count += 1

        symbol = tick.get("symbol", "UNKNOWN")
        price = tick.get("price", 0.0)

        if price <= 0.0:
            logger.warning("Invalid price %.4f for %s, skipping tick", price, symbol)
            return

        # Run engine tick to update state
        try:
            engine.run_ticks(1)

            # Get observation
            obs = np.array(
                engine.reset() if tick_count == 1 else engine.step(0.0, 0.0).observation
            )
        except Exception:
            logger.exception("Engine error at tick %d", tick_count)
            return

        # Run inference
        try:
            action, _ = model.predict(obs, deterministic=True)
        except Exception:
            logger.exception("Model prediction error at tick %d", tick_count)
            return

        strike_width_norm = float(action[0])
        delta_position_norm = float(action[1])

        # Denormalize actions
        strike_width = 2.5 + (strike_width_norm + 1.0) / 2.0 * 12.5
        target_delta = -0.05 - (delta_position_norm + 1.0) / 2.0 * 0.45

        # Compute confidence based on action magnitude and margin state
        action_magnitude = np.sqrt(
            strike_width_norm**2 + delta_position_norm**2
        ) / np.sqrt(2)
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

            signal_obj = Signal(
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

            signal_bus.emit(signal_obj)
            logger.info(
                "Signal: %s %s | width=$%.1f delta=%.3f conf=%.2f",
                symbol,
                action_type,
                strike_width,
                target_delta,
                confidence,
            )

    # Connect and subscribe with error handling
    try:
        feed.connect()
    except Exception:
        logger.exception("Failed to connect to market data feed")
        raise

    try:
        feed.subscribe(symbols)
    except Exception:
        logger.exception("Failed to subscribe to symbols: %s", symbols)
        feed.disconnect()
        raise

    feed.on_tick(on_tick)

    logger.info("Live inference started for %s", symbols)
    logger.info("Confidence threshold: %s", confidence_threshold)

    # If using WebSocketFeed stub, start simulation with graceful shutdown
    if isinstance(feed, WebSocketFeed):
        logger.info("Running in simulation mode (WebSocketFeed stub)")
        with GracefulShutdown() as shutdown_event:
            feed._shutdown_event = shutdown_event
            try:
                feed.simulate_ticks(n_ticks=1000, interval=0.01)
            except KeyboardInterrupt:
                logger.info("Stopped by user")
            finally:
                feed.disconnect()
    else:
        # For real feeds, just register the shutdown handler
        with GracefulShutdown():
            try:
                # Block until shutdown is requested
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopped by user")
            finally:
                feed.disconnect()
