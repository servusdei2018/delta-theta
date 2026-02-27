"""Tests for the live module: Signal, SignalBus, WebSocketFeed, GracefulShutdown."""

from __future__ import annotations

import threading

import pytest

from delta_theta_matrix.live import (
    GracefulShutdown,
    OptionLegSpec,
    Signal,
    SignalBus,
    WebSocketFeed,
    setup_logging,
)


# ── Signal tests ─────────────────────────────────────────────────────────


class TestSignal:
    """Tests for the Signal dataclass."""

    def test_signal_creation(self) -> None:
        sig = Signal(ticker="MU", action_type="buy")
        assert sig.ticker == "MU"
        assert sig.action_type == "buy"
        assert sig.confidence == 0.0
        assert sig.legs == []
        assert sig.timestamp  # Should have a default timestamp

    def test_signal_with_legs(self) -> None:
        legs = [
            OptionLegSpec(
                strike=95.0, expiry="2026-03-20", option_type="put", side="sell"
            ),
            OptionLegSpec(
                strike=90.0, expiry="2026-03-20", option_type="put", side="buy"
            ),
        ]
        sig = Signal(ticker="MU", action_type="sell", legs=legs, confidence=0.85)
        assert len(sig.legs) == 2
        assert sig.legs[0].strike == 95.0
        assert sig.confidence == 0.85

    def test_signal_to_dict(self) -> None:
        sig = Signal(ticker="AMD", action_type="close", confidence=0.5)
        d = sig.to_dict()
        assert d["ticker"] == "AMD"
        assert d["action_type"] == "close"
        assert d["confidence"] == 0.5
        assert isinstance(d["legs"], list)
        assert isinstance(d["metadata"], dict)

    def test_signal_to_json(self) -> None:
        sig = Signal(ticker="MU", action_type="buy")
        json_str = sig.to_json()
        assert '"ticker": "MU"' in json_str
        assert '"action_type": "buy"' in json_str

    def test_signal_metadata(self) -> None:
        sig = Signal(
            ticker="MU",
            action_type="buy",
            metadata={"strike_width": 5.0, "target_delta": -0.3},
        )
        assert sig.metadata["strike_width"] == 5.0


class TestOptionLegSpec:
    """Tests for the OptionLegSpec dataclass."""

    def test_leg_creation(self) -> None:
        leg = OptionLegSpec(
            strike=95.0, expiry="2026-03-20", option_type="put", side="sell"
        )
        assert leg.strike == 95.0
        assert leg.expiry == "2026-03-20"
        assert leg.option_type == "put"
        assert leg.side == "sell"
        assert leg.quantity == 1

    def test_leg_custom_quantity(self) -> None:
        leg = OptionLegSpec(
            strike=95.0, expiry="2026-03-20", option_type="put", side="sell", quantity=5
        )
        assert leg.quantity == 5


# ── SignalBus tests ──────────────────────────────────────────────────────


class TestSignalBus:
    """Tests for the SignalBus pub/sub system."""

    def test_empty_bus(self) -> None:
        bus = SignalBus()
        assert bus.callback_count == 0

    def test_register_callback(self) -> None:
        bus = SignalBus()
        bus.register_callback(lambda s: None)
        assert bus.callback_count == 1

    def test_emit_signal(self) -> None:
        bus = SignalBus()
        received: list[Signal] = []
        bus.register_callback(received.append)

        sig = Signal(ticker="MU", action_type="buy")
        bus.emit(sig)

        assert len(received) == 1
        assert received[0].ticker == "MU"

    def test_multiple_callbacks(self) -> None:
        bus = SignalBus()
        counts = [0, 0]

        def cb1(s: Signal) -> None:
            counts[0] += 1

        def cb2(s: Signal) -> None:
            counts[1] += 1

        bus.register_callback(cb1)
        bus.register_callback(cb2)

        bus.emit(Signal(ticker="MU", action_type="buy"))
        assert counts == [1, 1]

    def test_callback_error_does_not_stop_others(self) -> None:
        bus = SignalBus()
        received: list[Signal] = []

        def bad_callback(s: Signal) -> None:
            raise ValueError("test error")

        bus.register_callback(bad_callback)
        bus.register_callback(received.append)

        bus.emit(Signal(ticker="MU", action_type="buy"))
        assert len(received) == 1  # Second callback still ran

    def test_thread_safety(self) -> None:
        bus = SignalBus()
        count = [0]
        lock = threading.Lock()

        def cb(s: Signal) -> None:
            with lock:
                count[0] += 1

        bus.register_callback(cb)

        threads = []
        for _ in range(10):
            t = threading.Thread(
                target=lambda: bus.emit(Signal(ticker="MU", action_type="buy"))
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert count[0] == 10


# ── WebSocketFeed tests ──────────────────────────────────────────────────


class TestWebSocketFeed:
    """Tests for the WebSocketFeed with reconnect and backoff."""

    def test_creation(self) -> None:
        feed = WebSocketFeed()
        assert not feed._connected
        assert feed.url == "wss://example.com/market"

    def test_connect_disconnect(self) -> None:
        feed = WebSocketFeed()
        feed.connect()
        assert feed._connected
        feed.disconnect()
        assert not feed._connected

    def test_subscribe(self) -> None:
        feed = WebSocketFeed()
        feed.connect()
        feed.subscribe(["MU", "AMD"])
        assert feed._subscribed_symbols == ["MU", "AMD"]

    def test_on_tick(self) -> None:
        feed = WebSocketFeed()
        ticks: list[dict] = []
        feed.on_tick(ticks.append)
        assert len(feed._tick_callbacks) == 1

    def test_simulate_ticks_not_connected(self) -> None:
        feed = WebSocketFeed()
        with pytest.raises(RuntimeError, match="Not connected"):
            feed.simulate_ticks(n_ticks=1)

    def test_simulate_ticks(self) -> None:
        feed = WebSocketFeed()
        feed.connect()
        feed.subscribe(["MU"])

        ticks: list[dict] = []
        feed.on_tick(ticks.append)
        feed.simulate_ticks(n_ticks=5, interval=0.001)

        assert len(ticks) == 5
        assert all("symbol" in t for t in ticks)
        assert all("price" in t for t in ticks)

    def test_simulate_ticks_shutdown(self) -> None:
        feed = WebSocketFeed()
        feed.connect()
        feed.subscribe(["MU"])

        ticks: list[dict] = []
        feed.on_tick(ticks.append)

        # Request shutdown immediately
        feed.request_shutdown()
        feed.simulate_ticks(n_ticks=1000, interval=0.001)

        # Should have stopped early
        assert len(ticks) < 1000

    def test_reconnect_backoff(self) -> None:
        feed = WebSocketFeed(
            max_reconnect_attempts=3,
            initial_backoff=0.01,
            max_backoff=0.05,
            backoff_multiplier=2.0,
        )
        # Feed starts disconnected, reconnect should succeed (stub always connects)
        result = feed.reconnect()
        assert result is True
        assert feed._connected

    def test_reconnect_shutdown_during_backoff(self) -> None:
        feed = WebSocketFeed(
            max_reconnect_attempts=10,
            initial_backoff=10.0,  # Long backoff
        )
        feed.request_shutdown()
        result = feed.reconnect()
        assert result is False

    def test_custom_reconnect_params(self) -> None:
        feed = WebSocketFeed(
            max_reconnect_attempts=5,
            initial_backoff=0.5,
            max_backoff=30.0,
            backoff_multiplier=3.0,
        )
        assert feed._max_reconnect_attempts == 5
        assert feed._initial_backoff == 0.5
        assert feed._max_backoff == 30.0
        assert feed._backoff_multiplier == 3.0

    def test_tick_callback_error_handling(self) -> None:
        feed = WebSocketFeed()
        feed.connect()
        feed.subscribe(["MU"])

        good_ticks: list[dict] = []

        def bad_cb(tick: dict) -> None:
            raise ValueError("test error")

        feed.on_tick(bad_cb)
        feed.on_tick(good_ticks.append)

        # Should not raise despite bad callback
        feed.simulate_ticks(n_ticks=2, interval=0.001)
        assert len(good_ticks) == 2


# ── GracefulShutdown tests ───────────────────────────────────────────────


class TestGracefulShutdown:
    """Tests for the GracefulShutdown context manager."""

    def test_context_manager(self) -> None:
        with GracefulShutdown() as event:
            assert not event.is_set()

    def test_is_set_method(self) -> None:
        gs = GracefulShutdown()
        assert not gs.is_set()


# ── setup_logging tests ──────────────────────────────────────────────────


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_logging_does_not_raise(self) -> None:
        setup_logging()  # Should not raise

    def test_setup_logging_custom_level(self) -> None:
        import logging

        setup_logging(level=logging.DEBUG)
