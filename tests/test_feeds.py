"""Tests for the Alpaca and Tradier feed modules."""

from __future__ import annotations

import pytest

from delta_theta_matrix.feeds.alpaca_feed import AlpacaFeed
from delta_theta_matrix.feeds.tradier_feed import TradierFeed


# ── AlpacaFeed tests ─────────────────────────────────────────────────────


class TestAlpacaFeed:
    """Tests for AlpacaFeed error handling and validation."""

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API key and secret are required"):
            AlpacaFeed(api_key="", secret_key="")

    def test_missing_secret_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API key and secret are required"):
            AlpacaFeed(api_key="test_key", secret_key="")

    def test_creation_with_keys(self) -> None:
        feed = AlpacaFeed(api_key="test_key", secret_key="test_secret")
        assert feed._api_key == "test_key"
        assert feed._secret_key == "test_secret"

    def test_paper_mode_default(self) -> None:
        feed = AlpacaFeed(api_key="k", secret_key="s")
        assert feed._paper is True

    def test_live_mode(self) -> None:
        feed = AlpacaFeed(api_key="k", secret_key="s", paper=False)
        assert feed._paper is False

    def test_not_connected_raises(self) -> None:
        feed = AlpacaFeed(api_key="k", secret_key="s")
        with pytest.raises(RuntimeError, match="Not connected"):
            feed.get_quotes(["MU"])

    def test_subscribe(self) -> None:
        feed = AlpacaFeed(api_key="k", secret_key="s")
        feed.subscribe(["MU", "AMD"])
        assert feed._subscribed_symbols == ["MU", "AMD"]

    def test_on_tick(self) -> None:
        feed = AlpacaFeed(api_key="k", secret_key="s")
        cb_list: list = []
        feed.on_tick(cb_list.append)
        assert len(feed._tick_callbacks) == 1

    def test_disconnect_when_not_connected(self) -> None:
        feed = AlpacaFeed(api_key="k", secret_key="s")
        feed.disconnect()  # Should not raise

    def test_get_quotes_no_symbols_raises(self) -> None:
        feed = AlpacaFeed(api_key="k", secret_key="s")
        feed.connect()
        with pytest.raises(ValueError, match="No symbols specified"):
            feed.get_quotes()


# ── TradierFeed tests ────────────────────────────────────────────────────


class TestTradierFeed:
    """Tests for TradierFeed error handling and validation."""

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            TradierFeed(api_key="")

    def test_creation_with_key(self) -> None:
        feed = TradierFeed(api_key="test_key")
        assert feed._api_key == "test_key"

    def test_sandbox_mode_default(self) -> None:
        feed = TradierFeed(api_key="k")
        assert feed._sandbox is True

    def test_production_mode(self) -> None:
        feed = TradierFeed(api_key="k", sandbox=False)
        assert feed._sandbox is False
        assert "api.tradier.com" in feed._base_url

    def test_not_connected_raises(self) -> None:
        feed = TradierFeed(api_key="k")
        with pytest.raises(RuntimeError, match="Not connected"):
            feed.get_quotes(["MU"])

    def test_subscribe(self) -> None:
        feed = TradierFeed(api_key="k")
        feed.subscribe(["MU", "AMD"])
        assert feed._subscribed_symbols == ["MU", "AMD"]

    def test_on_tick(self) -> None:
        feed = TradierFeed(api_key="k")
        cb_list: list = []
        feed.on_tick(cb_list.append)
        assert len(feed._tick_callbacks) == 1

    def test_disconnect_when_not_connected(self) -> None:
        feed = TradierFeed(api_key="k")
        feed.disconnect()  # Should not raise

    def test_get_quotes_no_symbols_raises(self) -> None:
        feed = TradierFeed(api_key="k")
        feed.connect()
        with pytest.raises(ValueError, match="No symbols specified"):
            feed.get_quotes()

    def test_normalize_option(self) -> None:
        raw = {
            "symbol": "MU260320P00095000",
            "underlying": "MU",
            "strike": 95.0,
            "option_type": "put",
            "expiration_date": "2026-03-20",
            "bid": 1.50,
            "ask": 1.75,
            "last": 1.60,
            "volume": 100,
            "open_interest": 500,
            "greeks": {
                "delta": -0.30,
                "gamma": 0.05,
                "theta": -0.02,
                "vega": 0.15,
                "mid_iv": 0.28,
            },
        }
        result = TradierFeed._normalize_option(raw)
        assert result["symbol"] == "MU260320P00095000"
        assert result["strike"] == 95.0
        assert result["greeks"]["delta"] == -0.30
        assert result["greeks"]["mid_iv"] == 0.28

    def test_normalize_option_no_greeks(self) -> None:
        raw = {
            "symbol": "MU260320P00095000",
            "strike": 95.0,
        }
        result = TradierFeed._normalize_option(raw)
        assert "greeks" not in result
