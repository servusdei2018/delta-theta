"""
Alpaca Markets Data API feed integration.

Fetches options chain snapshots, real-time/delayed stock quotes, and
historical bars from the Alpaca Data API.

Supports both paper (paper-api.alpaca.markets) and live (api.alpaca.markets)
endpoints, configurable via the ``ALPACA_PAPER`` environment variable.

Usage:
    from delta_theta_matrix.feeds import AlpacaFeed

    feed = AlpacaFeed()  # reads ALPACA_API_KEY / ALPACA_SECRET_KEY env vars
    feed.connect()
    feed.subscribe(["MU", "AMD"])

    # Fetch latest quotes
    quotes = feed.get_quotes(["MU", "AMD"])

    # Fetch options chain snapshot
    chain = feed.get_options_chain("MU", expiration_gte="2026-03-01")

    # Fetch historical bars
    bars = feed.get_history("MU", timeframe="1Day", start="2026-01-01", end="2026-02-01")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable

import pyreqwest

from delta_theta_matrix.live import MarketDataFeed

logger = logging.getLogger(__name__)

_LIVE_DATA_BASE = "https://data.alpaca.markets"
_PAPER_DATA_BASE = "https://data.alpaca.markets"  # data API is the same for paper
_LIVE_TRADING_BASE = "https://api.alpaca.markets"
_PAPER_TRADING_BASE = "https://paper-api.alpaca.markets"


class AlpacaFeedError(Exception):
    """Error raised by AlpacaFeed operations."""

    pass


class AlpacaFeed(MarketDataFeed):
    """Market data feed backed by the Alpaca Data API.

    All HTTP calls include proper error handling — no bare unwrap/raise_for_status.

    Args:
        api_key: Alpaca API key ID. Falls back to ``ALPACA_API_KEY`` env var.
        secret_key: Alpaca secret key. Falls back to ``ALPACA_SECRET_KEY`` env var.
        paper: Use paper-trading endpoint for account calls. Falls back to
               ``ALPACA_PAPER`` env var (``"true"`` → paper). Defaults to ``True``.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca API key and secret are required. "
                "Pass api_key=/secret_key= or set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        if paper is None:
            paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
        self._paper = paper

        self._data_base_url = _LIVE_DATA_BASE
        self._trading_base_url = (
            _PAPER_TRADING_BASE if self._paper else _LIVE_TRADING_BASE
        )
        self._timeout = timeout
        self._client: pyreqwest.Client | None = None
        self._subscribed_symbols: list[str] = []
        self._tick_callbacks: list[Callable[[dict[str, Any]], None]] = []

    # ── MarketDataFeed interface ─────────────────────────────────────────

    def connect(self) -> None:
        """Open an HTTP client session to Alpaca."""
        self._client = pyreqwest.Client(
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
                "Accept": "application/json",
            },
            timeout=self._timeout,
        )
        mode = "paper" if self._paper else "live"
        logger.info("AlpacaFeed connected (%s)", mode)

    def disconnect(self) -> None:
        """Close the HTTP client session."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.exception("Error closing Alpaca client")
            finally:
                self._client = None
        logger.info("AlpacaFeed disconnected")

    def subscribe(self, symbols: list[str]) -> None:
        """Register symbols for subsequent data requests.

        Args:
            symbols: Ticker symbols to track.
        """
        self._subscribed_symbols = list(symbols)
        logger.info("AlpacaFeed subscribed to %s", symbols)

    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked when new tick data arrives.

        Args:
            callback: Receives a dict with at least ``symbol``, ``price``,
                      ``timestamp`` keys.
        """
        self._tick_callbacks.append(callback)

    # ── Provider-specific methods ────────────────────────────────────────

    def get_options_chain(
        self,
        underlying_symbol: str,
        *,
        expiration_gte: str | None = None,
        expiration_lte: str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch options chain snapshots for *underlying_symbol*.

        Uses the Alpaca Options Snapshots endpoint.

        Args:
            underlying_symbol: Underlying ticker (e.g. ``"MU"``).
            expiration_gte: Minimum expiration date ``YYYY-MM-DD``.
            expiration_lte: Maximum expiration date ``YYYY-MM-DD``.
            strike_gte: Minimum strike price.
            strike_lte: Maximum strike price.
            option_type: ``"call"`` or ``"put"`` filter.
            limit: Maximum number of contracts to return.

        Returns:
            List of option snapshot dicts with keys: symbol, underlying,
            strike, option_type, expiration_date, bid, ask, last, volume,
            open_interest, iv, greeks.

        Raises:
            AlpacaFeedError: If the HTTP request fails.
        """
        self._ensure_connected()
        assert self._client is not None

        params: dict[str, Any] = {
            "underlying_symbols": underlying_symbol,
            "limit": limit,
        }
        if expiration_gte:
            params["expiration_date_gte"] = expiration_gte
        if expiration_lte:
            params["expiration_date_lte"] = expiration_lte
        if strike_gte is not None:
            params["strike_price_gte"] = str(strike_gte)
        if strike_lte is not None:
            params["strike_price_lte"] = str(strike_lte)
        if option_type:
            params["type"] = option_type

        url = f"{self._data_base_url}/v1beta1/options/snapshots"

        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(
                "Failed to fetch options chain for %s: %s",
                underlying_symbol,
                exc,
            )
            raise AlpacaFeedError(
                f"Failed to fetch options chain for {underlying_symbol}: {exc}"
            ) from exc

        snapshots = data.get("snapshots", {})
        results: list[dict[str, Any]] = []
        for contract_symbol, snap in snapshots.items():
            latest_quote = snap.get("latestQuote", {})
            latest_trade = snap.get("latestTrade", {})
            greeks = snap.get("greeks", {})
            results.append(
                {
                    "symbol": contract_symbol,
                    "underlying": underlying_symbol,
                    "bid": latest_quote.get("bp"),
                    "ask": latest_quote.get("ap"),
                    "last": latest_trade.get("p"),
                    "volume": latest_trade.get("s"),
                    "open_interest": snap.get("openInterest"),
                    "iv": snap.get("impliedVolatility"),
                    "greeks": {
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "theta": greeks.get("theta"),
                        "vega": greeks.get("vega"),
                    },
                }
            )
        return results

    def get_quotes(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        """Fetch latest stock quotes.

        Args:
            symbols: Tickers to quote. Defaults to subscribed symbols.

        Returns:
            List of quote dicts with keys: symbol, bid, ask, last, volume,
            timestamp.

        Raises:
            AlpacaFeedError: If the HTTP request fails.
            ValueError: If no symbols are specified.
        """
        self._ensure_connected()
        assert self._client is not None

        syms = symbols or self._subscribed_symbols
        if not syms:
            raise ValueError("No symbols specified and none subscribed.")

        url = f"{self._data_base_url}/v2/stocks/quotes/latest"

        try:
            resp = self._client.get(url, params={"symbols": ",".join(syms)})
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch quotes for %s: %s", syms, exc)
            raise AlpacaFeedError(f"Failed to fetch quotes for {syms}: {exc}") from exc

        quotes_map = data.get("quotes", {})
        normalized: list[dict[str, Any]] = []
        for sym, q in quotes_map.items():
            quote = {
                "symbol": sym,
                "bid": q.get("bp"),
                "ask": q.get("ap"),
                "bid_size": q.get("bs"),
                "ask_size": q.get("as"),
                "timestamp": q.get("t", datetime.now(timezone.utc).isoformat()),
            }
            # Derive a mid-price for the tick callback
            bid = q.get("bp", 0) or 0
            ask = q.get("ap", 0) or 0
            quote["last"] = (bid + ask) / 2 if (bid and ask) else bid or ask
            normalized.append(quote)

        # Fire tick callbacks
        for quote in normalized:
            tick = {
                "symbol": quote["symbol"],
                "price": quote["last"],
                "bid": quote["bid"],
                "ask": quote["ask"],
                "timestamp": quote["timestamp"],
            }
            for cb in self._tick_callbacks:
                try:
                    cb(tick)
                except Exception:
                    logger.exception("Tick callback error")

        return normalized

    def get_history(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch historical price bars.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe, e.g. ``"1Min"``, ``"1Hour"``, ``"1Day"``.
            start: Start datetime (RFC-3339 or ``YYYY-MM-DD``).
            end: End datetime (RFC-3339 or ``YYYY-MM-DD``).
            limit: Maximum number of bars.

        Returns:
            List of bar dicts with keys: timestamp, open, high, low, close,
            volume, vwap.

        Raises:
            AlpacaFeedError: If the HTTP request fails.
        """
        self._ensure_connected()
        assert self._client is not None

        params: dict[str, Any] = {"timeframe": timeframe}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit is not None:
            params["limit"] = limit

        url = f"{self._data_base_url}/v2/stocks/{symbol}/bars"

        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch history for %s: %s", symbol, exc)
            raise AlpacaFeedError(
                f"Failed to fetch history for {symbol}: {exc}"
            ) from exc

        bars = data.get("bars", []) or []
        return [
            {
                "timestamp": b.get("t"),
                "open": b.get("o"),
                "high": b.get("h"),
                "low": b.get("l"),
                "close": b.get("c"),
                "volume": b.get("v"),
                "vwap": b.get("vw"),
            }
            for b in bars
        ]

    # ── Helpers ──────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        """Ensure the client is connected.

        Raises:
            RuntimeError: If not connected.
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
