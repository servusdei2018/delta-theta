"""
Tradier Market Data API feed integration.

Fetches options chains (with Greeks and IV), real-time/delayed quotes,
and historical price data from the Tradier API.

Supports both sandbox (sandbox.tradier.com) and production (api.tradier.com)
endpoints, configurable via the ``TRADIER_SANDBOX`` environment variable.

Usage:
    from delta_theta_matrix.feeds import TradierFeed

    feed = TradierFeed()  # reads TRADIER_API_KEY env var
    feed.connect()
    feed.subscribe(["MU", "AMD"])

    # Fetch options chain with Greeks
    chain = feed.get_options_chain("MU", "2026-03-20")

    # Fetch a real-time quote
    quotes = feed.get_quotes(["MU", "AMD"])

    # Fetch historical bars
    history = feed.get_history("MU", interval="daily", start="2026-01-01", end="2026-02-01")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable

import pyreqwest

from delta_theta_matrix.live import MarketDataFeed

logger = logging.getLogger(__name__)

_SANDBOX_BASE = "https://sandbox.tradier.com/v1"
_PRODUCTION_BASE = "https://api.tradier.com/v1"


class TradierFeedError(Exception):
    """Error raised by TradierFeed operations."""

    pass


class TradierFeed(MarketDataFeed):
    """Market data feed backed by the Tradier REST API.

    All HTTP calls include proper error handling — no bare unwrap/raise_for_status.

    Args:
        api_key: Tradier API token. Falls back to ``TRADIER_API_KEY`` env var.
        sandbox: Use sandbox endpoint. Falls back to ``TRADIER_SANDBOX`` env var
                 (``"true"`` → sandbox, anything else → production).
                 Defaults to ``True`` when neither argument nor env var is set.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        sandbox: bool | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("TRADIER_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Tradier API key is required. Pass api_key= or set TRADIER_API_KEY."
            )

        if sandbox is None:
            sandbox = os.environ.get("TRADIER_SANDBOX", "true").lower() == "true"
        self._sandbox = sandbox

        self._base_url = _SANDBOX_BASE if self._sandbox else _PRODUCTION_BASE
        self._timeout = timeout
        self._client: pyreqwest.Client | None = None
        self._subscribed_symbols: list[str] = []
        self._tick_callbacks: list[Callable[[dict[str, Any]], None]] = []

    # ── MarketDataFeed interface ─────────────────────────────────────────

    def connect(self) -> None:
        """Open an HTTP client session to Tradier."""
        self._client = pyreqwest.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            timeout=self._timeout,
        )
        mode = "sandbox" if self._sandbox else "production"
        logger.info("TradierFeed connected (%s)", mode)

    def disconnect(self) -> None:
        """Close the HTTP client session."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.exception("Error closing Tradier client")
            finally:
                self._client = None
        logger.info("TradierFeed disconnected")

    def subscribe(self, symbols: list[str]) -> None:
        """Register symbols for subsequent data requests.

        Args:
            symbols: Ticker symbols to track.
        """
        self._subscribed_symbols = list(symbols)
        logger.info("TradierFeed subscribed to %s", symbols)

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
        symbol: str,
        expiration: str,
        greeks: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch an options chain for *symbol* at *expiration*.

        Args:
            symbol: Underlying ticker (e.g. ``"MU"``).
            expiration: Expiration date in ``YYYY-MM-DD`` format.
            greeks: Include Greeks and IV in the response.

        Returns:
            List of option contract dicts, each containing strike, bid, ask,
            last, volume, open_interest, option_type, and (if *greeks*) a
            nested ``greeks`` dict with delta, gamma, theta, vega, mid_iv.

        Raises:
            TradierFeedError: If the HTTP request fails.
        """
        self._ensure_connected()
        assert self._client is not None

        params: dict[str, str] = {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": str(greeks).lower(),
        }

        try:
            resp = self._client.get("/markets/options/chains", params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(
                "Failed to fetch options chain for %s exp=%s: %s",
                symbol,
                expiration,
                exc,
            )
            raise TradierFeedError(
                f"Failed to fetch options chain for {symbol} exp={expiration}: {exc}"
            ) from exc

        options = data.get("options", {})
        if options is None:
            return []
        raw = options.get("option", [])
        if isinstance(raw, dict):
            raw = [raw]
        return [self._normalize_option(opt) for opt in raw]

    def get_options_expirations(self, symbol: str) -> list[str]:
        """Return available option expiration dates for *symbol*.

        Args:
            symbol: Underlying ticker.

        Returns:
            Sorted list of expiration date strings (``YYYY-MM-DD``).

        Raises:
            TradierFeedError: If the HTTP request fails.
        """
        self._ensure_connected()
        assert self._client is not None

        try:
            resp = self._client.get(
                "/markets/options/expirations", params={"symbol": symbol}
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error(
                "Failed to fetch expirations for %s: %s", symbol, exc
            )
            raise TradierFeedError(
                f"Failed to fetch expirations for {symbol}: {exc}"
            ) from exc

        expirations = data.get("expirations", {})
        if expirations is None:
            return []
        dates = expirations.get("date", [])
        if isinstance(dates, str):
            dates = [dates]
        return sorted(dates)

    def get_quotes(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        """Fetch real-time (or delayed) quotes.

        Args:
            symbols: Tickers to quote. Defaults to subscribed symbols.

        Returns:
            List of quote dicts with keys: symbol, last, bid, ask, volume,
            change, change_pct, timestamp.

        Raises:
            TradierFeedError: If the HTTP request fails.
            ValueError: If no symbols are specified.
        """
        self._ensure_connected()
        assert self._client is not None

        syms = symbols or self._subscribed_symbols
        if not syms:
            raise ValueError("No symbols specified and none subscribed.")

        try:
            resp = self._client.get(
                "/markets/quotes", params={"symbols": ",".join(syms)}
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch quotes for %s: %s", syms, exc)
            raise TradierFeedError(
                f"Failed to fetch quotes for {syms}: {exc}"
            ) from exc

        quotes_wrapper = data.get("quotes", {})
        raw = quotes_wrapper.get("quote", [])
        if isinstance(raw, dict):
            raw = [raw]

        normalized: list[dict[str, Any]] = []
        for q in raw:
            normalized.append(
                {
                    "symbol": q.get("symbol"),
                    "last": q.get("last"),
                    "bid": q.get("bid"),
                    "ask": q.get("ask"),
                    "volume": q.get("volume"),
                    "change": q.get("change"),
                    "change_pct": q.get("change_percentage"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Fire tick callbacks
        for quote in normalized:
            tick = {
                "symbol": quote["symbol"],
                "price": quote["last"],
                "bid": quote["bid"],
                "ask": quote["ask"],
                "volume": quote["volume"],
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
        interval: str = "daily",
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch historical price bars.

        Args:
            symbol: Ticker symbol.
            interval: One of ``"daily"``, ``"weekly"``, ``"monthly"``.
            start: Start date ``YYYY-MM-DD`` (inclusive).
            end: End date ``YYYY-MM-DD`` (inclusive).

        Returns:
            List of bar dicts with keys: date, open, high, low, close, volume.

        Raises:
            TradierFeedError: If the HTTP request fails.
        """
        self._ensure_connected()
        assert self._client is not None

        params: dict[str, str] = {"symbol": symbol, "interval": interval}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        try:
            resp = self._client.get("/markets/history", params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch history for %s: %s", symbol, exc)
            raise TradierFeedError(
                f"Failed to fetch history for {symbol}: {exc}"
            ) from exc

        history = data.get("history", {})
        if history is None:
            return []
        days = history.get("day", [])
        if isinstance(days, dict):
            days = [days]
        return [
            {
                "date": d.get("date"),
                "open": d.get("open"),
                "high": d.get("high"),
                "low": d.get("low"),
                "close": d.get("close"),
                "volume": d.get("volume"),
            }
            for d in days
        ]

    # ── Helpers ──────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        """Ensure the client is connected.

        Raises:
            RuntimeError: If not connected.
        """
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")

    @staticmethod
    def _normalize_option(opt: dict[str, Any]) -> dict[str, Any]:
        """Normalise a single Tradier option contract dict."""
        result: dict[str, Any] = {
            "symbol": opt.get("symbol"),
            "underlying": opt.get("underlying"),
            "strike": opt.get("strike"),
            "option_type": opt.get("option_type"),
            "expiration_date": opt.get("expiration_date"),
            "bid": opt.get("bid"),
            "ask": opt.get("ask"),
            "last": opt.get("last"),
            "volume": opt.get("volume"),
            "open_interest": opt.get("open_interest"),
        }
        greeks = opt.get("greeks")
        if greeks:
            result["greeks"] = {
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
                "mid_iv": greeks.get("mid_iv"),
            }
        return result
