from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

from engine.data.feeds.base_feed import (
    BaseFeed,
    Candle,
    FeedError,
    FeedMalformedResponseError,
    FeedRateLimitError,
    FeedTimeoutError,
    validate_candle_dict,
)


class AlphaVantageFeed(BaseFeed):
    """
    Alpha Vantage fallback adapter.

    NOTE: Alpha Vantage returns timestamps as strings in the response keys. We parse them
    deterministically as UTC (timezone-naive strings are treated as UTC). This may differ
    from exchange-local semantics; document and adjust when you add a symbol metadata layer.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    _TIMEFRAMES: Dict[str, str] = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "60min",
        # No direct 4h interval in AlphaVantage intraday; fallback will approximate using 60min*4 if needed.
        "4h": "60min",
        "1d": "daily",
    }

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 12.0, max_retries: int = 3):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        if not self.api_key:
            raise FeedError("ALPHAVANTAGE_API_KEY not set")

    def _request_json(self, params: Dict) -> Dict:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.get(self.BASE_URL, params=params)
                if resp.status_code == 429:
                    raise FeedRateLimitError("alphavantage_rate_limited")
                resp.raise_for_status()
                payload = resp.json()
                # AlphaVantage sometimes returns throttling notes with 200
                if "Note" in payload:
                    raise FeedRateLimitError("alphavantage_rate_limited_note")
                if "Error Message" in payload:
                    raise FeedMalformedResponseError("alphavantage_error_message")
                return payload
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt == self.max_retries - 1:
                    raise FeedTimeoutError("alphavantage_timeout") from e
            except FeedRateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1 + attempt)
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries - 1:
                    raise FeedError("alphavantage_request_failed") from e
                time.sleep(1 + attempt)
        raise FeedError("alphavantage_request_failed") from last_exc

    def _parse_timestamp_utc(self, ts_str: str) -> int:
        # Common formats: "2026-03-13 11:00:00" or "2026-03-13"
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except ValueError:
                continue
        raise FeedMalformedResponseError("alphavantage_bad_timestamp")

    def _extract_series(self, payload: Dict) -> Tuple[str, Dict]:
        # Find the first key that looks like a time series block
        for k, v in payload.items():
            if isinstance(v, dict) and ("Time Series" in k or "FX Intraday" in k or "Time Series FX" in k):
                return k, v
        raise FeedMalformedResponseError("alphavantage_missing_time_series")

    def _normalize_series(self, symbol: str, series: Dict) -> List[Dict]:
        candles: List[Dict] = []
        for ts_str, row in series.items():
            ts = self._parse_timestamp_utc(ts_str)
            # Keys vary: '1. open', '2. high', etc. Volume missing in FX.
            o = row.get("1. open") or row.get("1a. open (USD)") or row.get("1b. open (USD)")
            h = row.get("2. high") or row.get("2a. high (USD)") or row.get("2b. high (USD)")
            l = row.get("3. low") or row.get("3a. low (USD)") or row.get("3b. low (USD)")
            c = row.get("4. close") or row.get("4a. close (USD)") or row.get("4b. close (USD)")
            v = row.get("5. volume")
            candle = Candle(
                symbol=symbol,
                timestamp=ts,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v) if v is not None else 0.0,
            ).to_dict()
            err = validate_candle_dict(candle)
            if err:
                raise FeedMalformedResponseError(f"alphavantage_invalid_candle:{err}")
            candles.append(candle)
        candles.sort(key=lambda c: c["timestamp"])
        return candles

    def get_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        if timeframe not in self._TIMEFRAMES:
            raise FeedError(f"unsupported_timeframe:{timeframe}")

        interval = self._TIMEFRAMES[timeframe]

        # Heuristic: FX symbols are often 6 chars like EURUSD; stocks are letters.
        is_fx = symbol.isalpha() and len(symbol) == 6

        if timeframe == "1d":
            # Daily: use TIME_SERIES_DAILY (stocks) or FX_DAILY (forex). Not specified in prompt,
            # but we keep fallback minimal; daily is optional for Step 2 ingestion.
            function = "FX_DAILY" if is_fx else "TIME_SERIES_DAILY"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "compact",
            }
        else:
            function = "FX_INTRADAY" if is_fx else "TIME_SERIES_INTRADAY"
            params = {
                "function": function,
                "symbol": symbol,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": "compact",
            }

        payload = self._request_json(params)
        _, series = self._extract_series(payload)
        candles = self._normalize_series(symbol, series)

        if timeframe == "4h":
            # Approximate 4h by taking every 4th 60min bar (deterministic downsample).
            candles = candles[::4]

        return candles[-limit:]

    def get_latest_candle(self, symbol: str, timeframe: str) -> Dict:
        candles = self.get_recent_candles(symbol, timeframe, limit=2)
        if not candles:
            raise FeedMalformedResponseError("alphavantage_no_data")
        return candles[-1]

