from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple, Optional

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


class PolygonFeed(BaseFeed):
    """
    Polygon REST adapter.

    Uses /v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan} and normalizes output
    to TradeLens standard candle dict format.
    """

    BASE_URL = "https://api.polygon.io"

    _TIMEFRAMES: Dict[str, Tuple[int, str]] = {
        "1m": (1, "minute"),
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "1h": (1, "hour"),
        "4h": (4, "hour"),
        "1d": (1, "day"),
    }

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 10.0, max_retries: int = 3):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        if not self.api_key:
            raise FeedError("POLYGON_API_KEY not set")

    def _request_json(self, path: str, params: Dict) -> Dict:
        url = f"{self.BASE_URL}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.get(url, params=params)
                if resp.status_code == 429:
                    raise FeedRateLimitError("polygon_rate_limited")
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt == self.max_retries - 1:
                    raise FeedTimeoutError("polygon_timeout") from e
            except FeedRateLimitError:
                # deterministic backoff (no jitter)
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1 + attempt)
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries - 1:
                    raise FeedError("polygon_request_failed") from e
                time.sleep(1 + attempt)
        raise FeedError("polygon_request_failed") from last_exc

    def _normalize_results(self, symbol: str, results: List[Dict]) -> List[Dict]:
        candles: List[Dict] = []
        for r in results:
            # Polygon uses ms timestamps (t) and o/h/l/c/v
            if "t" not in r:
                raise FeedMalformedResponseError("polygon_missing_timestamp")
            candle = Candle(
                symbol=symbol,
                timestamp=int(r["t"]) // 1000,
                open=float(r.get("o")),
                high=float(r.get("h")),
                low=float(r.get("l")),
                close=float(r.get("c")),
                volume=float(r.get("v", 0.0)),
            ).to_dict()
            err = validate_candle_dict(candle)
            if err:
                raise FeedMalformedResponseError(f"polygon_invalid_candle:{err}")
            candles.append(candle)
        candles.sort(key=lambda c: c["timestamp"])
        return candles

    def get_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        if timeframe not in self._TIMEFRAMES:
            raise FeedError(f"unsupported_timeframe:{timeframe}")
        multiplier, timespan = self._TIMEFRAMES[timeframe]

        # Request a range that should contain >= limit bars. Use a conservative window.
        # Polygon's aggs require 'from' and 'to' in YYYY-MM-DD or ms; easiest is ms epoch.
        now_ms = int(time.time() * 1000)
        # Rough duration per bar in seconds:
        seconds_per_bar = {
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }[timespan] * multiplier
        window_ms = int(seconds_per_bar * max(limit, 10) * 1.5 * 1000)
        from_ms = now_ms - window_ms

        path = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_ms}/{now_ms}"
        payload = self._request_json(path, params={"apiKey": self.api_key, "limit": 50000})
        results = payload.get("results") or []
        candles = self._normalize_results(symbol, results)
        return candles[-limit:]

    def get_latest_candle(self, symbol: str, timeframe: str) -> Dict:
        candles = self.get_recent_candles(symbol, timeframe, limit=2)
        if not candles:
            raise FeedMalformedResponseError("polygon_no_data")
        return candles[-1]

