from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

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


class FinnhubFeed(BaseFeed):
    """
    Finnhub REST adapter (primarily forex candles).
    """

    BASE_URL = "https://finnhub.io/api/v1"

    _TIMEFRAMES: Dict[str, str] = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "D",
    }

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 10.0, max_retries: int = 3):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        if not self.api_key:
            raise FeedError("FINNHUB_API_KEY not set")

    def _request_json(self, path: str, params: Dict) -> Dict:
        url = f"{self.BASE_URL}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.get(url, params=params)
                if resp.status_code == 429:
                    raise FeedRateLimitError("finnhub_rate_limited")
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt == self.max_retries - 1:
                    raise FeedTimeoutError("finnhub_timeout") from e
            except FeedRateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1 + attempt)
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries - 1:
                    raise FeedError("finnhub_request_failed") from e
                time.sleep(1 + attempt)
        raise FeedError("finnhub_request_failed") from last_exc

    def get_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        if timeframe not in self._TIMEFRAMES:
            raise FeedError(f"unsupported_timeframe:{timeframe}")
        resolution = self._TIMEFRAMES[timeframe]

        now = int(time.time())
        # Conservative window: assume 60s for min resolution; scale roughly by limit.
        # For intraday resolutions, Finnhub requires from/to in unix seconds.
        seconds_per_bar = 60
        if resolution.isdigit():
            seconds_per_bar = int(resolution) * 60
        elif resolution == "D":
            seconds_per_bar = 86400

        frm = now - int(seconds_per_bar * max(limit, 10) * 1.5)

        # Using /forex/candle as requested.
        payload = self._request_json(
            "/forex/candle",
            params={
                "symbol": symbol,
                "resolution": resolution,
                "from": frm,
                "to": now,
                "token": self.api_key,
            },
        )

        if payload.get("s") not in (None, "ok"):
            # Finnhub uses s="no_data" etc.
            if payload.get("s") == "no_data":
                return []
            raise FeedMalformedResponseError(f"finnhub_status:{payload.get('s')}")

        ts = payload.get("t") or []
        opens = payload.get("o") or []
        highs = payload.get("h") or []
        lows = payload.get("l") or []
        closes = payload.get("c") or []
        vols = payload.get("v") or []

        if not (len(ts) == len(opens) == len(highs) == len(lows) == len(closes) == len(vols)):
            raise FeedMalformedResponseError("finnhub_length_mismatch")

        candles: List[Dict] = []
        for i in range(len(ts)):
            candle = Candle(
                symbol=symbol,
                timestamp=int(ts[i]),
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                volume=float(vols[i]),
            ).to_dict()
            err = validate_candle_dict(candle)
            if err:
                raise FeedMalformedResponseError(f"finnhub_invalid_candle:{err}")
            candles.append(candle)

        candles.sort(key=lambda c: c["timestamp"])
        return candles[-limit:]

    def get_latest_candle(self, symbol: str, timeframe: str) -> Dict:
        candles = self.get_recent_candles(symbol, timeframe, limit=2)
        if not candles:
            raise FeedMalformedResponseError("finnhub_no_data")
        return candles[-1]

