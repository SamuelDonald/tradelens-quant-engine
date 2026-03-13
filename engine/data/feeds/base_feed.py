from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


class FeedError(Exception):
    """Base exception for market data feed failures."""


class FeedTimeoutError(FeedError):
    """Raised when a feed request times out."""


class FeedRateLimitError(FeedError):
    """Raised when a feed request is rate-limited."""


class FeedMalformedResponseError(FeedError):
    """Raised when a feed response is missing required fields or is invalid."""


@dataclass(frozen=True)
class Candle:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": int(self.timestamp),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }


class BaseFeed(ABC):
    @abstractmethod
    def get_latest_candle(self, symbol: str, timeframe: str) -> Dict:
        """
        Return the latest available candle for (symbol, timeframe) in standard format.
        """

    @abstractmethod
    def get_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """
        Return up to `limit` most recent candles for (symbol, timeframe) in standard format,
        ordered from oldest to newest.
        """


def validate_candle_dict(candle: Dict) -> Optional[str]:
    """
    Validate standard candle dict schema.
    Returns None if valid; otherwise returns a short error message.
    """
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    for k in required:
        if k not in candle:
            return f"missing_field:{k}"
    if not isinstance(candle["symbol"], str) or not candle["symbol"]:
        return "invalid_symbol"
    try:
        int(candle["timestamp"])
        float(candle["open"])
        float(candle["high"])
        float(candle["low"])
        float(candle["close"])
        float(candle["volume"])
    except Exception:
        return "invalid_numeric_fields"
    return None

