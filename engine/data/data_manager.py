from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Any, Optional

from engine.data.collectors.mock_feed import MockMarketDataFeed


class DataManager:
    """
    Manages in-memory OHLCV history per symbol.

    Candle schema:
    {
        "symbol": str,
        "timestamp": int,
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
    }
    """

    def __init__(self, max_candles: int = 500):
        self.max_candles = max_candles
        self._candles: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_candles)
        )

    def update_market_data(self, symbol: str, candle: Dict[str, Any]) -> None:
        self._candles[symbol].append(candle)

    def get_latest_candle(self, symbol: str) -> Optional[Dict[str, Any]]:
        candles = self._candles.get(symbol)
        if not candles:
            return None
        return candles[-1]

    def get_last_n_candles(self, symbol: str, n: int) -> List[Dict[str, Any]]:
        candles = self._candles.get(symbol)
        if not candles or n <= 0:
            return []
        return list(candles)[-n:]

    def ensure_history(self, symbol: str, feed: MockMarketDataFeed, min_candles: int = 100) -> None:
        """
        Fill up history for symbol from mock feed until we have at least min_candles.
        Deterministic because feed itself is deterministic.
        """
        while len(self._candles[symbol]) < min_candles:
            candle = feed.get_next_candle(symbol)
            self.update_market_data(symbol, candle)

