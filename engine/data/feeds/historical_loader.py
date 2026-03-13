from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from engine.data.data_manager import DataManager
from engine.data.feeds.base_feed import FeedError, validate_candle_dict
from engine.data.feeds.feed_router import FeedRouter
from engine.data.store.sqlite_store import SQLiteCandleStore


class HistoricalLoader:
    """
    Loads sufficient historical candles into per-timeframe DataManagers before the API starts serving.
    """

    def __init__(self, router: FeedRouter, store: Optional[SQLiteCandleStore] = None):
        self.router = router
        self.store = store

    async def load(
        self,
        data_managers: Dict[str, DataManager],
        symbols: List[str],
        timeframes: List[str],
        min_candles: int = 200,
    ) -> None:
        # Sequential per symbol/timeframe to reduce provider burst; can be parallelized later with limits.
        for timeframe in timeframes:
            dm = data_managers[timeframe]
            for symbol in symbols:
                candles: List[Dict] = []

                # 1) Try persistence first
                if self.store:
                    candles = self.store.get_recent(symbol, timeframe, min_candles)

                # 2) If insufficient, fetch from provider and persist
                if len(candles) < min_candles:
                    fetched = await asyncio.to_thread(
                        self.router.get_recent_candles, symbol, timeframe, min_candles
                    )
                    if fetched:
                        fetched = sorted(fetched, key=lambda c: c["timestamp"])
                        if self.store:
                            self.store.put_many(symbol, timeframe, fetched)
                        candles = fetched

                if not candles:
                    continue
                # Ensure sorted + valid; insert oldest->newest.
                candles = sorted(candles, key=lambda c: c["timestamp"])
                for c in candles:
                    err = validate_candle_dict(c)
                    if err:
                        raise FeedError(f"historical_invalid_candle:{symbol}:{timeframe}:{err}")
                    dm.update_market_data(symbol, c)

