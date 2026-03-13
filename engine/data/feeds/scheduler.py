from __future__ import annotations

import asyncio
import time
from typing import Dict, List

from engine.data.data_manager import DataManager
from engine.data.feeds.base_feed import validate_candle_dict
from engine.data.feeds.feed_router import FeedRouter


class MarketDataScheduler:
    """
    Continuously updates DataManagers with the latest candles.

    Async-compatible: network calls are run in a worker thread to avoid blocking the event loop.
    """

    def __init__(
        self,
        router: FeedRouter,
        data_managers: Dict[str, DataManager],
        symbols: List[str],
        timeframes: List[str],
        poll_seconds: int = 60,
    ):
        self.router = router
        self.data_managers = data_managers
        self.symbols = symbols
        self.timeframes = timeframes
        self.poll_seconds = poll_seconds
        self._running = False

    def _should_append(self, dm: DataManager, symbol: str, candle: Dict) -> bool:
        latest = dm.get_latest_candle(symbol)
        if latest is None:
            return True
        return int(candle["timestamp"]) > int(latest["timestamp"])

    async def run(self) -> None:
        self._running = True
        while self._running:
            start = time.time()
            for timeframe in self.timeframes:
                dm = self.data_managers[timeframe]
                for symbol in self.symbols:
                    candle = await asyncio.to_thread(
                        self.router.get_latest_candle, symbol, timeframe
                    )
                    err = validate_candle_dict(candle)
                    if err:
                        continue
                    if self._should_append(dm, symbol, candle):
                        dm.update_market_data(symbol, candle)

            elapsed = time.time() - start
            sleep_for = max(0.0, self.poll_seconds - elapsed)
            await asyncio.sleep(sleep_for)

