from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class Candle:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MockMarketDataFeed:
    """
    Deterministic mock feed that generates synthetic OHLCV data.
    For a given symbol and sequence length, the output is reproducible.
    """

    def __init__(self, base_price: float = 100.0, base_volume: float = 1000.0, interval_sec: int = 60):
        self.base_price = base_price
        self.base_volume = base_volume
        self.interval_sec = interval_sec
        self._state: Dict[str, Candle] = {}

    def _init_symbol_state(self, symbol: str) -> None:
        now = int(time.time())
        price = self.base_price * (1.0 + (hash(symbol) % 50) / 100.0)
        candle = Candle(
            symbol=symbol,
            timestamp=now - self.interval_sec,
            open=price,
            high=price * 1.001,
            low=price * 0.999,
            close=price,
            volume=self.base_volume,
        )
        self._state[symbol] = candle

    def get_next_candle(self, symbol: str) -> Dict[str, Any]:
        """
        Generate the next deterministic candle for symbol.
        """
        if symbol not in self._state:
            self._init_symbol_state(symbol)

        prev = self._state[symbol]

        step = (prev.timestamp // self.interval_sec) % 360
        angle = math.radians(step)
        trend_component = 0.0005
        cycle_component = 0.003 * math.sin(angle)
        price_change_pct = trend_component + cycle_component

        new_close = prev.close * (1.0 + price_change_pct)
        midpoint = (prev.close + new_close) / 2.0

        vol_factor = 0.002 + 0.001 * math.cos(angle)
        high = max(new_close, prev.close) * (1.0 + vol_factor)
        low = min(new_close, prev.close) * (1.0 - vol_factor)

        volume = self.base_volume * (1.0 + 0.2 * math.sin(angle * 2))

        candle = Candle(
            symbol=symbol,
            timestamp=prev.timestamp + self.interval_sec,
            open=prev.close,
            high=high,
            low=low,
            close=new_close,
            volume=max(volume, 1.0),
        )
        self._state[symbol] = candle
        return asdict(candle)

