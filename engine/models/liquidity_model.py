from __future__ import annotations

from typing import Dict, List

from engine.utils.indicators import find_swings


class LiquidityModel:
    """
    Detects basic liquidity structures: equal highs/lows and recent sweeps.
    """

    def __init__(self, tolerance_pct: float = 0.001):
        self.tolerance_pct = tolerance_pct

    def _approx_equal(self, a: float, b: float) -> bool:
        if b == 0:
            return False
        return abs(a - b) / b <= self.tolerance_pct

    def analyze(self, candles: List[Dict]) -> Dict:
        swing_highs, swing_lows = find_swings(candles, lookback=3)
        equal_highs = False
        equal_lows = False
        recent_sweep = False

        if len(swing_highs) >= 2:
            h1 = candles[swing_highs[-2]]["high"]
            h2 = candles[swing_highs[-1]]["high"]
            equal_highs = self._approx_equal(h1, h2)

        if len(swing_lows) >= 2:
            l1 = candles[swing_lows[-2]]["low"]
            l2 = candles[swing_lows[-1]]["low"]
            equal_lows = self._approx_equal(l1, l2)

        last = candles[-1]
        close = last["close"]
        high = last["high"]
        low = last["low"]

        if swing_highs:
            prev_high = candles[swing_highs[-1]]["high"]
            if high > prev_high and close < prev_high:
                recent_sweep = True

        if swing_lows and not recent_sweep:
            prev_low = candles[swing_lows[-1]]["low"]
            if low < prev_low and close > prev_low:
                recent_sweep = True

        return {
            "equal_highs": equal_highs,
            "equal_lows": equal_lows,
            "recent_liquidity_sweep": recent_sweep,
        }

