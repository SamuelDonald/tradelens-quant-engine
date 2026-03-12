from __future__ import annotations

from typing import Dict, List

from engine.utils.indicators import find_swings


class StructureModel:
    """
    Detects directional structure (trend, HH/HL, BOS) from swing highs and lows.
    """

    def analyze(self, candles: List[Dict]) -> Dict:
        if len(candles) < 20:
            return {
                "trend": "neutral",
                "higher_highs": False,
                "higher_lows": False,
                "break_of_structure": False,
            }

        swing_highs, swing_lows = find_swings(candles, lookback=3)

        recent_highs = swing_highs[-3:]
        recent_lows = swing_lows[-3:]

        higher_highs = False
        higher_lows = False
        bos = False

        if len(recent_highs) >= 2:
            h1 = candles[recent_highs[-2]]["high"]
            h2 = candles[recent_highs[-1]]["high"]
            higher_highs = h2 > h1

        if len(recent_lows) >= 2:
            l1 = candles[recent_lows[-2]]["low"]
            l2 = candles[recent_lows[-1]]["low"]
            higher_lows = l2 > l1

        last_close = candles[-1]["close"]

        if recent_highs:
            last_swing_high = candles[recent_highs[-1]]["high"]
            if last_close > last_swing_high:
                bos = True

        if recent_lows:
            last_swing_low = candles[recent_lows[-1]]["low"]
            if last_close < last_swing_low:
                bos = True

        if higher_highs and higher_lows:
            trend = "bullish"
        elif (not higher_highs) and (not higher_lows):
            trend = "bearish"
        else:
            trend = "neutral"

        return {
            "trend": trend,
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "break_of_structure": bos,
        }

