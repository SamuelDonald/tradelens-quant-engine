from __future__ import annotations

from typing import Dict, List

from engine.utils.indicators import atr_percentile


class VolatilityModel:
    """
    Detects volatility regime from ATR percentile over recent history.
    """

    def __init__(self, atr_period: int = 14, atr_window: int = 50):
        self.atr_period = atr_period
        self.atr_window = atr_window

    def analyze(self, candles: List[Dict]) -> Dict:
        atr_val, atr_pct = atr_percentile(
            candles, period=self.atr_period, window=self.atr_window
        )

        if atr_val is None or atr_pct is None:
            return {
                "atr": None,
                "volatility_regime": "normal",
                "compression": False,
            }

        if atr_pct < 33:
            regime = "low"
        elif atr_pct > 66:
            regime = "high"
        else:
            regime = "normal"

        compression = atr_pct < 25

        return {
            "atr": float(atr_val),
            "volatility_regime": regime,
            "compression": compression,
        }

