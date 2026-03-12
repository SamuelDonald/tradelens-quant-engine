from __future__ import annotations

from typing import Dict, List

from engine.utils.indicators import _extract_closes, rsi, rate_of_change


class MomentumModel:
    """
    Measures momentum strength and direction using RSI and ROC.
    """

    def __init__(self, rsi_period: int = 14, roc_period: int = 5):
        self.rsi_period = rsi_period
        self.roc_period = roc_period

    def analyze(self, candles: List[Dict]) -> Dict:
        closes = _extract_closes(candles)
        rsi_val = rsi(closes, period=self.rsi_period)
        roc_val = rate_of_change(closes, period=self.roc_period)

        momentum_strength = 0.0
        direction = "neutral"

        if rsi_val is not None and roc_val is not None:
            rsi_component = (rsi_val - 50.0) / 50.0
            roc_component = roc_val
            combined = 0.7 * rsi_component + 0.3 * roc_component
            momentum_strength = float(combined)

            if combined > 0.1:
                direction = "up"
            elif combined < -0.1:
                direction = "down"

        return {
            "rsi": float(rsi_val) if rsi_val is not None else None,
            "momentum_strength": momentum_strength,
            "momentum_direction": direction,
        }

