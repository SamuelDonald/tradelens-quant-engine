from __future__ import annotations

from typing import List, Dict, Tuple, Optional


def _extract_closes(candles: List[Dict]) -> List[float]:
    return [c["close"] for c in candles]


def rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, period + 1):
        delta = prices[-i] - prices[-i - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-delta)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rate_of_change(prices: List[float], period: int = 5) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    prev = prices[-period - 1]
    if prev == 0:
        return None
    return (prices[-1] - prev) / prev


def atr(candles: List[Dict], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs: List[float] = []
    for i in range(-period, 0):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / period


def atr_percentile(
    candles: List[Dict], period: int = 14, window: int = 50
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute ATR and where the latest ATR sits as a percentile of recent ATR values.
    """
    if len(candles) < period + window:
        return None, None

    atr_values: List[float] = []
    for i in range(-window, 0):
        sub = candles[: i + len(candles)]
        val = atr(sub, period=period)
        if val is None:
            return None, None
        atr_values.append(val)

    latest_atr = atr_values[-1]
    sorted_vals = sorted(atr_values)
    rank = sorted_vals.index(latest_atr)
    pct = 100.0 * rank / max(len(sorted_vals) - 1, 1)
    return latest_atr, pct


def find_swings(
    candles: List[Dict],
    lookback: int = 3,
) -> Tuple[List[int], List[int]]:
    """
    Very simple swing high/low detection.
    Returns lists of indices for swing highs and swing lows.
    """
    highs: List[int] = []
    lows: List[int] = []
    if len(candles) < 2 * lookback + 1:
        return highs, lows

    for i in range(lookback, len(candles) - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]
        left = candles[i - lookback : i]
        right = candles[i + 1 : i + 1 + lookback]

        if all(high > c["high"] for c in left + right):
            highs.append(i)
        if all(low < c["low"] for c in left + right):
            lows.append(i)
    return highs, lows

