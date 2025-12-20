from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ======================
# APP SETUP
# ======================

app = FastAPI(title="TradeLens Quant Engine", version="1.1.0")

API_KEY = "YOUR_API_KEY_HERE"  # use env var in production


# ======================
# DATA MODELS
# ======================

class Candle(BaseModel):
    o: float
    h: float
    l: float
    c: float
    v: Optional[float] = None


class QuantRequest(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Candle]
    indicators: Optional[Dict[str, Any]] = None


# ======================
# AUTH
# ======================

def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


# ======================
# UTILITIES
# ======================

def normalize_dataframe(candles: List[Candle]) -> pd.DataFrame:
    df = pd.DataFrame([c.dict() for c in candles])
    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })
    df = df.sort_index()
    return df


def adaptive_lookback(df: pd.DataFrame) -> int:
    return min(50, len(df) // 2)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()

    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["ATR"] = atr.average_true_range()

    return df


def classify_regime(df: pd.DataFrame) -> str:
    atr_ratio = df["ATR"].iloc[-1] / df["close"].iloc[-1]
    ema_slope = df["EMA50"].iloc[-1] - df["EMA50"].iloc[-5]

    if atr_ratio < 0.001:
        return "LOW_VOL"
    elif ema_slope > 0:
        return "TRENDING_UP"
    elif ema_slope < 0:
        return "TRENDING_DOWN"
    else:
        return "RANGING"


def macd_cross(df: pd.DataFrame) -> Optional[str]:
    if df["MACD"].iloc[-2] <= df["MACD_SIGNAL"].iloc[-2] and df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]:
        return "BULLISH"
    if df["MACD"].iloc[-2] >= df["MACD_SIGNAL"].iloc[-2] and df["MACD"].iloc[-1] < df["MACD_SIGNAL"].iloc[-1]:
        return "BEARISH"
    return None


def walk_forward_confirm(df: pd.DataFrame) -> bool:
    split = int(len(df) * 0.7)
    insample = df.iloc[:split]
    outsample = df.iloc[split:]

    return (
        insample["EMA20"].iloc[-1] > insample["EMA50"].iloc[-1]
        or outsample["EMA20"].iloc[-1] > outsample["EMA50"].iloc[-1]
    )


# ======================
# MAIN ENGINE
# ======================

@app.post("/quant/analyze")
def quant_engine_analysis(
    req: QuantRequest,
    x_api_key: str = Header(...)
):
    verify_api_key(x_api_key)

    if len(req.candles) < 20:
        return neutral_response(req, "INSUFFICIENT_DATA")

    df = normalize_dataframe(req.candles)
    lookback = adaptive_lookback(df)

    if lookback < 14:
        return neutral_response(req, "INSUFFICIENT_DATA")

    df = compute_indicators(df)
    df = df.dropna()

    if df.empty:
        return neutral_response(req, "INDICATOR_FAILURE")

    regime = classify_regime(df)
    rsi = df["RSI"].iloc[-1]
    macd_signal = macd_cross(df)

    wf_confirmed = walk_forward_confirm(df)
    confidence_penalty = 0.15 if not wf_confirmed else 0.0

    signal = "NEUTRAL"

    if regime == "TRENDING_UP" and macd_signal == "BULLISH" and rsi < 70:
        signal = "BUY"
    elif regime == "TRENDING_DOWN" and macd_signal == "BEARISH" and rsi > 30:
        signal = "SELL"
    elif regime == "LOW_VOL" and 45 < rsi < 55:
        signal = "NEUTRAL"

    atr = df["ATR"].iloc[-1]
    close = df["close"].iloc[-1]

    if signal == "BUY":
        sl = close - atr
        tp = close + 2 * atr
    elif signal == "SELL":
        sl = close + atr
        tp = close - 2 * atr
    else:
        sl = None
        tp = None

    base_confidence = 0.75 if signal != "NEUTRAL" else 0.0
    confidence_probability = max(0.0, base_confidence - confidence_penalty)
    confidence_score = int(confidence_probability * 100)

    strength = (
        "STRONG" if confidence_probability >= 0.75
        else "MODERATE" if confidence_probability >= 0.55
        else "WEAK"
    )

    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "signal": signal,
        "signal_strength": strength,
        "confidence_score": confidence_score,
        "confidence_probability": round(confidence_probability, 2),
        "entry_price": close,
        "take_profit": tp,
        "stop_loss": sl,
        "regime": regime,
        "signal_validity": "next_candle",
        "conditions_met": {
            "ema_trend": True if signal != "NEUTRAL" else False,
            "macd_cross": macd_signal is not None,
            "rsi_filter": True,
            "regime_filter": regime != "RANGING"
        },
        "indicators": {
            "EMA20": float(df["EMA20"].iloc[-1]),
            "EMA50": float(df["EMA50"].iloc[-1]),
            "RSI": float(rsi),
            "MACD_line": float(df["MACD"].iloc[-1]),
            "MACD_signal": float(df["MACD_SIGNAL"].iloc[-1]),
            "ATR": float(atr)
        },
        "debug": {
            "candles_received": len(df),
            "lookback_used": lookback,
            "walk_forward_confirmed": wf_confirmed
        }
    }


def neutral_response(req: QuantRequest, reason: str):
    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "signal": "NEUTRAL",
        "signal_strength": "WEAK",
        "confidence_score": 0,
        "confidence_probability": 0.0,
        "entry_price": None,
        "take_profit": None,
        "stop_loss": None,
        "regime": reason,
        "signal_validity": None,
        "conditions_met": {},
        "indicators": {},
        "debug": {
            "reason": reason
        }
    }
