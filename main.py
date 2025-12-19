import os
from fastapi import Depends, Header
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import ta

app = FastAPI(title="TradeLens Quant Engine", version="1.0.0")
QUANT_ENGINE_API_KEY = os.getenv("QUANT_ENGINE_API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if QUANT_ENGINE_API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured")
    if x_api_key != QUANT_ENGINE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------------------------
# Input Models
# -------------------------

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class QuantRequest(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Candle]
    indicators: Optional[Dict] = {}

# -------------------------
# Utility Functions
# -------------------------

def validate_candles(df: pd.DataFrame, lookback: int):
    if len(df) < lookback * 2:
        return False
    if df[['open','high','low','close']].std().sum() == 0:
        return False
    return True

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["EMA50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["ATR"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )
    return df

def classify_regime(df: pd.DataFrame):
    last = df.iloc[-1]
    atr_ratio = last["ATR"] / last["close"]
    ema_slope = df["EMA50"].iloc[-1] - df["EMA50"].iloc[-2]

    if atr_ratio < 0.002:
        return "LOW_VOL"
    if ema_slope > 0:
        return "TRENDING_UP"
    if ema_slope < 0:
        return "TRENDING_DOWN"
    return "RANGING"

def macd_bullish(df: pd.DataFrame):
    return (
        df["MACD"].iloc[-2] <= df["MACD_SIGNAL"].iloc[-2]
        and df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]
    )

def macd_bearish(df: pd.DataFrame):
    return (
        df["MACD"].iloc[-2] >= df["MACD_SIGNAL"].iloc[-2]
        and df["MACD"].iloc[-1] < df["MACD_SIGNAL"].iloc[-1]
    )

def walk_forward_confirm(df: pd.DataFrame):
    split = int(len(df) * 0.7)
    insample = df.iloc[:split]
    outsample = df.iloc[split:]

    insignal = insample["EMA20"].iloc[-1] > insample["EMA50"].iloc[-1]
    outsignal = outsample["EMA20"].iloc[-1] > outsample["EMA50"].iloc[-1]

    return insignal == outsignal

# -------------------------
# API Endpoints
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/quant/analyze", dependencies=[Depends(verify_api_key)])
def quant_engine_analysis(req: QuantRequest):
    df = pd.DataFrame([c.dict() for c in req.candles])

    lookback = 50
    if not validate_candles(df, lookback):
        return neutral_response(req)

    df = calculate_indicators(df)

    if df.isna().any().any():
        return neutral_response(req)

    if df["ATR"].iloc[-1] == 0:
        return neutral_response(req)

    regime = classify_regime(df)
    if regime == "LOW_VOL":
        return neutral_response(req, regime)

    bullish = (
        df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
        and df["RSI"].iloc[-1] < 70
        and macd_bullish(df)
        and regime == "TRENDING_UP"
    )

    bearish = (
        df["EMA20"].iloc[-1] < df["EMA50"].iloc[-1]
        and df["RSI"].iloc[-1] > 30
        and macd_bearish(df)
        and regime == "TRENDING_DOWN"
    )

    if not walk_forward_confirm(df):
        return neutral_response(req, regime)

    last_close = df["close"].iloc[-1]
    atr = df["ATR"].iloc[-1]

    if bullish:
        signal = "BUY"
        sl = last_close - atr
        tp = last_close + (2 * atr)
    elif bearish:
        signal = "SELL"
        sl = last_close + atr
        tp = last_close - (2 * atr)
    else:
        return neutral_response(req, regime)

    confidence = calculate_confidence(df, regime, signal)

    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "signal": signal,
        "signal_strength": confidence["strength"],
        "confidence_score": confidence["score"],
        "confidence_probability": confidence["probability"],
        "entry_price": last_close,
        "take_profit": round(tp, 5),
        "stop_loss": round(sl, 5),
        "regime": regime,
        "signal_validity": "next_candle",
        "conditions_met": confidence["conditions"],
        "indicators": {
            "EMA20": round(df["EMA20"].iloc[-1], 5),
            "EMA50": round(df["EMA50"].iloc[-1], 5),
            "RSI": round(df["RSI"].iloc[-1], 5),
            "MACD_line": round(df["MACD"].iloc[-1], 5),
            "MACD_signal": round(df["MACD_SIGNAL"].iloc[-1], 5),
            "ATR": round(atr, 5),
        },
    }

# -------------------------
# Helpers
# -------------------------

def neutral_response(req, regime="NEUTRAL"):
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
        "regime": regime,
        "signal_validity": None,
        "conditions_met": {
            "ema_trend": False,
            "macd_cross": False,
            "rsi_filter": False,
            "regime_filter": False,
        },
        "indicators": {},
    }

def calculate_confidence(df, regime, signal):
    weights = {
        "ema": 0.30,
        "macd": 0.30,
        "rsi": 0.20,
        "regime": 0.20,
    }

    ema_ok = (
        df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
        if signal == "BUY"
        else df["EMA20"].iloc[-1] < df["EMA50"].iloc[-1]
    )

    macd_ok = macd_bullish(df) if signal == "BUY" else macd_bearish(df)
    rsi_dist = abs(df["RSI"].iloc[-1] - 50) / 50
    regime_ok = regime in ["TRENDING_UP", "TRENDING_DOWN"]

    prob = (
        weights["ema"] * int(ema_ok)
        + weights["macd"] * int(macd_ok)
        + weights["rsi"] * min(rsi_dist, 1)
        + weights["regime"] * int(regime_ok)
    )

    score = round(prob * 100)
    strength = "WEAK"
    if score >= 75:
        strength = "STRONG"
    elif score >= 55:
        strength = "MODERATE"

    return {
        "probability": round(prob, 2),
        "score": score,
        "strength": strength,
        "conditions": {
            "ema_trend": ema_ok,
            "macd_cross": macd_ok,
            "rsi_filter": True,
            "regime_filter": regime_ok,
        },
    }
