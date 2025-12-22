from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os

app = FastAPI()

# =======================
# API KEY GUARD
# =======================
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME)
ENGINE_API_KEY = os.getenv("ENGINE_API_KEY")

def verify_key(key: str = Depends(api_key_header)):
    if key != ENGINE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# =======================
# DATA MODELS
# =======================
class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0.0

class QuantRequest(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Candle]

# =======================
# UTILITY FUNCTIONS
# =======================
def aggregate_to_4h(df: pd.DataFrame, factor: int = 4) -> Optional[pd.DataFrame]:
    """
    Aggregate 1H candles into 4H candles.
    """
    if len(df) < factor * 10:
        return None

    rows = []
    for i in range(0, len(df), factor):
        chunk = df.iloc[i:i + factor]
        if len(chunk) < factor:
            continue

        rows.append({
            "open": chunk.iloc[0]["open"],
            "high": chunk["high"].max(),
            "low": chunk["low"].min(),
            "close": chunk.iloc[-1]["close"],
            "volume": chunk["volume"].sum()
        })

    return pd.DataFrame(rows)

def detect_swings(df: pd.DataFrame, lookback: int = 2):
    highs = df["high"]
    lows = df["low"]

    swing_highs = (highs.shift(lookback) < highs) & (highs.shift(-lookback) < highs)
    swing_lows = (lows.shift(lookback) > lows) & (lows.shift(-lookback) > lows)

    return swing_highs, swing_lows

def classify_structure(df: pd.DataFrame) -> Optional[str]:
    swing_highs, swing_lows = detect_swings(df)

    recent_highs = df.loc[swing_highs, "high"].tail(3)
    recent_lows = df.loc[swing_lows, "low"].tail(3)

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None

    if recent_highs.iloc[-1] > recent_highs.iloc[-2] and recent_lows.iloc[-1] > recent_lows.iloc[-2]:
        return "BULLISH"

    if recent_highs.iloc[-1] < recent_highs.iloc[-2] and recent_lows.iloc[-1] < recent_lows.iloc[-2]:
        return "BEARISH"

    return "RANGING"

def detect_liquidity_sweep(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 6:
        return None

    last = df.iloc[-1]
    prev_high = df["high"].iloc[-6:-1].max()
    prev_low = df["low"].iloc[-6:-1].min()

    if last.high > prev_high and last.close < prev_high:
        return "HIGH_SWEEP"

    if last.low < prev_low and last.close > prev_low:
        return "LOW_SWEEP"

    return None

def atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period + 1:
        return None

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )
    value = tr.rolling(period).mean().iloc[-1]
    if pd.isna(value):
        return None
    return value

# =======================
# MAIN SMC ENGINE
# =======================
@app.post("/quant/smc", dependencies=[Depends(verify_key)])
def smc_engine(req: QuantRequest):

    # ---- HARD DATA GUARD ----
    if len(req.candles) < 50:
        return {
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "state": "NO_TRADE",
            "direction": None,
            "confidence": 0.0,
            "structure": None,
            "liquidity_event": None,
            "htf_bias": None,
            "execution_validity": None,
            "engine_type": "SMC_STATE_ENGINE",
            "reason": "INSUFFICIENT_DATA"
        }

    df_1h = pd.DataFrame([c.dict() for c in req.candles])

    # ---- DERIVE HTF (4H) INTERNALLY ----
    df_4h = aggregate_to_4h(df_1h)
    htf_bias = classify_structure(df_4h) if df_4h is not None else None

    # ---- LTF ANALYSIS (1H) ----
    structure = classify_structure(df_1h)
    liquidity = detect_liquidity_sweep(df_1h)
    current_atr = atr(df_1h)

    if current_atr is None:
        return {
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "state": "NO_TRADE",
            "direction": None,
            "confidence": 0.0,
            "structure": structure,
            "liquidity_event": liquidity,
            "htf_bias": htf_bias,
            "execution_validity": None,
            "engine_type": "SMC_STATE_ENGINE",
            "reason": "ATR_UNAVAILABLE"
        }

    last = df_1h.iloc[-1]

    state = "NO_TRADE"
    direction = None
    confidence = 0.0

    # =======================
    # SETUP (CONTEXT FIRST)
    # =======================
    if structure == "BULLISH":
        state = "SETUP_LONG"
        direction = "LONG"
        confidence += 0.55
        if liquidity == "LOW_SWEEP":
            confidence += 0.15
        if htf_bias == "BULLISH":
            confidence += 0.1

    elif structure == "BEARISH":
        state = "SETUP_SHORT"
        direction = "SHORT"
        confidence += 0.55
        if liquidity == "HIGH_SWEEP":
            confidence += 0.15
        if htf_bias == "BEARISH":
            confidence += 0.1

    # =======================
    # EXECUTION (TRIGGER ONLY)
    # =======================
    displacement = abs(last.close - last.open) > current_atr

    if state == "SETUP_LONG":
        bos = last.close > df_1h["high"].iloc[-6:-1].max()
        if bos and displacement:
            state = "EXECUTE_LONG"
            confidence += 0.2

    if state == "SETUP_SHORT":
        bos = last.close < df_1h["low"].iloc[-6:-1].min()
        if bos and displacement:
            state = "EXECUTE_SHORT"
            confidence += 0.2

    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "state": state,
        "direction": direction,
        "confidence": round(min(confidence, 1.0), 2),
        "structure": structure,
        "liquidity_event": liquidity,
        "htf_bias": htf_bias,
        "execution_validity": "next_candle" if "EXECUTE" in state else None,
        "engine_type": "SMC_STATE_ENGINE"
    }
