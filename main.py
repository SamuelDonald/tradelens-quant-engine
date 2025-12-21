from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os

app = FastAPI()

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

ENGINE_API_KEY = os.getenv("ENGINE_API_KEY")

def verify_key(key: str = Depends(api_key_header)):
    if key != ENGINE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------- Data Models --------

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

class QuantRequest(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Candle]

# -------- Helper Functions --------

def detect_swings(df, lookback=2):
    highs = df["high"]
    lows = df["low"]

    swing_highs = (highs.shift(lookback) < highs) & (highs.shift(-lookback) < highs)
    swing_lows = (lows.shift(lookback) > lows) & (lows.shift(-lookback) > lows)

    return swing_highs, swing_lows

def classify_structure(df):
    swing_highs, swing_lows = detect_swings(df)

    recent_highs = df.loc[swing_highs, "high"].tail(3)
    recent_lows = df.loc[swing_lows, "low"].tail(3)

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return None

    hh = recent_highs.iloc[-1] > recent_highs.iloc[-2]
    hl = recent_lows.iloc[-1] > recent_lows.iloc[-2]
    lh = recent_highs.iloc[-1] < recent_highs.iloc[-2]
    ll = recent_lows.iloc[-1] < recent_lows.iloc[-2]

    if hh and hl:
        return "BULLISH"
    if lh and ll:
        return "BEARISH"

    return "RANGING"

def detect_liquidity_sweep(df):
    last = df.iloc[-1]
    prev_high = df["high"].iloc[-6:-1].max()
    prev_low = df["low"].iloc[-6:-1].min()

    if last.high > prev_high and last.close < prev_high:
        return "HIGH_SWEEP"
    if last.low < prev_low and last.close > prev_low:
        return "LOW_SWEEP"

    return None

def atr(df, period=14):
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )
    return tr.rolling(period).mean().iloc[-1]

# -------- Main Engine --------

@app.post("/quant/smc", dependencies=[Depends(verify_key)])
def smc_engine(req: QuantRequest):
    if len(req.candles) < 30:
        return {"state": "NO_TRADE", "reason": "INSUFFICIENT_DATA"}

    df = pd.DataFrame([c.dict() for c in req.candles])

    structure = classify_structure(df)
    liquidity = detect_liquidity_sweep(df)
    current_atr = atr(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Default response
    state = "NO_TRADE"
    direction = None
    confidence = 0.0

    # -------- SETUP LOGIC --------
    if structure == "BULLISH":
        state = "SETUP_LONG"
        direction = "LONG"
        confidence += 0.6
        if liquidity == "LOW_SWEEP":
            confidence += 0.15

    elif structure == "BEARISH":
        state = "SETUP_SHORT"
        direction = "SHORT"
        confidence += 0.6
        if liquidity == "HIGH_SWEEP":
            confidence += 0.15

    # -------- EXECUTION LOGIC --------
    displacement = abs(last.close - last.open) > current_atr

    if state == "SETUP_LONG":
        bos = last.close > df["high"].iloc[-5:-1].max()
        if bos and displacement:
            state = "EXECUTE_LONG"
            confidence += 0.25

    if state == "SETUP_SHORT":
        bos = last.close < df["low"].iloc[-5:-1].min()
        if bos and displacement:
            state = "EXECUTE_SHORT"
            confidence += 0.25

    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "state": state,
        "direction": direction,
        "confidence": round(min(confidence, 1.0), 2),
        "structure": structure,
        "liquidity_event": liquidity,
        "execution_validity": "next_candle",
        "engine_type": "SMC_STATE_ENGINE"
    }
