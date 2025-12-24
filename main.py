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
# UTILS
# =======================
def aggregate_to_4h(df: pd.DataFrame, factor: int = 4):
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

def swings(df, lb=2):
    sh = (df.high.shift(lb) < df.high) & (df.high.shift(-lb) < df.high)
    sl = (df.low.shift(lb) > df.low) & (df.low.shift(-lb) > df.low)
    return sh, sl

def structure_state(df):
    sh, sl = swings(df)
    highs = df.loc[sh, "high"].tail(3)
    lows = df.loc[sl, "low"].tail(3)

    if len(highs) < 2 or len(lows) < 2:
        return "UNDEFINED"

    if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
        return "HH-HL"

    if highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
        return "LH-LL"

    return "BROKEN"

def liquidity_event(df):
    last = df.iloc[-1]
    prev_high = df.high.iloc[-10:-1].max()
    prev_low = df.low.iloc[-10:-1].min()

    if last.high > prev_high and last.close < prev_high:
        return "BUY_SIDE_TAKEN"

    if last.low < prev_low and last.close > prev_low:
        return "SELL_SIDE_TAKEN"

    return "NONE"

# =======================
# MAIN ENGINE
# =======================
@app.post("/quant/smc", dependencies=[Depends(verify_key)])
def smc_engine(req: QuantRequest):

    if len(req.candles) < 60:
        raise HTTPException(400, "Insufficient candle data")

    df = pd.DataFrame([c.dict() for c in req.candles])
    df_4h = aggregate_to_4h(df)

    ltf_structure = structure_state(df)
    htf_structure = structure_state(df_4h) if len(df_4h) >= 10 else "UNDEFINED"
    liquidity = liquidity_event(df)

    high = df.high.max()
    low = df.low.min()
    mid = (high + low) / 2
    last_price = df.close.iloc[-1]

    price_location = (
        "DISCOUNT" if last_price < mid else
        "PREMIUM" if last_price > mid else
        "EQUILIBRIUM"
    )

    bias = "NEUTRAL"
    bias_conf = 0.5
    bias_reason = []

    if htf_structure == "HH-HL":
        bias = "LONG"
        bias_conf += 0.15
        bias_reason.append("HTF bullish structure")

    if htf_structure == "LH-LL":
        bias = "SHORT"
        bias_conf += 0.15
        bias_reason.append("HTF bearish structure")

    if liquidity == "SELL_SIDE_TAKEN" and bias == "LONG":
        bias_conf += 0.1
        bias_reason.append("Sell-side liquidity swept")

    if liquidity == "BUY_SIDE_TAKEN" and bias == "SHORT":
        bias_conf += 0.1
        bias_reason.append("Buy-side liquidity swept")

    plays = []

    if bias == "LONG" and price_location == "DISCOUNT":
        plays.append({
            "play_type": "CONTINUATION",
            "direction": "LONG",
            "setup_condition": "Pullback into discount with bullish HTF bias",
            "invalidation": "Break below last higher low",
            "execution_ready": False,
            "probability": round(bias_conf, 2)
        })

    if bias == "SHORT" and price_location == "PREMIUM":
        plays.append({
            "play_type": "CONTINUATION",
            "direction": "SHORT",
            "setup_condition": "Pullback into premium with bearish HTF bias",
            "invalidation": "Break above last lower high",
            "execution_ready": False,
            "probability": round(bias_conf, 2)
        })

    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "market_state": {
            "trend": bias,
            "structure": ltf_structure,
            "liquidity_state": liquidity
        },
        "price_location": {
            "relative_position": price_location
        },
        "bias": {
            "direction": bias,
            "confidence": round(min(bias_conf, 1.0), 2),
            "reason": bias_reason
        },
        "plays": plays,
        "execution": {
            "allowed": False,
            "reason": "Awaiting confirmation"
        },
        "data_quality": {
            "candles_used": len(df),
            "confidence": "HIGH"
        }
    }
