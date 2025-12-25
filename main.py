import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

# ======================
# 🔐 SECURITY
# ======================

ENGINE_API_KEY = os.getenv("QUANT_ENGINE_API_KEY")
if not ENGINE_API_KEY:
    raise RuntimeError("QUANT_ENGINE_API_KEY is not set")

# ======================
# 🚀 APP INIT
# ======================

app = FastAPI(title="TradeLens SMC Quant Engine")

# ======================
# 📦 MODELS
# ======================

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

# ======================
# 🔑 AUTH
# ======================

def verify_key(authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    if token != ENGINE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ======================
# 🔧 HELPERS
# ======================

def candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    df = pd.DataFrame([c.dict() for c in candles])
    return df.reset_index(drop=True)

def aggregate_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group"] = df.index // 4
    agg = df.groupby("group").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })
    return agg.dropna()

def detect_structure(df: pd.DataFrame) -> str:
    if len(df) < 5:
        return "UNDEFINED"

    highs = df["high"].rolling(3).max()
    lows = df["low"].rolling(3).min()

    if highs.iloc[-1] > highs.iloc[-3] and lows.iloc[-1] > lows.iloc[-3]:
        return "HH-HL"

    if highs.iloc[-1] < highs.iloc[-3] and lows.iloc[-1] < lows.iloc[-3]:
        return "LH-LL"

    return "RANGING"

def detect_liquidity(df: pd.DataFrame) -> str:
    if len(df) < 10:
        return "NONE"

    recent_high = df["high"].iloc[-2]
    recent_low = df["low"].iloc[-2]

    if df["high"].iloc[-1] > recent_high:
        return "BUY_SIDE_TAKEN"

    if df["low"].iloc[-1] < recent_low:
        return "SELL_SIDE_TAKEN"

    return "NONE"

def price_location(df: pd.DataFrame) -> dict:
    high = df["high"].max()
    low = df["low"].min()
    eq = (high + low) / 2
    price = df["close"].iloc[-1]

    if price > eq:
        pos = "PREMIUM"
    elif price < eq:
        pos = "DISCOUNT"
    else:
        pos = "EQUILIBRIUM"

    return {
        "relative_position": pos,
        "range_high": float(high),
        "range_low": float(low)
    }

# ======================
# 🧠 CORE ENGINE
# ======================

@app.post("/quant/smc")
def smc_engine(payload: QuantRequest, authorization: str = Header(...)):
    verify_key(authorization)

    df = candles_to_df(payload.candles)

    if len(df) < 30:
        raise HTTPException(status_code=400, detail="Insufficient candles")

    # HTF Context
    df_4h = aggregate_to_4h(df)
    htf_structure = detect_structure(df_4h)
    liquidity = detect_liquidity(df_4h)

    # Bias
    if htf_structure == "HH-HL":
        bias_dir = "LONG"
    elif htf_structure == "LH-LL":
        bias_dir = "SHORT"
    else:
        bias_dir = "NEUTRAL"

    # LTF
    ltf_structure = detect_structure(df)
    location = price_location(df)

    # Plays (non-binary, descriptive)
    plays = []

    if bias_dir == "LONG" and location["relative_position"] == "DISCOUNT":
        plays.append({
            "play_type": "CONTINUATION",
            "direction": "LONG",
            "setup_condition": "Bullish HTF structure with discount price location",
            "invalidation": "Break of last higher low",
            "execution_ready": False,
            "probability": 0.65
        })

    if bias_dir == "SHORT" and location["relative_position"] == "PREMIUM":
        plays.append({
            "play_type": "CONTINUATION",
            "direction": "SHORT",
            "setup_condition": "Bearish HTF structure with premium price location",
            "invalidation": "Break of last lower high",
            "execution_ready": False,
            "probability": 0.65
        })

    # Execution gate (never fabricated)
    execution_allowed = False
    execution_reason = "Await LTF confirmation"

    response = {
        "symbol": payload.symbol,
        "timeframe": payload.timeframe,

        "market_state": {
            "trend": "BULLISH" if bias_dir == "LONG" else "BEARISH" if bias_dir == "SHORT" else "RANGING",
            "structure": htf_structure,
            "phase": "PULLBACK",
            "liquidity_state": liquidity
        },

        "price_location": location,

        "bias": {
            "direction": bias_dir,
            "confidence": 0.7 if bias_dir != "NEUTRAL" else 0.4,
            "reason": [
                f"HTF structure: {htf_structure}",
                f"Liquidity: {liquidity}",
                f"Price location: {location['relative_position']}"
            ]
        },

        "plays": plays,

        "execution": {
            "allowed": execution_allowed,
            "reason": execution_reason
        },

        "data_quality": {
            "candles_used": len(df),
            "confidence": "HIGH" if len(df) > 80 else "MEDIUM"
        }
    }

    return response
