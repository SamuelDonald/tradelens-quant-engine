from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import os
from datetime import datetime

app = FastAPI(title="TradeLens Quant Engine – SMC v2")

# =========================
# AUTH
# =========================

ENGINE_API_KEY = os.getenv("QUANT_ENGINE_API_KEY")

def verify_key(auth_header: str):
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = auth_header.split(" ")[1]
    if token != ENGINE_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# =========================
# DATA MODELS
# =========================

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

# =========================
# CORE ANALYSIS UTILITIES
# =========================

def calc_range(candles: List[Candle]):
    highs = [c.h for c in candles]
    lows = [c.l for c in candles]
    return max(highs), min(lows)

def detect_structure(candles: List[Candle]):
    closes = np.array([c.c for c in candles])
    slope = np.polyfit(range(len(closes)), closes, 1)[0]

    if slope > 0:
        return "bullish"
    elif slope < 0:
        return "bearish"
    return "ranging"

def detect_liquidity(candles: List[Candle], range_high, range_low):
    last = candles[-1]
    if last.h > range_high:
        return "buy-side", "swept"
    if last.l < range_low:
        return "sell-side", "swept"
    return "none", "building"

def price_location(price, range_high, range_low):
    mid = (range_high + range_low) / 2
    if price > mid:
        return "premium"
    elif price < mid:
        return "discount"
    return "equilibrium"

# =========================
# PLAY GENERATORS
# =========================

def generate_plays(state):
    plays = []

    # Liquidity sweep continuation
    if state["liquidity"]["status"] == "building":
        plays.append({
            "id": "LIQUIDITY_SWEEP_WAIT",
            "type": "LIQUIDITY_SWEEP",
            "direction": "long" if state["bias"]["direction"] == "bullish" else "short",
            "timeframe": state["timeframe"],
            "probability": 0.62 if state["bias"]["confidence"] > 0.5 else 0.45,
            "status": "forming",
            "conditions": {
                "bias_aligned": True,
                "liquidity_not_taken": True,
                "structure_clear": state["structure"]["ltf"] != "undefined"
            },
            "invalidation": {
                "level": state["range"]["low"] if state["bias"]["direction"] == "bullish" else state["range"]["high"],
                "reason": "range_break"
            },
            "next_confirmation": "liquidity sweep + displacement"
        })

    # Range fade
    if state["structure"]["ltf"] == "ranging":
        plays.append({
            "id": "RANGE_FADE",
            "type": "RANGE_FADE",
            "direction": "short" if state["price_location"] == "premium" else "long",
            "timeframe": state["timeframe"],
            "probability": 0.48,
            "status": "watch",
            "conditions": {
                "range_intact": True,
                "no_breakout": True
            },
            "invalidation": {
                "level": state["range"]["high"] if state["price_location"] == "premium" else state["range"]["low"],
                "reason": "range_expansion"
            },
            "next_confirmation": "rejection candle"
        })

    return plays

# =========================
# MAIN ENDPOINT
# =========================

@app.post("/quant/analyze")
def quant_analyze(payload: QuantRequest, authorization: str = Header(...)):
    verify_key(authorization)

    if len(payload.candles) < 50:
        raise HTTPException(status_code=400, detail="Insufficient candle data")

    candles = payload.candles

    range_high, range_low = calc_range(candles)
    structure = detect_structure(candles)
    liquidity_side, liquidity_status = detect_liquidity(candles, range_high, range_low)
    location = price_location(candles[-1].c, range_high, range_low)

    bias_direction = structure if structure in ["bullish", "bearish"] else "neutral"
    bias_confidence = 0.65 if structure != "ranging" else 0.35

    market_state = {
        "bias": {
            "direction": bias_direction,
            "confidence": round(bias_confidence, 2)
        },
        "phase": "consolidation" if structure == "ranging" else "expansion",
        "structure": {
            "htf": structure,
            "ltf": structure
        },
        "liquidity": {
            "status": liquidity_status,
            "side": liquidity_side
        },
        "price_location": location,
        "range": {
            "high": round(range_high, 5),
            "low": round(range_low, 5)
        }
    }

    plays = generate_plays({
        **market_state,
        "timeframe": payload.timeframe
    })

    execution_allowed = False
    active_play = None

    for play in plays:
        if play["status"] == "active" and play["probability"] >= 0.7:
            execution_allowed = True
            active_play = play["id"]

    return {
        "symbol": payload.symbol,
        "timeframe": payload.timeframe,
        "market_state": market_state,
        "plays": plays,
        "execution": {
            "allowed": execution_allowed,
            "active_play_id": active_play,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "AWAITING_CONFIRMATION" if not execution_allowed else "SETUP_ACTIVE"
        },
        "meta": {
            "engine_version": "smc-v2",
            "data_candles": len(candles),
            "generated_at": datetime.utcnow().isoformat()
        }
    }
