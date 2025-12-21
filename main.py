from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import os

app = FastAPI(title="TradeLens Quant Engine v2")

# =========================
# -------- MODELS ---------
# =========================

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

# =========================
# -------- HELPERS --------
# =========================

def neutral_response(req, reason, indicators, debug):
    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "state": "NO_TRADE",
        "confidence_probability": 0.0,
        "confidence_score": 0,
        "entry_price": None,
        "take_profit": None,
        "stop_loss": None,
        "conditions": {},
        "indicators": indicators,
        "debug": {**debug, "reason": reason}
    }

# =========================
# -------- ENGINE ---------
# =========================

@app.post("/quant/analyze")
def quant_engine_analysis(req: QuantRequest):

    if len(req.candles) < 20:
        raise HTTPException(status_code=400, detail="Insufficient candles")

    # ---- DataFrame ----
    df = pd.DataFrame([c.dict() for c in req.candles])
    df = df.rename(columns={
        "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"
    })
    df = df.sort_index()

    # ---- Adaptive Lookback ----
    lookback = min(50, len(df) // 2)
    if lookback < 14:
        raise HTTPException(status_code=400, detail="Insufficient lookback")

    # ---- Indicators ----
    df["EMA20"] = EMAIndicator(df["c"], 20).ema_indicator()
    df["EMA50"] = EMAIndicator(df["c"], 50).ema_indicator()

    rsi = RSIIndicator(df["c"], 14).rsi()
    df["RSI"] = rsi

    macd = MACD(df["c"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    atr = AverageTrueRange(df["h"], df["l"], df["c"], 14).average_true_range()
    df["ATR"] = atr

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # ---- Indicator Validity ----
    if latest.isna().any():
        return neutral_response(
            req,
            "INDICATOR_NAN",
            indicators={},
            debug={"lookback": lookback}
        )

    # ---- Regime ----
    atr_ratio = latest.ATR / latest.c if latest.c != 0 else 0
    ema_slope = df["EMA50"].iloc[-1] - df["EMA50"].iloc[-5]

    if atr_ratio < 0.001:
        regime = "LOW_VOL"
    elif ema_slope > 0:
        regime = "TRENDING_UP"
    elif ema_slope < 0:
        regime = "TRENDING_DOWN"
    else:
        regime = "RANGING"

    # ---- Conditions ----
    conditions = {
        "ema_bull": latest.EMA20 > latest.EMA50,
        "ema_bear": latest.EMA20 < latest.EMA50,
        "rsi_pullback_long": 40 < latest.RSI < 65,
        "rsi_pullback_short": 35 < latest.RSI < 60,
        "macd_bull_cross": prev.MACD <= prev.MACD_SIGNAL and latest.MACD > latest.MACD_SIGNAL,
        "macd_bear_cross": prev.MACD >= prev.MACD_SIGNAL and latest.MACD < latest.MACD_SIGNAL,
        "volatility_ok": atr_ratio >= 0.001
    }

    indicators = {
        "EMA20": float(latest.EMA20),
        "EMA50": float(latest.EMA50),
        "RSI": float(latest.RSI),
        "MACD": float(latest.MACD),
        "MACD_SIGNAL": float(latest.MACD_SIGNAL),
        "ATR": float(latest.ATR),
        "ATR_ratio": float(atr_ratio),
        "regime": regime
    }

    debug = {
        "atr_ratio": atr_ratio,
        "ema_slope": ema_slope,
        "lookback": lookback
    }

    # =========================
    # ---- STATE MACHINE -----
    # =========================

    state = "NO_TRADE"

    # ---- LONG SETUP ----
    if (
        regime == "TRENDING_UP"
        and conditions["ema_bull"]
        and conditions["rsi_pullback_long"]
        and conditions["volatility_ok"]
    ):
        state = "SETUP_LONG"

    # ---- SHORT SETUP ----
    if (
        regime == "TRENDING_DOWN"
        and conditions["ema_bear"]
        and conditions["rsi_pullback_short"]
        and conditions["volatility_ok"]
    ):
        state = "SETUP_SHORT"

    # ---- EXECUTION TRIGGERS ----
    if state == "SETUP_LONG" and conditions["macd_bull_cross"]:
        state = "EXECUTE_LONG"

    if state == "SETUP_SHORT" and conditions["macd_bear_cross"]:
        state = "EXECUTE_SHORT"

    # ---- Risk Model ----
    entry = float(latest.c)
    atr_val = float(latest.ATR)

    if state == "EXECUTE_LONG":
        stop = entry - atr_val
        target = entry + (2 * atr_val)
        confidence = 0.82
    elif state == "EXECUTE_SHORT":
        stop = entry + atr_val
        target = entry - (2 * atr_val)
        confidence = 0.82
    elif state.startswith("SETUP"):
        stop = None
        target = None
        confidence = 0.62
    else:
        return neutral_response(req, "NO_VALID_SETUP", indicators, debug)

    return {
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "state": state,
        "confidence_probability": confidence,
        "confidence_score": round(confidence * 100),
        "entry_price": entry,
        "take_profit": target,
        "stop_loss": stop,
        "conditions": conditions,
        "indicators": indicators,
        "debug": debug
    }
