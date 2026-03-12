from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException

from engine.data.collectors.mock_feed import MockMarketDataFeed
from engine.data.data_manager import DataManager
from engine.models.structure_model import StructureModel
from engine.models.momentum_model import MomentumModel
from engine.models.volatility_model import VolatilityModel
from engine.models.liquidity_model import LiquidityModel
from engine.signals.signal_generator import SignalGenerator
from engine.decision.trade_decision_engine import TradeDecisionEngine


app = FastAPI(title="TradeLens Quant Engine", version="0.1.0")

feed = MockMarketDataFeed(base_price=64000.0, base_volume=1200.0)
data_manager = DataManager(max_candles=500)

structure_model = StructureModel()
momentum_model = MomentumModel()
volatility_model = VolatilityModel()
liquidity_model = LiquidityModel()

signal_generator = SignalGenerator()
decision_engine = TradeDecisionEngine(min_aligned_models=3)


def _ensure_symbol_history(symbol: str, min_candles: int = 120) -> None:
    data_manager.ensure_history(symbol, feed=feed, min_candles=min_candles)


def _run_models(symbol: str) -> Dict:
    candles = data_manager.get_last_n_candles(symbol, n=200)
    if len(candles) < 30:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis")

    structure_output = structure_model.analyze(candles)
    momentum_output = momentum_model.analyze(candles)
    volatility_output = volatility_model.analyze(candles)
    liquidity_output = liquidity_model.analyze(candles)

    snapshot = signal_generator.build_analysis_snapshot(
        symbol=symbol,
        structure_output=structure_output,
        momentum_output=momentum_output,
        volatility_output=volatility_output,
        liquidity_output=liquidity_output,
    )
    return {"snapshot": snapshot, "candles": candles}


@app.get("/health")
def get_health() -> Dict:
    symbols = list(data_manager._candles.keys())
    return {
        "status": "ok",
        "tracked_symbols": symbols,
    }


@app.get("/analysis/{symbol}")
def get_analysis(symbol: str) -> Dict:
    _ensure_symbol_history(symbol)
    result = _run_models(symbol)
    return result["snapshot"]


@app.get("/signal/{symbol}")
def get_signal(symbol: str) -> Dict:
    _ensure_symbol_history(symbol)
    result = _run_models(symbol)
    snapshot = result["snapshot"]
    candles = result["candles"]

    signal = decision_engine.generate_signal(snapshot, candles)

    if signal["bias"] == "neutral" or signal["confidence"] <= 0.0:
        return {
            "symbol": symbol,
            "bias": "neutral",
            "confidence": 0.0,
            "entry_zone": [],
            "stop_loss": None,
            "targets": [],
            "reasoning": ["no sufficient model alignment"],
            "analysis_snapshot": snapshot,
        }

    return {
        **signal,
        "analysis_snapshot": snapshot,
    }

