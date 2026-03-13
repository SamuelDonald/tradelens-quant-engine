from __future__ import annotations

import asyncio
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from engine.data.data_manager import DataManager
from engine.models.structure_model import StructureModel
from engine.models.momentum_model import MomentumModel
from engine.models.volatility_model import VolatilityModel
from engine.models.liquidity_model import LiquidityModel
from engine.signals.signal_generator import SignalGenerator
from engine.decision.trade_decision_engine import TradeDecisionEngine
from engine.data.feeds.feed_router import FeedRouter
from engine.data.feeds.historical_loader import HistoricalLoader
from engine.data.feeds.scheduler import MarketDataScheduler
from config.symbols import SYMBOLS, TIMEFRAMES, ASSET_TYPES


app = FastAPI(title="TradeLens Quant Engine", version="0.1.0")

load_dotenv()

data_managers: Dict[str, DataManager] = {tf: DataManager(max_candles=500) for tf in TIMEFRAMES}

structure_model = StructureModel()
momentum_model = MomentumModel()
volatility_model = VolatilityModel()
liquidity_model = LiquidityModel()

signal_generator = SignalGenerator()
decision_engine = TradeDecisionEngine(min_aligned_models=3)

router: Optional[FeedRouter] = None
scheduler: Optional[MarketDataScheduler] = None

def _run_models(symbol: str, timeframe: str) -> Dict:
    if timeframe not in data_managers:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")

    candles = data_managers[timeframe].get_last_n_candles(symbol, n=200)
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
    symbols = sorted({s for dm in data_managers.values() for s in dm._candles.keys()})
    return {
        "status": "ok",
        "tracked_symbols": symbols,
        "timeframes": list(data_managers.keys()),
    }


@app.get("/analysis/{symbol}")
def get_analysis(symbol: str, timeframe: str = "1m") -> Dict:
    result = _run_models(symbol, timeframe=timeframe)
    return result["snapshot"]


@app.get("/signal/{symbol}")
def get_signal(symbol: str, timeframe: str = "1m") -> Dict:
    result = _run_models(symbol, timeframe=timeframe)
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


@app.on_event("startup")
async def _startup() -> None:
    global router, scheduler

    router = FeedRouter(symbol_asset_overrides=ASSET_TYPES)
    loader = HistoricalLoader(router=router)
    await loader.load(data_managers=data_managers, symbols=SYMBOLS, timeframes=TIMEFRAMES, min_candles=200)

    scheduler = MarketDataScheduler(router=router, data_managers=data_managers, symbols=SYMBOLS, timeframes=TIMEFRAMES)
    asyncio.create_task(scheduler.run())

