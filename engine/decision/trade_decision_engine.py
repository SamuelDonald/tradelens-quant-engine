from __future__ import annotations

from typing import Dict, List, Any


class TradeDecisionEngine:
    """
    Evaluates analysis snapshot and produces structured, explainable signals.
    All logic is rule-based and deterministic.
    """

    def __init__(self, min_aligned_models: int = 3):
        self.min_aligned_models = min_aligned_models

    def _compute_bullish_alignment(self, snapshot: Dict) -> Dict[str, bool]:
        s = snapshot["structure"]
        m = snapshot["momentum"]
        v = snapshot["volatility"]
        l = snapshot["liquidity"]

        conds = {
            "bullish_structure": s.get("trend") == "bullish",
            "positive_momentum": m.get("momentum_direction") == "up",
            "adequate_volatility": v.get("volatility_regime") != "low",
            "liquidity_sweep": l.get("recent_liquidity_sweep") is True,
        }
        return conds

    def _compute_bearish_alignment(self, snapshot: Dict) -> Dict[str, bool]:
        s = snapshot["structure"]
        m = snapshot["momentum"]
        v = snapshot["volatility"]
        l = snapshot["liquidity"]

        conds = {
            "bearish_structure": s.get("trend") == "bearish",
            "negative_momentum": m.get("momentum_direction") == "down",
            "adequate_volatility": v.get("volatility_regime") != "low",
            "liquidity_sweep": l.get("recent_liquidity_sweep") is True,
        }
        return conds

    def _confidence_from_alignment(self, conds: Dict[str, bool], total_models: int = 4) -> float:
        aligned_models = sum(1 for ok in conds.values() if ok)
        return aligned_models / float(total_models)

    def _build_reasoning(self, conds: Dict[str, bool], bias: str) -> List[str]:
        reasons: List[str] = []
        for name, ok in conds.items():
            if not ok:
                continue
            if name == "bullish_structure":
                reasons.append("bullish structure")
            elif name == "bearish_structure":
                reasons.append("bearish structure")
            elif name == "positive_momentum":
                reasons.append("positive momentum")
            elif name == "negative_momentum":
                reasons.append("negative momentum")
            elif name == "adequate_volatility":
                reasons.append("volatility not suppressed")
            elif name == "liquidity_sweep":
                reasons.append("liquidity sweep detected")

        if not reasons and bias == "neutral":
            reasons.append("insufficient model alignment")
        return reasons

    def _derive_levels(self, candles: List[Dict[str, Any]], bias: str) -> Dict[str, Any]:
        last = candles[-1]
        low = last["low"]
        high = last["high"]
        close = last["close"]

        if bias == "bullish":
            entry_zone = [low, high]
            stop_loss = low * 0.99
            targets = [close * 1.025, close * 1.05]
        elif bias == "bearish":
            entry_zone = [low, high]
            stop_loss = high * 1.01
            targets = [close * 0.975, close * 0.95]
        else:
            entry_zone = []
            stop_loss = None
            targets = []

        return {
            "entry_zone": [float(entry_zone[0]), float(entry_zone[1])] if entry_zone else [],
            "stop_loss": float(stop_loss) if stop_loss is not None else None,
            "targets": [float(t) for t in targets],
        }

    def generate_signal(self, snapshot: Dict, candles: List[Dict]) -> Dict:
        """
        Returns a structured signal. If alignment is insufficient, bias is neutral and
        no trade levels are suggested.
        """
        bullish_conds = self._compute_bullish_alignment(snapshot)
        bearish_conds = self._compute_bearish_alignment(snapshot)

        bullish_conf = self._confidence_from_alignment(bullish_conds)
        bearish_conf = self._confidence_from_alignment(bearish_conds)

        if bullish_conf >= bearish_conf and bullish_conf >= (self.min_aligned_models / 4.0):
            bias = "bullish"
            conds = bullish_conds
            confidence = bullish_conf
        elif bearish_conf > bullish_conf and bearish_conf >= (self.min_aligned_models / 4.0):
            bias = "bearish"
            conds = bearish_conds
            confidence = bearish_conf
        else:
            bias = "neutral"
            conds = {}
            confidence = 0.0

        levels = self._derive_levels(candles, bias)
        reasoning = self._build_reasoning(conds, bias)

        return {
            "symbol": snapshot["symbol"],
            "bias": bias,
            "confidence": round(float(confidence), 4),
            "entry_zone": levels["entry_zone"],
            "stop_loss": levels["stop_loss"],
            "targets": levels["targets"],
            "reasoning": reasoning,
        }

