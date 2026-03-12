from __future__ import annotations

from typing import Dict


class SignalGenerator:
    """
    Aggregates model outputs into a single analysis snapshot.
    """

    def build_analysis_snapshot(
        self,
        symbol: str,
        structure_output: Dict,
        momentum_output: Dict,
        volatility_output: Dict,
        liquidity_output: Dict,
    ) -> Dict:
        return {
            "symbol": symbol,
            "structure": structure_output,
            "momentum": momentum_output,
            "volatility": volatility_output,
            "liquidity": liquidity_output,
        }

