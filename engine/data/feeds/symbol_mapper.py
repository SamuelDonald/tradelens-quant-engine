from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SymbolMapper:
    """
    Deterministic symbol mapping layer.

    Canonical symbols live in config/symbols.py. Provider-specific identifiers must be
    supplied explicitly via a mapping dict.
    """

    provider_symbols: Dict[str, Dict[str, str]]

    def to_provider(self, canonical_symbol: str, provider: str) -> str:
        mapping = self.provider_symbols.get(canonical_symbol, {})
        return mapping.get(provider, canonical_symbol)

    def canonicalize(self, canonical_symbol: str) -> str:
        # Placeholder for future canonicalization rules; today caller already uses canonical input.
        return canonical_symbol

    def provider_name(self, feed) -> Optional[str]:
        """
        Best-effort provider name identifier used by router for mapping keys.
        """
        name = feed.__class__.__name__.lower()
        if "polygon" in name:
            return "polygon"
        if "finnhub" in name:
            return "finnhub"
        if "alphavantage" in name or "alpha" in name:
            return "alphavantage"
        return None

