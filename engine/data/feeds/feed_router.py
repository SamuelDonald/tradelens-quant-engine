from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from engine.data.feeds.base_feed import BaseFeed, FeedError
from engine.data.feeds.polygon_feed import PolygonFeed
from engine.data.feeds.finnhub_feed import FinnhubFeed
from engine.data.feeds.alphavantage_feed import AlphaVantageFeed
from engine.data.feeds.symbol_mapper import SymbolMapper
from engine.observability import ENGINE_STATS


@dataclass(frozen=True)
class FeedRouterConfig:
    """
    Routing policy:
      - stocks -> polygon
      - crypto  -> polygon
      - forex   -> finnhub
      - fallback -> alphavantage
    """

    enable_fallback: bool = True


class FeedRouter:
    def __init__(
        self,
        polygon: Optional[BaseFeed] = None,
        finnhub: Optional[BaseFeed] = None,
        alphavantage: Optional[BaseFeed] = None,
        config: Optional[FeedRouterConfig] = None,
        symbol_asset_overrides: Optional[Dict[str, str]] = None,
        provider_symbols: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.polygon = polygon or PolygonFeed()
        self.finnhub = finnhub or FinnhubFeed()
        self.alphavantage = alphavantage or AlphaVantageFeed()
        self.config = config or FeedRouterConfig()
        self.symbol_asset_overrides = symbol_asset_overrides or {}
        self.symbol_mapper = SymbolMapper(provider_symbols=provider_symbols or {})

    def _asset_type(self, symbol: str) -> str:
        """
        Basic classifier. Prefer explicit config overrides for production.
        Returns: "stocks" | "crypto" | "forex"
        """
        if symbol in self.symbol_asset_overrides:
            return self.symbol_asset_overrides[symbol]

        s = symbol.upper()

        # Common crypto quote suffixes
        if s.endswith(("USDT", "USDC", "BTC", "ETH")) and len(s) >= 6:
            return "crypto"

        # FX and metals often represented as 6-letter pairs like EURUSD, XAUUSD
        if s.isalpha() and len(s) == 6 and s.endswith("USD"):
            return "forex"

        # Default: assume stock/index ticker
        return "stocks"

    def _primary_feed(self, asset_type: str) -> BaseFeed:
        if asset_type in ("stocks", "crypto"):
            return self.polygon
        if asset_type == "forex":
            return self.finnhub
        return self.alphavantage

    def get_latest_candle(self, symbol: str, timeframe: str) -> Dict:
        asset_type = self._asset_type(symbol)
        primary = self._primary_feed(asset_type)
        primary_provider = self.symbol_mapper.provider_name(primary) or "unknown"
        mapped_symbol = self.symbol_mapper.to_provider(symbol, primary_provider)
        try:
            ENGINE_STATS.record_request(primary_provider)
            candle = primary.get_latest_candle(mapped_symbol, timeframe)
            candle["symbol"] = symbol
            ENGINE_STATS.record_success(primary_provider)
            return candle
        except Exception as e:
            ENGINE_STATS.record_failure(primary_provider, repr(e))
            if not self.config.enable_fallback:
                raise
            # Fallback provider
            try:
                fb_provider = self.symbol_mapper.provider_name(self.alphavantage) or "alphavantage"
                fb_symbol = self.symbol_mapper.to_provider(symbol, fb_provider)
                ENGINE_STATS.record_fallback(primary_provider, fb_provider)
                ENGINE_STATS.record_request(fb_provider)
                candle = self.alphavantage.get_latest_candle(fb_symbol, timeframe)
                candle["symbol"] = symbol
                ENGINE_STATS.record_success(fb_provider)
                return candle
            except Exception as e2:
                ENGINE_STATS.record_failure("alphavantage", repr(e2))
                raise FeedError(f"router_failed:{asset_type}") from (e2 or e)

    def get_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        asset_type = self._asset_type(symbol)
        primary = self._primary_feed(asset_type)
        primary_provider = self.symbol_mapper.provider_name(primary) or "unknown"
        mapped_symbol = self.symbol_mapper.to_provider(symbol, primary_provider)
        try:
            ENGINE_STATS.record_request(primary_provider)
            candles = primary.get_recent_candles(mapped_symbol, timeframe, limit)
            for c in candles:
                c["symbol"] = symbol
            ENGINE_STATS.record_success(primary_provider)
            return candles
        except Exception as e:
            ENGINE_STATS.record_failure(primary_provider, repr(e))
            if not self.config.enable_fallback:
                raise
            try:
                fb_provider = self.symbol_mapper.provider_name(self.alphavantage) or "alphavantage"
                fb_symbol = self.symbol_mapper.to_provider(symbol, fb_provider)
                ENGINE_STATS.record_fallback(primary_provider, fb_provider)
                ENGINE_STATS.record_request(fb_provider)
                candles = self.alphavantage.get_recent_candles(fb_symbol, timeframe, limit)
                for c in candles:
                    c["symbol"] = symbol
                ENGINE_STATS.record_success(fb_provider)
                return candles
            except Exception as e2:
                ENGINE_STATS.record_failure("alphavantage", repr(e2))
                raise FeedError(f"router_failed:{asset_type}") from (e2 or e)

