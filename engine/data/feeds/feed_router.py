from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from engine.data.feeds.base_feed import BaseFeed, FeedError
from engine.data.feeds.polygon_feed import PolygonFeed
from engine.data.feeds.finnhub_feed import FinnhubFeed
from engine.data.feeds.alphavantage_feed import AlphaVantageFeed


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
    ):
        self.polygon = polygon or PolygonFeed()
        self.finnhub = finnhub or FinnhubFeed()
        self.alphavantage = alphavantage or AlphaVantageFeed()
        self.config = config or FeedRouterConfig()
        self.symbol_asset_overrides = symbol_asset_overrides or {}

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
        try:
            return primary.get_latest_candle(symbol, timeframe)
        except Exception as e:
            if not self.config.enable_fallback:
                raise
            # Fallback provider
            try:
                return self.alphavantage.get_latest_candle(symbol, timeframe)
            except Exception as e2:
                raise FeedError(f"router_failed:{asset_type}") from (e2 or e)

    def get_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        asset_type = self._asset_type(symbol)
        primary = self._primary_feed(asset_type)
        try:
            return primary.get_recent_candles(symbol, timeframe, limit)
        except Exception as e:
            if not self.config.enable_fallback:
                raise
            try:
                return self.alphavantage.get_recent_candles(symbol, timeframe, limit)
            except Exception as e2:
                raise FeedError(f"router_failed:{asset_type}") from (e2 or e)

