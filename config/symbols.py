SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "EURUSD",
    "XAUUSD",
    "AAPL",
    "TSLA",
    "SPY",
]

TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "1h",
    "4h",
]

# Optional explicit asset type overrides for routing:
# "stocks" | "crypto" | "forex"
ASSET_TYPES = {
    "BTCUSDT": "crypto",
    "ETHUSDT": "crypto",
    "EURUSD": "forex",
    "XAUUSD": "forex",
    "AAPL": "stocks",
    "TSLA": "stocks",
    "SPY": "stocks",
}

# Provider-specific symbol mappings.
# If a mapping is missing for a given provider, the canonical symbol is passed through as-is.
#
# IMPORTANT: Do not guess mappings in production—set them explicitly per provider.
PROVIDER_SYMBOLS = {
    # Polygon tickers (stocks are usually pass-through; crypto often needs explicit mapping).
    # Examples (adjust to your Polygon account/product symbols):
    # "BTCUSDT": {"polygon": "X:BTCUSD"},
    # "ETHUSDT": {"polygon": "X:ETHUSD"},

    # Finnhub forex symbols often require a provider-specific prefix (example):
    # "EURUSD": {"finnhub": "OANDA:EUR_USD"},
    # "XAUUSD": {"finnhub": "OANDA:XAU_USD"},

    # AlphaVantage symbols generally accept "EURUSD" for FX_INTRADAY and stock tickers as-is.
    # "EURUSD": {"alphavantage": "EURUSD"},
}


