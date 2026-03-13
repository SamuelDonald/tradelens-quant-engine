# tradelens-quant-engine

The TradeLens Quant Engine, built to curate actionable trading insights inside the tradelens ecosystem.

## Running the quant API

### Environment variables

Create a repo-root `.env` (do not commit it) with:

```env
POLYGON_API_KEY=...
FINNHUB_API_KEY=...
ALPHAVANTAGE_API_KEY=...
```

1. **Install dependencies** (Python 3.11 recommended):

```bash
pip install -r requirements.txt
```

2. **Start the FastAPI server** from the project root:

```bash
uvicorn api.main:app --reload
```

3. **Hit the endpoints** in a browser or via `curl`:

- Health:

  - `GET http://127.0.0.1:8000/health`

- Market analysis for a symbol (e.g. BTCUSDT):

  - `GET http://127.0.0.1:8000/analysis/BTCUSDT?timeframe=15m`

- Signal for a symbol (model-alignment based, deterministic):

  - `GET http://127.0.0.1:8000/signal/BTCUSDT?timeframe=15m`

The `/analysis/{symbol}` endpoint returns the raw model outputs (structure, momentum, volatility, liquidity).

The `/signal/{symbol}` endpoint returns a structured signal only when there is sufficient alignment between models; otherwise it returns a neutral bias with explicit reasoning and always includes an `analysis_snapshot` so every signal is fully explainable.

### Notes

- **Multi-timeframe**: pass `timeframe` as a query parameter. Supported values are defined in `config/symbols.py` under `TIMEFRAMES` (default is `1m`).
- **Live data ingestion**: on startup the engine loads historical candles (200 per symbol/timeframe), then starts an async scheduler that polls once per minute and updates the in-memory rolling windows.

