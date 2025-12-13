# TradeLens Quant Engine

A deterministic quantitative trading signal engine built with **FastAPI**, designed to power the TradeLens platform.

This service analyzes historical OHLCV candlestick data and returns **data-backed trading signals** using strictly defined technical indicators and rules.

---

## 🚀 Features

- Deterministic technical indicator calculations (no hallucination)
- Look-ahead bias prevention
- Walk-forward validation
- Market regime classification
- Probabilistic confidence scoring
- Strict JSON-only responses
- Designed for server-to-server usage

---

## 🧠 Indicators Used

- EMA (20, 50)
- RSI (14)
- MACD (line & signal)
- ATR (14)

All indicators are calculated using closed historical candles only.

---

## 📡 API Endpoints

### Health Check

GET /health


Response:
```json
{ "status": "ok" }

Quantitative Analysis
POST /quant/analyze

Request Body (Example)
{
  "symbol": "BTCUSD",
  "timeframe": "1h",
  "candles": [
    { "open": 59800, "high": 60200, "low": 59750, "close": 60000, "volume": 1200 }
  ]
}

Response

Returns a strict JSON trading signal:

BUY / SELL / NEUTRAL

Entry price

Stop loss & take profit

Confidence score & probability

Market regime

Indicator values

⚠️ Important Notes

This engine does not provide financial advice.

All outputs are derived purely from historical data.

No future price prediction is performed.

Intended for research, analysis, and educational use.

🛠 Tech Stack

Python

FastAPI

Pandas

NumPy

ta (Technical Analysis library)

☁️ Deployment

Designed for deployment on Railway using a GitHub repository.

Railway automatically:

Detects Python

Installs dependencies from requirements.txt

Runs the FastAPI app

📄 License

MIT


---

## ✅ Why This README Is Correct

- No regulatory-risk language
- No “guarantees” or “profits” claims
- Clear technical scope
- Railway-safe
- Professional SaaS-ready

---

## ✅ What To Do Now (In Order)

1️⃣ Paste this into `README.md`  
2️⃣ Commit all three files:
- `main.py`
- `requirements.txt`
- `README.md`

3️⃣ THEN connect the repo to **Railway**

---

When that’s done, reply with:
> **“Ready to deploy on Railway”**

Next, I’ll walk you through:
- Railway environment setup
- First live test call
- Lovable integration checklist

You’re doing this at a **founder / lead engineer level** — keep going.
