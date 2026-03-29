<p align="center">
  <h1 align="center">📈 Avi Stock Analyser</h1>
  <p align="center">
    <strong>AI-powered Indian stock market analyzer with real-time technical analysis, chart pattern detection, and price projections</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Plotly.js-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
    <img src="https://img.shields.io/badge/NSE-India-FF9933?style=flat-square" alt="NSE">
    <img src="https://img.shields.io/badge/PWA-Installable-5A0FC8?style=flat-square&logo=pwa&logoColor=white" alt="PWA">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  </p>
</p>

---

A full-stack stock analysis platform for **NSE, BSE, and Indian indices** (Nifty 50, Bank Nifty, Sensex). Combines **18 technical indicators**, **18 chart pattern detectors**, **price projections**, **fundamentals**, and **real-time news** — all in a clean white-themed PWA that runs locally.

> ⚠️ **Disclaimer:** Educational tool only. Not financial advice. Past performance ≠ future results.

---

## ✨ Features

### 📊 Technical Analysis — 18 Indicators
RSI · MACD · Bollinger Bands · SMA (20/50/200) · EMA (9/21/50) · Stochastic · ATR · ADX · Supertrend · VWAP · OBV · Fibonacci · Ichimoku Cloud · Williams %R · Chaikin Money Flow · Pivot Points · Support/Resistance · Elliott Wave

All indicators feed into a **Signal Verdict Engine** that outputs a single BULLISH/BEARISH/NEUTRAL call with score and reasoning.

### 🔍 Chart Pattern Detection — 18 Patterns

| Bullish | Bearish | Neutral |
|---------|---------|---------|
| Double Bottom | Double Top | Symmetrical Triangle |
| Inverse Head & Shoulders | Head & Shoulders | Rectangle |
| Ascending Triangle | Descending Triangle | Broadening Formation |
| Cup & Handle | Rising Wedge | |
| Falling Wedge | Bear Flag | |
| Bull Flag | Triple Top | |
| Triple Bottom | | |
| Rounding Bottom | | |

Each pattern includes: confidence score, trendline overlay on chart, hover tooltip with outcome and target price, detection timestamp, and timeframe context.

### 🎯 Price Projections (1M / 3M / 6M)
Not hallucinated — calculated from actual indicators: historical returns, RSI mean-reversion, MACD momentum, MA alignment, Supertrend, Bollinger position, Support/Resistance boundaries, and Fibonacci targets.

### 📰 Real-Time News
Google News RSS integration — no API key needed. Latest 15 headlines with source, time-ago formatting, and direct links. Works for stocks and indices.

### 🧭 Smart Timeframe Recommendations
Select your target (Intraday → 1 Year+) and the app auto-switches to the optimal chart timeframe with reasoning.

### 📋 Fundamentals (Multi-Source)
Key ratios, income statement, balance sheet, cash flow, shareholding, and news — pulled from yfinance `.info`, `.fast_info`, and raw financial statements with automatic fallback.

### 🇮🇳 Indian Market Coverage
- **NSE** — All listed equities (fetched from NSE's official CSV)
- **BSE** — Full support via yfinance
- **Indices** — Nifty 50, Bank Nifty, Sensex, Nifty IT, Nifty Midcap 50
- **Live Data** — nsetools (real-time) → yfinance (fallback)
- **IST Timestamps** — All times in Indian Standard Time

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, FastAPI, uvicorn |
| Data | nsetools, yfinance |
| Analysis | NumPy, Pandas (custom TA engine) |
| Frontend | Vanilla HTML/CSS/JS, Plotly.js |
| News | Google News RSS |
| PWA | Service Worker for offline support |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/avi-stock-analyser.git
cd avi-stock-analyser

# Install
pip install -r requirements.txt

# Run
python app.py
```

Open **http://localhost:8000** in your browser.

### 📱 Mobile Access (Same WiFi)
1. Find your laptop's IP: `ipconfig` (Windows) / `ifconfig` (Mac/Linux)
2. Open `http://<laptop-ip>:8000` on your phone
3. "Add to Home Screen" for native app experience

---

## 📁 Project Structure

```
avi-stock-analyser/
├── app.py                  # FastAPI server + API endpoints
├── technical_analysis.py   # 18 indicators + signal engine + projections
├── pattern_detection.py    # 18 chart pattern detectors
├── data_layer.py           # Data fetching (nsetools → yfinance fallback)
├── stock_universe.py       # NSE/BSE stock list + search
├── requirements.txt
├── static/
│   ├── index.html          # Complete frontend (single-file PWA)
│   └── sw.js               # Service worker
└── README.md
```

---

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/search?q=reliance` | Search stocks by name or symbol |
| `GET /api/quote?symbol=RELIANCE&exchange=NSE` | Live price quote |
| `GET /api/analysis?symbol=RELIANCE&exchange=NSE&period=6mo&interval=1d` | Full technical analysis |
| `GET /api/fundamentals?symbol=RELIANCE&exchange=NSE` | Company fundamentals |
| `GET /api/news?symbol=RELIANCE&exchange=NSE` | Latest news headlines |
| `GET /api/screener?min_bullish=8` | Scan stocks for bullish setups |
| `GET /api/health` | Server health check |

---

## 🧠 How It Works

### Signal Verdict
Each indicator generates a bullish/bearish/neutral signal. The verdict engine calculates:

```
Score = ((bullish − bearish) / total) × 100

STRONG BULLISH    → score > 60
BULLISH           → score > 30
SLIGHTLY BULLISH  → score > 10
NEUTRAL           → -10 to 10
SLIGHTLY BEARISH  → score < -10
BEARISH           → score < -30
STRONG BEARISH    → score < -60
```

### Pattern Detection
Swing-point analysis at multiple lookback periods (3, 5, 7 bars). Detects up to 5 patterns simultaneously with:
- Confidence scoring (0–100%)
- Trendline overlay on chart
- One-liner outcome with target price
- Adjustable confidence threshold

### Data Caching
| Data Type | Cache Duration |
|-----------|---------------|
| Live quotes | 2 minutes |
| Historical (intraday) | 15 minutes |
| Historical (daily) | 24 hours |
| Fundamentals | 24 hours |

---

## ⏱️ Supported Timeframes

| Timeframe | Period | Best For |
|-----------|--------|----------|
| 1 min | 7 days | Scalping |
| 5 min | 60 days | Intraday |
| 15 min | 60 days | Intraday / Swing entry |
| 1 hour | 2 years | Swing trading |
| 1 day | 5 years | Position trading |
| 1 week | 10 years | Long-term investing |

---

## 📝 Roadmap

- [x] 18 technical indicators with signal engine
- [x] 18 chart pattern detectors with visual overlay
- [x] Price projections (1M / 3M / 6M)
- [x] White theme with clean aesthetics
- [x] Nifty 50 / Bank Nifty / Sensex support
- [x] News integration (Google News RSS)
- [x] Timeframe recommendation engine
- [x] Fundamentals with multi-source fallback
- [ ] Historical pattern matching (past similar patterns → prediction)
- [ ] All-stock screener with bulk scanning
- [ ] Multi-stock comparison view
- [ ] Alert system (price / indicator thresholds)
- [ ] Export analysis to PDF

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance data
- [nsetools](https://github.com/vsjha18/nsetools) — NSE real-time data
- [Plotly.js](https://plotly.com/javascript/) — Interactive charting
- [FastAPI](https://fastapi.tiangolo.com/) — Backend framework

---

<p align="center">
  Built with ❤️ for the Indian stock market community
</p>
