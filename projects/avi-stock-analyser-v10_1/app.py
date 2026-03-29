"""
Stock Analyzer — FastAPI Backend
Run: python app.py
"""

import os, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from stock_universe import search_stocks, initialize_stocks, get_stock_count
from data_layer import get_live_quote, get_historical_data, get_fundamentals, get_cache_stats
from technical_analysis import run_full_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading stock universe...")
    initialize_stocks()
    logger.info(f"Ready: {get_stock_count()} stocks loaded")
    yield


app = FastAPI(title="Avi Stock Analyser", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/search")
async def api_search(q: str = Query("", min_length=1, max_length=50)):
    results = search_stocks(q)
    return {"query": q, "count": len(results), "total_stocks": get_stock_count(), "results": results}


@app.get("/api/quote")
async def api_quote(symbol: str, exchange: str = "NSE"):
    quote = get_live_quote(symbol, exchange)
    if not quote:
        return JSONResponse(status_code=404, content={"error": f"No quote for {symbol} on {exchange}"})
    return quote


@app.get("/api/analysis")
async def api_analysis(symbol: str, exchange: str = "NSE", period: str = "6mo", interval: str = "1d"):
    # Auto-correct period for intraday
    corrections = {"1m": "7d", "2m": "7d", "5m": "60d", "15m": "60d", "60m": "2y"}
    period = corrections.get(interval, period)

    df = get_historical_data(symbol, exchange, period, interval)
    if df is None or df.empty:
        return JSONResponse(status_code=404, content={"error": f"No data for {symbol} on {exchange}. Check if the symbol is correct."})

    analysis = run_full_analysis(df, interval=interval)
    analysis["symbol"] = symbol
    analysis["exchange"] = exchange
    analysis["interval"] = interval
    analysis["period"] = period
    return analysis


@app.get("/api/fundamentals")
async def api_fundamentals(symbol: str, exchange: str = "NSE"):
    data = get_fundamentals(symbol, exchange)
    if not data:
        return JSONResponse(status_code=404, content={"error": f"No fundamentals for {symbol}"})
    return data


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "stocks_loaded": get_stock_count(), "cache": get_cache_stats()}


@app.get("/api/news")
async def api_news(symbol: str = "RELIANCE", exchange: str = "NSE"):
    """Fetch latest news for a stock from Google News RSS."""
    import xml.etree.ElementTree as ET
    import urllib.parse
    import re
    from datetime import datetime

    # Build search query
    if exchange == "INDEX":
        query = f"{symbol} India stock market"
    else:
        query = f"{symbol} stock NSE India"

    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"

    news_items = []
    try:
        import requests
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:15]:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                source = item.findtext("source", "")
                # Clean up title (Google News appends " - Source" at end)
                clean_title = re.sub(r'\s*-\s*[^-]+$', '', title) if ' - ' in title else title
                news_items.append({
                    "title": clean_title,
                    "source": source or title.split(" - ")[-1] if " - " in title else "",
                    "link": link,
                    "published": pub_date,
                })
    except Exception as e:
        logger.warning(f"News fetch failed for {symbol}: {e}")

    # Fallback: try yfinance news
    if not news_items:
        try:
            from data_layer import get_fundamentals
            fund = get_fundamentals(symbol, exchange)
            if fund and fund.get("news"):
                for n in fund["news"][:10]:
                    if n.get("title"):
                        news_items.append(n)
        except Exception:
            pass

    return {"symbol": symbol, "news": news_items, "count": len(news_items)}


# Screener: Nifty 200 symbols for scanning
SCREENER_SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN",
    "SUNPHARMA","BAJFINANCE","WIPRO","HCLTECH","NTPC","TATAMOTORS","M&M",
    "ADANIENT","POWERGRID","BAJAJFINSV","TECHM","JSWSTEEL","TATASTEEL",
    "COALINDIA","ONGC","CIPLA","DRREDDY","HINDALCO","BPCL","HEROMOTOCO",
    "TATACONSUM","BRITANNIA","DIVISLAB","SBILIFE","HDFCLIFE","EICHERMOT",
    "APOLLOHOSP","LTIM","INDUSINDBK","ADANIPORTS","GRASIM","BAJAJ-AUTO",
    "HAL","BEL","IRCTC","TATAPOWER","ETERNAL","DLF","TRENT","DMART",
    "SUZLON","IREDA","RVNL","IRFC","MAZDOCK","CGPOWER","POLYCAB","DIXON",
    "PERSISTENT","VBL","JSWENERGY","JINDALSTEL","KEI","MAXHEALTH",
    # Nifty Next 50 + Midcap Select
    "VEDL","GODREJCP","AMBUJACEM","DABUR","PIDILITIND","SHREECEM","HAVELLS",
    "INDIGO","SIEMENS","ABB","MARICO","NAUKRI","TORNTPHARM","ICICIGI",
    "ICICIPRULI","CHOLAFIN","BANKBARODA","PNB","CANBK","GODREJPROP",
    "COFORGE","MPHASIS","LTTS","COLPAL","BERGEPAINT","SRF","FORTIS",
    "LUPIN","AUROPHARMA","BIOCON","MUTHOOTFIN","INDHOTEL","PAGEIND",
    "VOLTAS","CROMPTON","BHARATFORG","ASHOKLEY","CUMMINSIND","ACC","MRF",
    "MOTHERSON","BOSCHLTD","ADANIGREEN","ATGL","ADANIPOWER","NHPC","IOC",
    "HINDPETRO","GAIL","PETRONET","INDUSTOWER","POLICYBZR","NYKAA",
    "DELHIVERY","TATAELXSI","OFSS","BANDHANBNK","IDFCFIRSTB","FEDERALBNK",
    "RBLBANK","AUBANK","MANAPPURAM","LTF","SHRIRAMFIN","LICHSGFIN",
    "STARHEALTH","COCHINSHIP","ZYDUSLIFE","BALKRISIND","OBEROIRLTY",
    "PHOENIXLTD","PRESTIGE","SOLARINDS","KAYNES","KALYANKJIL","DEEPAKNTR",
    "NAVINFLUOR","LAURUSLABS","PIIND","TIINDIA","BAJAJHLDNG",
    # Additional popular stocks
    "TATACHEM","CONCOR","SAIL","NMDC","RECLTD","PFC","HUDCO","SJVN",
    "NHPC","BSOFT","MFSL","CANFINHOME","IPCALAB","NATCOPHARM","ALKEM",
    "GLENMARK","ABCAPITAL","SUNTV","TV18BRDCST","ZEEL","IDEA",
    "TATACOMM","TATAELXSI","MINDTREE","JUBLFOOD","DEVYANI","PATANJALI",
    "GRINDWELL","SCHAEFFLER","TIMKEN","SKFINDIA","SUPREMEIND","ASTRAL",
    "RELAXO","BATA","TITAN","WHIRLPOOL","BLUEDART","DALBHARAT",
    "JKCEMENT","RAMCOCEM","AARTIIND","ATUL","PIIND","CLEAN",
]
# Remove duplicates
SCREENER_SYMBOLS = list(dict.fromkeys(SCREENER_SYMBOLS))


@app.get("/api/screener")
async def api_screener(min_bullish: int = 8, exchange: str = "NSE"):
    """
    Scan ~200 major stocks. Returns those with min_bullish+ indicators bullish.
    Also checks if undervalued (PE<25 or PB<3).
    Pass min_bullish=6,7,8 etc to adjust threshold at runtime.
    """
    import time as _time

    results = []
    scanned = 0
    errors = 0

    for sym in SCREENER_SYMBOLS:
        try:
            df = get_historical_data(sym, exchange, "6mo", "1d")
            if df is None or df.empty or len(df) < 30:
                errors += 1
                continue

            analysis = run_full_analysis(df, interval="1d")
            signal = analysis.get("signal", {})
            bull_count = signal.get("bullish_count", 0)
            total = signal.get("total_indicators", 0)

            if bull_count < min_bullish:
                scanned += 1
                continue

            # Check if undervalued via fundamentals
            fund = get_fundamentals(sym, exchange)
            is_undervalued = False
            undervalue_reasons = []

            if fund:
                pe = fund.get("pe_ratio")
                pb = fund.get("pb_ratio")
                roe = fund.get("roe")
                debt_eq = fund.get("debt_to_equity")
                margins = fund.get("profit_margins")

                # Undervalued heuristics
                if pe is not None and pe < 25 and pe > 0:
                    is_undervalued = True
                    undervalue_reasons.append(f"PE {pe:.1f} (<25)")
                if pb is not None and pb < 3 and pb > 0:
                    is_undervalued = True
                    undervalue_reasons.append(f"PB {pb:.1f} (<3)")
                if roe is not None and roe > 0.15:
                    undervalue_reasons.append(f"ROE {roe*100:.1f}% (>15%)")
                if debt_eq is not None and debt_eq < 50:
                    undervalue_reasons.append(f"D/E {debt_eq:.0f} (<50)")
                if margins is not None and margins > 0.10:
                    undervalue_reasons.append(f"Margin {margins*100:.1f}%")

            current_price = analysis.get("current_price")
            projections = analysis.get("price_projections", {})

            results.append({
                "symbol": sym,
                "exchange": exchange,
                "current_price": current_price,
                "verdict": signal.get("verdict"),
                "bullish_count": bull_count,
                "bearish_count": signal.get("bearish_count", 0),
                "total_indicators": total,
                "score": signal.get("score", 0),
                "is_undervalued": is_undervalued,
                "undervalue_reasons": undervalue_reasons,
                "projection_1m": projections.get("1m", {}).get("target"),
                "projection_1m_chg": projections.get("1m", {}).get("change_pct"),
                "rsi": analysis.get("indicators", {}).get("rsi"),
            })

            scanned += 1
            _time.sleep(0.3)  # Rate limit respect

        except Exception as e:
            errors += 1
            logger.warning(f"Screener error for {sym}: {e}")
            continue

    # Sort: undervalued + highest bullish count first
    results.sort(key=lambda x: (-(1 if x["is_undervalued"] else 0), -x["bullish_count"], -x["score"]))

    return {
        "results": results,
        "scanned": scanned,
        "errors": errors,
        "total_matches": len(results),
        "min_bullish": min_bullish,
    }


@app.get("/manifest.json")
async def manifest():
    return {"name": "Avi Stock Analyser", "short_name": "AviStock", "start_url": "/", "display": "standalone", "background_color": "#ffffff", "theme_color": "#10b981"}


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "=" * 60)
    print("  AVI STOCK ANALYSER v10 — Indian Market Analysis")
    print("=" * 60)
    print(f"\n  Laptop:  http://localhost:8000")
    print(f"  Mobile:  http://{local_ip}:8000  (same WiFi)")
    print(f"  API:     http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
