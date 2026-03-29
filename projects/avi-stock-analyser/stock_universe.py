"""
Stock Universe — ALL NSE Stocks from NSE Official CSV + BSE support
Downloads the complete equity list from nsearchives.nseindia.com on startup.
"""

import json, os, time, io, logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".stock_cache.json")
CACHE_MAX_AGE = 86400  # Re-fetch daily
NSE_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

_all_stocks = []


def _fetch_nse_csv() -> list:
    """Download ALL NSE-listed equities from NSE's official CSV."""
    stocks = []
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        # Hit NSE homepage first to get cookies
        try:
            session.get("https://www.nseindia.com", timeout=10)
        except Exception:
            pass

        resp = session.get(NSE_CSV_URL, timeout=15)
        session.close()

        if resp.status_code != 200:
            logger.warning(f"NSE CSV returned status {resp.status_code}")
            return stocks

        df = pd.read_csv(io.BytesIO(resp.content))
        # Columns: SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE

        symbol_col = None
        name_col = None
        for col in df.columns:
            cl = col.strip().upper()
            if cl == "SYMBOL":
                symbol_col = col
            elif "NAME" in cl and "COMPANY" in cl:
                name_col = col

        if symbol_col is None or name_col is None:
            # Try positional
            symbol_col = df.columns[0]
            name_col = df.columns[1]

        for _, row in df.iterrows():
            sym = str(row[symbol_col]).strip()
            name = str(row[name_col]).strip()
            if not sym or sym == "SYMBOL" or sym == "nan":
                continue
            stocks.append({
                "name": name,
                "symbol": sym,
                "exchange": "NSE",
                "yf_symbol": f"{sym}.NS",
            })

        logger.info(f"Fetched {len(stocks)} NSE stocks from EQUITY_L.csv")
    except Exception as e:
        logger.warning(f"Failed to fetch NSE CSV: {e}")
    return stocks


def _fetch_nsetools_fallback() -> list:
    """Fallback: try nsetools if CSV download fails."""
    stocks = []
    try:
        from nsetools import Nse
        nse = Nse()
        codes = nse.get_stock_codes()
        if codes and len(codes) > 10:
            for sym, name in codes.items():
                if sym == "SYMBOL":
                    continue
                stocks.append({"name": name, "symbol": sym, "exchange": "NSE", "yf_symbol": f"{sym}.NS"})
            logger.info(f"Loaded {len(stocks)} stocks from nsetools fallback")
    except Exception as e:
        logger.warning(f"nsetools fallback also failed: {e}")
    return stocks


def _load_from_cache() -> Optional[list]:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            age = time.time() - data.get("timestamp", 0)
            if age < CACHE_MAX_AGE:
                logger.info(f"Loaded {len(data['stocks'])} stocks from cache (age: {int(age/3600)}h)")
                return data["stocks"]
            else:
                logger.info(f"Cache expired (age: {int(age/3600)}h), will re-fetch")
    except Exception:
        pass
    return None


def _save_to_cache(stocks: list):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"timestamp": time.time(), "stocks": stocks, "count": len(stocks)}, f, ensure_ascii=False)
        logger.info(f"Saved {len(stocks)} stocks to cache")
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


def _get_hardcoded_fallback() -> list:
    """Absolute last resort — top 100 stocks + major indices."""
    # INDICES (special — yfinance uses ^NSEI etc)
    indices = [
        ("Nifty 50", "NIFTY50", "^NSEI"),
        ("Nifty Bank", "BANKNIFTY", "^NSEBANK"),
        ("Sensex", "SENSEX", "^BSESN"),
        ("Nifty IT", "NIFTYIT", "^CNXIT"),
        ("Nifty Midcap 50", "NIFTYMIDCAP", "NIFTY_MIDCAP_50.NS"),
    ]
    top = [
        ("Reliance Industries Limited", "RELIANCE"), ("Tata Consultancy Services Limited", "TCS"),
        ("HDFC Bank Limited", "HDFCBANK"), ("Infosys Limited", "INFY"), ("ICICI Bank Limited", "ICICIBANK"),
        ("Hindustan Unilever Limited", "HINDUNILVR"), ("ITC Limited", "ITC"),
        ("State Bank of India", "SBIN"), ("Bharti Airtel Limited", "BHARTIARTL"),
        ("Kotak Mahindra Bank Limited", "KOTAKBANK"), ("Larsen & Toubro Limited", "LT"),
        ("Axis Bank Limited", "AXISBANK"), ("Asian Paints Limited", "ASIANPAINT"),
        ("Maruti Suzuki India Limited", "MARUTI"), ("Titan Company Limited", "TITAN"),
        ("Sun Pharmaceutical Industries Limited", "SUNPHARMA"), ("Bajaj Finance Limited", "BAJFINANCE"),
        ("Wipro Limited", "WIPRO"), ("HCL Technologies Limited", "HCLTECH"),
        ("NTPC Limited", "NTPC"), ("Tata Motors Limited", "TATAMOTORS"),
        ("Mahindra & Mahindra Limited", "M&M"), ("Adani Enterprises Limited", "ADANIENT"),
        ("Power Grid Corporation of India Limited", "POWERGRID"),
        ("Bajaj Finserv Limited", "BAJAJFINSV"), ("Tech Mahindra Limited", "TECHM"),
        ("JSW Steel Limited", "JSWSTEEL"), ("Tata Steel Limited", "TATASTEEL"),
        ("Coal India Limited", "COALINDIA"), ("Oil & Natural Gas Corporation Limited", "ONGC"),
        ("Cipla Limited", "CIPLA"), ("Dr. Reddy's Laboratories Limited", "DRREDDY"),
        ("Hindalco Industries Limited", "HINDALCO"), ("Bharat Petroleum Corporation Limited", "BPCL"),
        ("Eternal Limited", "ETERNAL"), ("DLF Limited", "DLF"),
        ("Hindustan Aeronautics Limited", "HAL"), ("Bharat Electronics Limited", "BEL"),
        ("Indian Railway Catering And Tourism Corporation Limited", "IRCTC"),
        ("Tata Power Company Limited", "TATAPOWER"), ("Suzlon Energy Limited", "SUZLON"),
        ("Indian Renewable Energy Development Agency Limited", "IREDA"),
        ("Trent Limited", "TRENT"), ("Avenue Supermarts Limited", "DMART"),
        ("Bajaj Auto Limited", "BAJAJ-AUTO"), ("Eicher Motors Limited", "EICHERMOT"),
        ("Apollo Hospitals Enterprise Limited", "APOLLOHOSP"),
        ("Nestle India Limited", "NESTLEIND"), ("UltraTech Cement Limited", "ULTRACEMCO"),
        ("Hero MotoCorp Limited", "HEROMOTOCO"), ("IndusInd Bank Limited", "INDUSINDBK"),
        ("Adani Ports and Special Economic Zone Limited", "ADANIPORTS"),
        ("Grasim Industries Limited", "GRASIM"), ("Britannia Industries Limited", "BRITANNIA"),
        ("Divi's Laboratories Limited", "DIVISLAB"), ("SBI Life Insurance Company Limited", "SBILIFE"),
        ("HDFC Life Insurance Company Limited", "HDFCLIFE"),
        ("Tata Consumer Products Limited", "TATACONSUM"),
        ("LTIMindtree Limited", "LTIM"), ("InterGlobe Aviation Limited", "INDIGO"),
        ("Siemens Limited", "SIEMENS"), ("ABB India Limited", "ABB"),
        ("Marico Limited", "MARICO"), ("Info Edge (India) Limited", "NAUKRI"),
        ("Cholamandalam Investment and Finance Company Limited", "CHOLAFIN"),
        ("Bank of Baroda", "BANKBARODA"), ("Punjab National Bank", "PNB"),
        ("Canara Bank", "CANBK"), ("DLF Limited", "DLF"),
        ("Godrej Properties Limited", "GODREJPROP"), ("Polycab India Limited", "POLYCAB"),
        ("Dixon Technologies (India) Limited", "DIXON"),
        ("Persistent Systems Limited", "PERSISTENT"), ("Coforge Limited", "COFORGE"),
        ("Varun Beverages Limited", "VBL"), ("JSW Energy Limited", "JSWENERGY"),
        ("Jindal Steel & Power Limited", "JINDALSTEL"),
        ("Tata Elxsi Limited", "TATAELXSI"), ("Bandhan Bank Limited", "BANDHANBNK"),
        ("IDFC First Bank Limited", "IDFCFIRSTB"), ("Federal Bank Limited", "FEDERALBNK"),
        ("AU Small Finance Bank Limited", "AUBANK"), ("Shriram Finance Limited", "SHRIRAMFIN"),
        ("Mazagon Dock Shipbuilders Limited", "MAZDOCK"), ("Cochin Shipyard Limited", "COCHINSHIP"),
        ("Rail Vikas Nigam Limited", "RVNL"), ("IRFC Limited", "IRFC"),
        ("CG Power and Industrial Solutions Limited", "CGPOWER"),
        ("KEI Industries Limited", "KEI"), ("Max Healthcare Institute Limited", "MAXHEALTH"),
        ("Lupin Limited", "LUPIN"), ("Aurobindo Pharma Limited", "AUROPHARMA"),
        ("Indian Hotels Company Limited", "INDHOTEL"), ("MRF Limited", "MRF"),
        ("Bosch Limited", "BOSCHLTD"), ("Adani Green Energy Limited", "ADANIGREEN"),
        ("Adani Power Limited", "ADANIPOWER"), ("NHPC Limited", "NHPC"),
        ("Indian Oil Corporation Limited", "IOC"), ("GAIL (India) Limited", "GAIL"),
        ("Vedanta Limited", "VEDL"), ("Dabur India Limited", "DABUR"),
        ("Pidilite Industries Limited", "PIDILITIND"), ("Havells India Limited", "HAVELLS"),
        ("PB Fintech Limited", "POLICYBZR"),
    ]
    # Build stock list + indices
    stock_list = [{"name": n, "symbol": s, "exchange": "NSE", "yf_symbol": f"{s}.NS"} for n, s in top]
    index_list = [{"name": n, "symbol": s, "exchange": "INDEX", "yf_symbol": yf} for n, s, yf in indices]
    return index_list + stock_list


def initialize_stocks():
    """Initialize full stock list. Call on app startup."""
    global _all_stocks

    # 1. Try cache
    cached = _load_from_cache()
    if cached and len(cached) > 500:
        _all_stocks = cached
        return

    # 2. Try NSE CSV (best source — all ~2500 stocks with current tickers)
    stocks = _fetch_nse_csv()
    if stocks and len(stocks) > 500:
        _all_stocks = stocks
        _save_to_cache(stocks)
        return

    # 3. Try nsetools
    stocks = _fetch_nsetools_fallback()
    if stocks and len(stocks) > 100:
        _all_stocks = stocks
        _save_to_cache(stocks)
        return

    # 4. Hardcoded fallback
    _all_stocks = _get_hardcoded_fallback()
    logger.warning(f"Using hardcoded fallback ({len(_all_stocks)} stocks)")


def search_stocks(query: str, limit: int = 20) -> list:
    """Fuzzy search stocks by name or symbol. Returns matching stocks sorted by relevance."""
    if not query or len(query) < 1:
        return []

    q = query.lower().strip()
    results = []

    for stock in _all_stocks:
        nl = stock["name"].lower()
        sl = stock["symbol"].lower()
        score = 0

        if q == sl:
            score = 100  # Exact symbol match
        elif sl.startswith(q):
            score = 80   # Symbol starts with query
        elif nl.startswith(q):
            score = 70   # Name starts with query
        elif q in sl:
            score = 60   # Query in symbol
        elif q in nl:
            score = 50   # Query in name
        elif any(w.startswith(q) for w in nl.split()):
            score = 40   # Word in name starts with query
        else:
            continue

        results.append({**stock, "score": score})

    results.sort(key=lambda x: (-x["score"], x["name"]))
    return results[:limit]


def get_stock_count() -> int:
    return len(_all_stocks)


def get_yf_symbol(symbol: str, exchange: str = "NSE") -> str:
    """Convert plain symbol to yfinance format."""
    if symbol.startswith("^"):
        return symbol
    # Check if this is a known index
    for s in _all_stocks:
        if s["symbol"] == symbol and s.get("exchange") == "INDEX":
            return s["yf_symbol"]
    if exchange.upper() == "INDEX":
        # Lookup from stock list
        for s in _all_stocks:
            if s["symbol"] == symbol:
                return s["yf_symbol"]
    if exchange.upper() == "BSE":
        return f"{symbol}.BO"
    return f"{symbol}.NS"
