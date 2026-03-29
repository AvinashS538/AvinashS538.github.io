"""
Data Layer — Smart data fetching with nsetools (primary) → yfinance (fallback)
Includes intelligent caching to respect API rate limits.
"""

import time
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional, Dict, Any

import yfinance as yf
import pandas as pd

# Try importing nsetools — graceful fallback if not available
try:
    from nsetools import Nse
    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False

from stock_universe import get_yf_symbol

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════
#  CACHE MANAGER
# ════════════════════════════════════════════════════════

class CacheManager:
    """Thread-safe in-memory cache with TTL (time-to-live) per entry."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry["expires_at"]:
                    return entry["data"]
                else:
                    del self._cache[key]
        return None

    def set(self, key: str, data: Any, ttl_seconds: int):
        with self._lock:
            self._cache[key] = {
                "data": data,
                "expires_at": time.time() + ttl_seconds,
                "cached_at": time.time(),
            }

    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            active = sum(1 for v in self._cache.values() if now < v["expires_at"])
            return {"total_entries": len(self._cache), "active_entries": active}


# Global cache instance
cache = CacheManager()

# Cache TTLs
LIVE_PRICE_TTL = 120       # 2 minutes
HISTORICAL_TTL = 900       # 15 minutes
FUNDAMENTALS_TTL = 86400   # 24 hours
SEARCH_TTL = 3600          # 1 hour

# ════════════════════════════════════════════════════════
#  NSE TOOLS DATA SOURCE
# ════════════════════════════════════════════════════════

def _fetch_nsetools_quote(symbol: str) -> Optional[dict]:
    """Fetch live quote from nsetools. Returns normalized dict or None on failure."""
    if not NSE_AVAILABLE:
        return None

    try:
        nse = Nse()
        quote = nse.get_quote(symbol)
        if not quote or "priceInfo" not in quote:
            return None

        price_info = quote.get("priceInfo", {})
        info = quote.get("info", {})

        return {
            "symbol": symbol,
            "name": info.get("companyName", symbol),
            "exchange": "NSE",
            "source": "nsetools",
            "current_price": price_info.get("lastPrice"),
            "open": price_info.get("open"),
            "high": price_info.get("high"),
            "low": price_info.get("low"),
            "close": price_info.get("close") or price_info.get("previousClose"),
            "previous_close": price_info.get("previousClose"),
            "change": price_info.get("change"),
            "change_pct": price_info.get("pChange"),
            "volume": quote.get("securityWiseDP", {}).get("quantityTraded"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.warning(f"nsetools failed for {symbol}: {e}")
        return None


# ════════════════════════════════════════════════════════
#  YFINANCE DATA SOURCE
# ════════════════════════════════════════════════════════

def _fetch_yfinance_quote(symbol: str, exchange: str = "NSE") -> Optional[dict]:
    """Fetch live quote from yfinance. Returns normalized dict or None."""
    try:
        yf_symbol = get_yf_symbol(symbol, exchange)
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        if not info or "regularMarketPrice" not in info:
            # Try fast_info as fallback
            fi = ticker.fast_info
            return {
                "symbol": symbol,
                "name": info.get("shortName", symbol),
                "exchange": exchange,
                "source": "yfinance",
                "current_price": fi.get("lastPrice") or fi.get("last_price"),
                "open": fi.get("open"),
                "high": fi.get("dayHigh") or fi.get("day_high"),
                "low": fi.get("dayLow") or fi.get("day_low"),
                "close": fi.get("previousClose") or fi.get("previous_close"),
                "previous_close": fi.get("previousClose") or fi.get("previous_close"),
                "change": None,
                "change_pct": None,
                "volume": fi.get("lastVolume") or fi.get("last_volume"),
                "timestamp": datetime.now().isoformat(),
            }

        current = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
        change = round(current - prev_close, 2) if current and prev_close else None
        change_pct = round((change / prev_close) * 100, 2) if change and prev_close else None

        return {
            "symbol": symbol,
            "name": info.get("shortName", symbol),
            "exchange": exchange,
            "source": "yfinance",
            "current_price": current,
            "open": info.get("regularMarketOpen") or info.get("open"),
            "high": info.get("regularMarketDayHigh") or info.get("dayHigh"),
            "low": info.get("regularMarketDayLow") or info.get("dayLow"),
            "close": prev_close,
            "previous_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "volume": info.get("regularMarketVolume") or info.get("volume"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.warning(f"yfinance quote failed for {symbol} ({exchange}): {e}")
        return None


# ════════════════════════════════════════════════════════
#  SMART FETCH (nsetools → yfinance fallback)
# ════════════════════════════════════════════════════════

def get_live_quote(symbol: str, exchange: str = "NSE") -> Optional[dict]:
    """
    Get live quote with smart fallback:
    1. Check cache first
    2. If NSE → try nsetools first → fallback to yfinance
    3. If BSE → go directly to yfinance
    """
    cache_key = f"quote:{symbol}:{exchange}"
    cached = cache.get(cache_key)
    if cached:
        cached["from_cache"] = True
        return cached

    result = None

    # For NSE: try nsetools first
    if exchange.upper() == "NSE":
        result = _fetch_nsetools_quote(symbol)

    # Fallback to yfinance (or primary for BSE)
    if result is None:
        result = _fetch_yfinance_quote(symbol, exchange)

    if result:
        result["from_cache"] = False
        cache.set(cache_key, result, LIVE_PRICE_TTL)

    return result


def get_historical_data(
    symbol: str,
    exchange: str = "NSE",
    period: str = "6mo",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data. Always uses yfinance (best source for historical).

    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        exchange: "NSE" or "BSE"
        period: "1d","5d","1mo","3mo","6mo","1y","2y","5y","max"
        interval: "1m","2m","5m","15m","60m","1d","1wk","1mo"

    Note: 1m data only available for last 7 days.
          2m/5m/15m data available for last 60 days.
    """
    cache_key = f"hist:{symbol}:{exchange}:{period}:{interval}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        yf_symbol = get_yf_symbol(symbol, exchange)
        df = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df is None or df.empty:
            logger.warning(f"No historical data for {yf_symbol}")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column names
        df.columns = [c.strip().title() for c in df.columns]
        for required in ["Open", "High", "Low", "Close", "Volume"]:
            if required not in df.columns:
                logger.warning(f"Missing column {required} in data for {yf_symbol}")
                return None

        # Remove rows with NaN in critical columns
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        # Set appropriate cache TTL based on interval
        ttl = HISTORICAL_TTL
        if interval in ("1m", "2m", "5m"):
            ttl = 60  # 1 minute for intraday
        elif interval == "15m":
            ttl = 120  # 2 minutes

        cache.set(cache_key, df, ttl)
        return df

    except Exception as e:
        logger.warning(f"Historical data failed for {symbol} ({exchange}): {e}")
        return None


def get_fundamentals(symbol: str, exchange: str = "NSE") -> Optional[dict]:
    """Fetch fundamental data with multiple fallback approaches."""
    cache_key = f"fundamentals:{symbol}:{exchange}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        yf_symbol = get_yf_symbol(symbol, exchange)
        ticker = yf.Ticker(yf_symbol)

        # ── Approach 1: Try .info (sometimes empty for Indian stocks) ──
        info = {}
        try:
            info = ticker.info or {}
            logger.info(f"Fundamentals .info for {yf_symbol}: {len(info)} fields")
        except Exception as e:
            logger.warning(f".info failed for {yf_symbol}: {e}")

        # ── Approach 2: Use .fast_info as fallback for price data ──
        fast = {}
        try:
            fi = ticker.fast_info
            fast = {
                "market_cap": getattr(fi, "market_cap", None),
                "last_price": getattr(fi, "last_price", None),
                "fifty_day_average": getattr(fi, "fifty_day_average", None),
                "two_hundred_day_average": getattr(fi, "two_hundred_day_average", None),
                "year_high": getattr(fi, "year_high", None),
                "year_low": getattr(fi, "year_low", None),
                "shares": getattr(fi, "shares", None),
            }
        except Exception:
            pass

        # ── Approach 3: Calculate from financial statements ──
        income_data = {}
        balance_data = {}
        cashflow_data = {}

        try:
            inc = ticker.financials
            if inc is not None and not inc.empty:
                latest = inc.iloc[:, 0]  # Most recent year
                income_data = {
                    "total_revenue": _safe_get(latest, "Total Revenue"),
                    "gross_profit": _safe_get(latest, "Gross Profit"),
                    "operating_income": _safe_get(latest, "Operating Income"),
                    "net_income": _safe_get(latest, "Net Income"),
                    "ebitda": _safe_get(latest, "EBITDA"),
                }
        except Exception:
            pass

        try:
            bs = ticker.balance_sheet
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]
                total_assets = _safe_get(latest, "Total Assets")
                total_equity = _safe_get(latest, "Stockholders Equity") or _safe_get(latest, "Total Equity Gross Minority Interest")
                total_debt_val = _safe_get(latest, "Total Debt") or _safe_get(latest, "Net Debt")
                current_assets = _safe_get(latest, "Current Assets")
                current_liab = _safe_get(latest, "Current Liabilities")
                total_cash_val = _safe_get(latest, "Cash And Cash Equivalents")

                balance_data = {
                    "total_assets": total_assets,
                    "total_equity": total_equity,
                    "total_debt": total_debt_val,
                    "total_cash": total_cash_val,
                    "current_assets": current_assets,
                    "current_liabilities": current_liab,
                }
        except Exception:
            pass

        try:
            cf = ticker.cashflow
            if cf is not None and not cf.empty:
                latest = cf.iloc[:, 0]
                cashflow_data = {
                    "operating_cashflow": _safe_get(latest, "Operating Cash Flow") or _safe_get(latest, "Cash Flow From Continuing Operating Activities"),
                    "free_cashflow": _safe_get(latest, "Free Cash Flow"),
                    "capex": _safe_get(latest, "Capital Expenditure"),
                }
        except Exception:
            pass

        # ── Approach 4: Compute 52-week from historical data ──
        hist_52w = None
        try:
            hist = ticker.history(period="1y")
            if hist is not None and not hist.empty:
                hist_52w = {
                    "high": round(hist["High"].max(), 2),
                    "low": round(hist["Low"].min(), 2),
                }
        except Exception:
            pass

        # ── Calculate derived ratios from statement data ──
        calc_pe = None
        calc_pb = None
        calc_roe = None
        calc_de = None
        calc_cr = None
        calc_margins = None
        calc_op_margin = None

        mkt_cap = info.get("marketCap") or fast.get("market_cap")
        net_income = income_data.get("net_income")
        total_equity = balance_data.get("total_equity")
        total_revenue = income_data.get("total_revenue")
        total_debt_v = balance_data.get("total_debt")
        operating_income = income_data.get("operating_income")
        current_assets = balance_data.get("current_assets")
        current_liab = balance_data.get("current_liabilities")

        if mkt_cap and net_income and net_income > 0:
            calc_pe = round(mkt_cap / net_income, 2)
        if mkt_cap and total_equity and total_equity > 0:
            calc_pb = round(mkt_cap / total_equity, 2)
        if net_income and total_equity and total_equity > 0:
            calc_roe = round(net_income / total_equity, 4)
        if total_debt_v is not None and total_equity and total_equity > 0:
            calc_de = round(total_debt_v / total_equity * 100, 1)
        if current_assets and current_liab and current_liab > 0:
            calc_cr = round(current_assets / current_liab, 2)
        if net_income and total_revenue and total_revenue > 0:
            calc_margins = round(net_income / total_revenue, 4)
        if operating_income and total_revenue and total_revenue > 0:
            calc_op_margin = round(operating_income / total_revenue, 4)

        # ── Compute EPS ──
        shares = fast.get("shares")
        calc_eps = None
        if net_income and shares and shares > 0:
            calc_eps = round(net_income / shares, 2)

        # ── News ──
        news = []
        try:
            n = ticker.news
            if n:
                for item in n[:10]:
                    if isinstance(item, dict):
                        content = item.get("content", {})
                        if isinstance(content, dict):
                            title = content.get("title") or item.get("title", "")
                            publisher = ""
                            prov = content.get("provider")
                            if isinstance(prov, dict):
                                publisher = prov.get("displayName", "")
                            elif isinstance(item.get("publisher"), str):
                                publisher = item.get("publisher", "")
                            link = ""
                            curl = content.get("canonicalUrl")
                            if isinstance(curl, dict):
                                link = curl.get("url", "")
                            elif isinstance(item.get("link"), str):
                                link = item.get("link", "")
                        else:
                            title = item.get("title", "")
                            publisher = item.get("publisher", "")
                            link = item.get("link", "")
                        if title:
                            news.append({"title": title, "publisher": publisher, "link": link})
        except Exception:
            pass

        # ── Holders ──
        holders_data = None
        try:
            h = ticker.major_holders
            if h is not None and not h.empty:
                holders_data = []
                for _, row in h.iterrows():
                    holders_data.append({"value": str(row.iloc[0]), "label": str(row.iloc[1]) if len(row) > 1 else ""})
        except Exception:
            pass

        # ── Merge: prefer .info values, fallback to calculated ──
        result = {
            "symbol": symbol,
            "exchange": exchange,
            "name": info.get("shortName") or info.get("longName") or symbol,
            "sector": info.get("sector") or "—",
            "industry": info.get("industry") or "—",
            "description": (info.get("longBusinessSummary") or "")[:500],

            # Key ratios (prefer .info, fallback to calculated)
            "pe_ratio": info.get("trailingPE") or calc_pe,
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook") or calc_pb,
            "eps": info.get("trailingEps") or calc_eps,
            "market_cap": mkt_cap,
            "enterprise_value": info.get("enterpriseValue"),
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh") or (hist_52w["high"] if hist_52w else None),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow") or (hist_52w["low"] if hist_52w else None),

            # Profitability
            "revenue": info.get("totalRevenue") or total_revenue,
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins") or calc_op_margin,
            "profit_margins": info.get("profitMargins") or calc_margins,
            "roe": info.get("returnOnEquity") or calc_roe,
            "roa": info.get("returnOnAssets"),

            # Debt
            "total_debt": info.get("totalDebt") or balance_data.get("total_debt"),
            "total_cash": info.get("totalCash") or balance_data.get("total_cash"),
            "debt_to_equity": info.get("debtToEquity") or calc_de,
            "current_ratio": info.get("currentRatio") or calc_cr,
            "quick_ratio": info.get("quickRatio"),

            # Cash flow
            "free_cashflow": info.get("freeCashflow") or cashflow_data.get("free_cashflow"),
            "operating_cashflow": info.get("operatingCashflow") or cashflow_data.get("operating_cashflow"),

            # Statement summaries
            "income_summary": income_data if any(v is not None for v in income_data.values()) else None,
            "balance_summary": balance_data if any(v is not None for v in balance_data.values()) else None,
            "cashflow_summary": cashflow_data if any(v is not None for v in cashflow_data.values()) else None,

            # Holders
            "major_holders": holders_data,

            # News
            "news": news,

            "data_source": "yfinance" + (" + statements" if income_data.get("total_revenue") else ""),
            "timestamp": datetime.now().isoformat(),
        }

        cache.set(cache_key, result, FUNDAMENTALS_TTL)
        return result

    except Exception as e:
        logger.warning(f"Fundamentals failed for {symbol} ({exchange}): {e}")
        return None


def _safe_get(series, key):
    """Safely extract a value from a pandas Series by label."""
    try:
        if key in series.index:
            val = series[key]
            if pd.notna(val):
                return float(val)
    except Exception:
        pass
    return None


def get_cache_stats() -> dict:
    """Return cache statistics."""
    return cache.stats()
