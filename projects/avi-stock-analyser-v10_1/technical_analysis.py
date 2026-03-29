"""
Technical Analysis Engine
=========================
Computes all technical indicators and generates buy/sell signals.
Uses pandas-ta where available, falls back to manual calculation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Try pandas_ta, fallback to manual calculations
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.info("pandas_ta not available, using manual indicator calculations")


# ════════════════════════════════════════════════════════
#  INDIVIDUAL INDICATOR CALCULATIONS
# ════════════════════════════════════════════════════════

def calc_sma(df: pd.DataFrame, periods: list = [20, 50, 200]) -> dict:
    """Simple Moving Average."""
    result = {}
    for p in periods:
        col = f"SMA_{p}"
        if len(df) >= p:
            df[col] = df["Close"].rolling(window=p).mean()
            result[col] = df[col].iloc[-1]
        else:
            result[col] = None
    return result


def calc_ema(df: pd.DataFrame, periods: list = [9, 21, 50]) -> dict:
    """Exponential Moving Average."""
    result = {}
    for p in periods:
        col = f"EMA_{p}"
        if len(df) >= p:
            df[col] = df["Close"].ewm(span=p, adjust=False).mean()
            result[col] = df[col].iloc[-1]
        else:
            result[col] = None
    return result


def calc_rsi(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Relative Strength Index."""
    if len(df) < period + 1:
        return None
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder's smoothing after initial SMA
    for i in range(period, len(avg_gain)):
        if pd.notna(avg_gain.iloc[i - 1]):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi
    return rsi.iloc[-1]


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD — Moving Average Convergence Divergence."""
    if len(df) < slow + signal:
        return {"macd": None, "signal": None, "histogram": None}

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = histogram

    return {
        "macd": macd_line.iloc[-1],
        "signal": signal_line.iloc[-1],
        "histogram": histogram.iloc[-1],
    }


def calc_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands."""
    if len(df) < period:
        return {"upper": None, "middle": None, "lower": None, "bandwidth": None}

    middle = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    df["BB_Upper"] = upper
    df["BB_Middle"] = middle
    df["BB_Lower"] = lower

    curr_price = df["Close"].iloc[-1]
    bandwidth = ((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]) * 100 if middle.iloc[-1] else None

    return {
        "upper": upper.iloc[-1],
        "middle": middle.iloc[-1],
        "lower": lower.iloc[-1],
        "bandwidth": bandwidth,
        "percent_b": ((curr_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])) if upper.iloc[-1] != lower.iloc[-1] else 0.5,
    }


def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    """Stochastic Oscillator (%K and %D)."""
    if len(df) < k_period + d_period:
        return {"k": None, "d": None}

    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()

    df["Stoch_K"] = k
    df["Stoch_D"] = d

    return {"k": k.iloc[-1], "d": d.iloc[-1]}


def calc_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Average True Range — volatility indicator."""
    if len(df) < period + 1:
        return None

    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close).abs(),
        "lc": (low - close).abs(),
    }).max(axis=1)

    atr = tr.rolling(window=period).mean()
    df["ATR"] = atr
    return atr.iloc[-1]


def calc_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Average Directional Index — trend strength."""
    if len(df) < period * 2:
        return None

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low - close.shift(1)).abs(),
    }).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    df["ADX"] = adx
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di

    return adx.iloc[-1]


def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict:
    """Supertrend indicator."""
    if len(df) < period + 1:
        return {"value": None, "direction": None}

    hl2 = (df["High"] + df["Low"]) / 2
    atr = calc_atr(df.copy(), period) or 0

    # Simplified: use current ATR for bands
    atr_series = pd.Series(index=df.index, dtype=float)
    tr = pd.DataFrame({
        "hl": df["High"] - df["Low"],
        "hc": (df["High"] - df["Close"].shift(1)).abs(),
        "lc": (df["Low"] - df["Close"].shift(1)).abs(),
    }).max(axis=1)
    atr_series = tr.rolling(window=period).mean()

    upper_band = hl2 + (multiplier * atr_series)
    lower_band = hl2 - (multiplier * atr_series)

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    supertrend.iloc[period] = upper_band.iloc[period]
    direction.iloc[period] = -1

    for i in range(period + 1, len(df)):
        if df["Close"].iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1]) if direction.iloc[i - 1] == 1 else lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1]) if direction.iloc[i - 1] == -1 else upper_band.iloc[i]
            direction.iloc[i] = -1

    df["Supertrend"] = supertrend
    df["Supertrend_Dir"] = direction

    return {
        "value": supertrend.iloc[-1],
        "direction": "bullish" if direction.iloc[-1] == 1 else "bearish",
    }


def calc_vwap(df: pd.DataFrame) -> Optional[float]:
    """Volume Weighted Average Price."""
    if len(df) < 2 or "Volume" not in df.columns:
        return None
    try:
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["VWAP"] = vwap
        return vwap.iloc[-1]
    except Exception:
        return None


def calc_obv(df: pd.DataFrame) -> Optional[float]:
    """On Balance Volume."""
    if len(df) < 2 or "Volume" not in df.columns:
        return None

    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df["Volume"].iloc[i]
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df["Volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    df["OBV"] = obv
    return obv.iloc[-1]


def calc_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Fibonacci retracement levels from recent swing high/low."""
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
    high = recent["High"].max()
    low = recent["Low"].min()
    diff = high - low

    return {
        "swing_high": high,
        "swing_low": low,
        "level_0": high,
        "level_236": high - (diff * 0.236),
        "level_382": high - (diff * 0.382),
        "level_500": high - (diff * 0.500),
        "level_618": high - (diff * 0.618),
        "level_786": high - (diff * 0.786),
        "level_100": low,
    }


def calc_support_resistance(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Auto-detect support and resistance levels from price pivots."""
    if len(df) < 10:
        return {"support": [], "resistance": []}

    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values
    closes = recent["Close"].values
    current = closes[-1]

    # Find local pivots
    resistance_levels = []
    support_levels = []

    for i in range(2, len(highs) - 2):
        # Local high (resistance)
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistance_levels.append(round(highs[i], 2))
        # Local low (support)
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            support_levels.append(round(lows[i], 2))

    # Cluster nearby levels (within 1% of each other)
    def cluster_levels(levels, threshold_pct=0.01):
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - clustered[-1]) / clustered[-1] < threshold_pct:
                clustered[-1] = (clustered[-1] + lvl) / 2  # Average
            else:
                clustered.append(lvl)
        return [round(l, 2) for l in clustered]

    support = [s for s in cluster_levels(support_levels) if s < current]
    resistance = [r for r in cluster_levels(resistance_levels) if r > current]

    return {
        "support": sorted(support, reverse=True)[:3],
        "resistance": sorted(resistance)[:3],
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> list:
    """Detect common candlestick patterns in last few candles."""
    patterns = []
    if len(df) < 3:
        return patterns

    o, h, l, c = df["Open"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
    po, ph, pl, pc = df["Open"].iloc[-2], df["High"].iloc[-2], df["Low"].iloc[-2], df["Close"].iloc[-2]
    body = abs(c - o)
    prev_body = abs(pc - po)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    candle_range = h - l

    if candle_range == 0:
        return patterns

    # Doji: very small body relative to range
    if body / candle_range < 0.1:
        patterns.append({"pattern": "Doji", "signal": "neutral", "description": "Indecision — buyers and sellers balanced"})

    # Hammer: small body at top, long lower shadow (bullish reversal)
    if lower_shadow > body * 2 and upper_shadow < body * 0.5 and c > o:
        patterns.append({"pattern": "Hammer", "signal": "bullish", "description": "Potential bullish reversal — rejection of lower prices"})

    # Shooting Star: small body at bottom, long upper shadow (bearish reversal)
    if upper_shadow > body * 2 and lower_shadow < body * 0.5 and c < o:
        patterns.append({"pattern": "Shooting Star", "signal": "bearish", "description": "Potential bearish reversal — rejection of higher prices"})

    # Bullish Engulfing
    if pc > po and c > o and o <= pc and c >= po and body > prev_body:
        patterns.append({"pattern": "Bullish Engulfing", "signal": "bullish", "description": "Strong buying — bullish candle engulfs previous bearish candle"})

    # Bearish Engulfing
    if po > pc and o > c and o >= po and c <= pc and body > prev_body:
        patterns.append({"pattern": "Bearish Engulfing", "signal": "bearish", "description": "Strong selling — bearish candle engulfs previous bullish candle"})

    # Morning Star (3-candle bullish reversal)
    if len(df) >= 3:
        o3, c3 = df["Open"].iloc[-3], df["Close"].iloc[-3]
        body3 = abs(c3 - o3)
        if c3 < o3 and body3 > candle_range * 0.3 and prev_body < body3 * 0.3 and c > o and c > (o3 + c3) / 2:
            patterns.append({"pattern": "Morning Star", "signal": "bullish", "description": "Three-candle bullish reversal pattern"})

    return patterns


def detect_elliott_wave_basic(df: pd.DataFrame, lookback: int = 100) -> dict:
    """
    Basic Elliott Wave detection — identifies potential wave structure.
    This is simplified — real Elliott Wave requires subjective analysis.
    """
    if len(df) < lookback:
        lookback = len(df)
    if lookback < 20:
        return {"status": "insufficient_data", "wave": None, "description": "Need more data for wave analysis"}

    recent = df.tail(lookback)
    closes = recent["Close"].values

    # Find significant swing points (simplified)
    pivots = []
    threshold = np.std(closes) * 0.3  # Significance threshold

    for i in range(5, len(closes) - 5):
        local_max = closes[i] == max(closes[i-5:i+6])
        local_min = closes[i] == min(closes[i-5:i+6])
        if local_max:
            pivots.append(("high", i, closes[i]))
        elif local_min:
            pivots.append(("low", i, closes[i]))

    if len(pivots) < 5:
        return {"status": "no_clear_pattern", "wave": None, "description": "No clear wave structure detected"}

    # Check for impulse wave (5 waves up) or corrective wave (3 waves down)
    last_5 = pivots[-5:]
    types = [p[0] for p in last_5]
    values = [p[2] for p in last_5]

    # Simplified: check if alternating highs and lows with upward trend
    if types == ["low", "high", "low", "high", "low"]:
        if values[1] > values[0] and values[3] > values[2] and values[3] > values[1]:
            return {
                "status": "impulse_up",
                "wave": "Wave 5 completing (impulse)",
                "description": "Upward 5-wave impulse pattern detected. Wave 5 may be completing — watch for reversal.",
                "signal": "cautious_bullish",
            }

    if types == ["high", "low", "high", "low", "high"]:
        if values[1] < values[0] and values[3] < values[2]:
            return {
                "status": "corrective_down",
                "wave": "ABC Correction",
                "description": "Downward corrective pattern (ABC) detected. Correction may be nearing completion.",
                "signal": "cautious_bullish",
            }

    # Check for downward impulse
    if types == ["high", "low", "high", "low", "high"]:
        if values[1] < values[0] and values[3] < values[1]:
            return {
                "status": "impulse_down",
                "wave": "Bearish impulse",
                "description": "Downward impulse pattern detected. Strong bearish momentum.",
                "signal": "bearish",
            }

    current_price = closes[-1]
    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

    if position > 0.7:
        return {"status": "near_high", "wave": "Possible wave 3/5 top", "description": "Price near recent highs — could be topping wave.", "signal": "cautious"}
    elif position < 0.3:
        return {"status": "near_low", "wave": "Possible wave A/C bottom", "description": "Price near recent lows — could be bottoming.", "signal": "cautious_bullish"}
    else:
        return {"status": "mid_range", "wave": "Mid-wave", "description": "Price in middle of recent range — wave position unclear.", "signal": "neutral"}


# ════════════════════════════════════════════════════════
#  PROJECTED RANGE (30-min forward estimate)
# ════════════════════════════════════════════════════════

def calc_projected_range(df: pd.DataFrame, indicators: dict) -> dict:
    """
    Estimate a 30-minute forward price range based on current indicators.
    NOT a prediction — a probabilistic range band.
    """
    if len(df) < 20:
        return {"low": None, "high": None, "bias": "neutral", "confidence": "low"}

    current = df["Close"].iloc[-1]
    atr = indicators.get("atr")
    rsi = indicators.get("rsi")
    bb = indicators.get("bollinger", {})

    if atr is None:
        # Estimate ATR from recent volatility
        returns = df["Close"].pct_change().tail(20)
        atr = current * returns.std() * np.sqrt(20)

    # Base range: ~0.3 ATR for 30-min window (scaled from daily ATR)
    # Daily ATR / sqrt(number of 30-min periods in a day ~13) ≈ ATR / 3.6
    range_estimate = atr / 3.6

    # Adjust bias based on indicators
    bias_score = 0

    if rsi is not None:
        if rsi > 70:
            bias_score -= 1  # Overbought, likely to pull back
        elif rsi < 30:
            bias_score += 1  # Oversold, likely to bounce

    macd = indicators.get("macd", {})
    if macd.get("histogram") is not None:
        if macd["histogram"] > 0:
            bias_score += 0.5
        else:
            bias_score -= 0.5

    supertrend = indicators.get("supertrend", {})
    if supertrend.get("direction") == "bullish":
        bias_score += 0.5
    elif supertrend.get("direction") == "bearish":
        bias_score -= 0.5

    # Calculate range
    bias_shift = range_estimate * 0.2 * bias_score
    proj_low = round(current - range_estimate + bias_shift, 2)
    proj_high = round(current + range_estimate + bias_shift, 2)

    if bias_score > 0.5:
        bias = "bullish"
    elif bias_score < -0.5:
        bias = "bearish"
    else:
        bias = "neutral"

    return {
        "current": round(current, 2),
        "low": proj_low,
        "high": proj_high,
        "bias": bias,
        "bias_score": round(bias_score, 2),
        "confidence": "medium" if abs(bias_score) > 1 else "low",
        "note": "Estimated range based on ATR and current indicators. Not a prediction.",
    }


# ════════════════════════════════════════════════════════
#  ICHIMOKU CLOUD
# ════════════════════════════════════════════════════════

def calc_ichimoku(df: pd.DataFrame) -> dict:
    """Calculate Ichimoku Cloud components."""
    if len(df) < 52:
        return {}
    h, l, c = df["High"], df["Low"], df["Close"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    chikou = c.shift(-26)

    curr_price = float(c.iloc[-1])
    t_val = float(tenkan.iloc[-1]) if pd.notna(tenkan.iloc[-1]) else None
    k_val = float(kijun.iloc[-1]) if pd.notna(kijun.iloc[-1]) else None
    sa_val = float(senkou_a.iloc[-1]) if pd.notna(senkou_a.iloc[-1]) else None
    sb_val = float(senkou_b.iloc[-1]) if pd.notna(senkou_b.iloc[-1]) else None

    # Determine cloud position
    if sa_val and sb_val:
        cloud_top = max(sa_val, sb_val)
        cloud_bottom = min(sa_val, sb_val)
        if curr_price > cloud_top:
            position = "above_cloud"
        elif curr_price < cloud_bottom:
            position = "below_cloud"
        else:
            position = "in_cloud"
    else:
        position = "unknown"

    # TK Cross
    tk_cross = "neutral"
    if t_val and k_val:
        if t_val > k_val:
            tk_cross = "bullish"
        elif t_val < k_val:
            tk_cross = "bearish"

    return {
        "tenkan": round(t_val, 2) if t_val else None,
        "kijun": round(k_val, 2) if k_val else None,
        "senkou_a": round(sa_val, 2) if sa_val else None,
        "senkou_b": round(sb_val, 2) if sb_val else None,
        "cloud_position": position,
        "tk_cross": tk_cross,
    }


# ════════════════════════════════════════════════════════
#  WILLIAMS %R
# ════════════════════════════════════════════════════════

def calc_williams_r(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Williams %R oscillator. Range: -100 to 0."""
    if len(df) < period:
        return None
    h = df["High"].rolling(period).max()
    l = df["Low"].rolling(period).min()
    wr = -100 * (h.iloc[-1] - df["Close"].iloc[-1]) / (h.iloc[-1] - l.iloc[-1]) if (h.iloc[-1] - l.iloc[-1]) != 0 else -50
    return round(float(wr), 2)


# ════════════════════════════════════════════════════════
#  CHAIKIN MONEY FLOW (CMF)
# ════════════════════════════════════════════════════════

def calc_cmf(df: pd.DataFrame, period: int = 20) -> Optional[float]:
    """CMF — positive = buying pressure, negative = selling pressure."""
    if len(df) < period:
        return None
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    hl_range = h - l
    mfm = ((c - l) - (h - c)) / hl_range.replace(0, np.nan)
    mfv = mfm * v
    cmf = mfv.rolling(period).sum() / v.rolling(period).sum()
    val = cmf.iloc[-1]
    return round(float(val), 4) if pd.notna(val) else None


# ════════════════════════════════════════════════════════
#  PIVOT POINTS (Classic)
# ════════════════════════════════════════════════════════

def calc_pivot_points(df: pd.DataFrame) -> dict:
    """Classic pivot points from previous session."""
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    pp = (h + l + c) / 3
    return {
        "pp": round(pp, 2),
        "r1": round(2 * pp - l, 2),
        "r2": round(pp + (h - l), 2),
        "r3": round(h + 2 * (pp - l), 2),
        "s1": round(2 * pp - h, 2),
        "s2": round(pp - (h - l), 2),
        "s3": round(l - 2 * (h - pp), 2),
    }


# ════════════════════════════════════════════════════════
#  SIGNAL ENGINE — Combines all indicators into verdict
# ════════════════════════════════════════════════════════

def generate_signal(indicators: dict, current_price: float) -> dict:
    """
    Combine all indicators into a single verdict with reasoning.
    Returns: { verdict, score, bullish_count, bearish_count, neutral_count, reasons }
    """
    signals = []  # List of (indicator_name, signal, reason)

    # 1. RSI
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi > 70:
            signals.append(("RSI", "bearish", f"RSI at {rsi:.1f} — overbought territory (>70)"))
        elif rsi < 30:
            signals.append(("RSI", "bullish", f"RSI at {rsi:.1f} — oversold territory (<30)"))
        elif rsi > 55:
            signals.append(("RSI", "bullish", f"RSI at {rsi:.1f} — bullish momentum"))
        elif rsi < 45:
            signals.append(("RSI", "bearish", f"RSI at {rsi:.1f} — bearish momentum"))
        else:
            signals.append(("RSI", "neutral", f"RSI at {rsi:.1f} — neutral zone"))

    # 2. MACD
    macd = indicators.get("macd", {})
    if macd.get("macd") is not None:
        if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
            signals.append(("MACD", "bullish", "MACD above signal line — bullish crossover"))
        elif macd["histogram"] < 0 and macd["macd"] < macd["signal"]:
            signals.append(("MACD", "bearish", "MACD below signal line — bearish crossover"))
        else:
            signals.append(("MACD", "neutral", "MACD near signal line — no clear signal"))

    # 3. Bollinger Bands
    bb = indicators.get("bollinger", {})
    if bb.get("upper") is not None:
        pct_b = bb.get("percent_b", 0.5)
        if pct_b > 1.0:
            signals.append(("Bollinger", "bearish", f"Price above upper band — overbought, mean reversion likely"))
        elif pct_b < 0.0:
            signals.append(("Bollinger", "bullish", f"Price below lower band — oversold, bounce likely"))
        elif pct_b > 0.8:
            signals.append(("Bollinger", "bearish", f"Price near upper band — approaching resistance"))
        elif pct_b < 0.2:
            signals.append(("Bollinger", "bullish", f"Price near lower band — approaching support"))
        else:
            signals.append(("Bollinger", "neutral", "Price within Bollinger Bands — normal range"))

    # 4. Moving Averages (SMA trend)
    sma = indicators.get("sma", {})
    sma_20 = sma.get("SMA_20")
    sma_50 = sma.get("SMA_50")
    sma_200 = sma.get("SMA_200")

    if sma_20 and sma_50:
        if current_price > sma_20 > sma_50:
            signals.append(("SMA", "bullish", "Price > SMA20 > SMA50 — strong uptrend"))
        elif current_price < sma_20 < sma_50:
            signals.append(("SMA", "bearish", "Price < SMA20 < SMA50 — strong downtrend"))
        elif current_price > sma_20:
            signals.append(("SMA", "bullish", "Price above SMA20 — short-term bullish"))
        else:
            signals.append(("SMA", "bearish", "Price below SMA20 — short-term bearish"))

    if sma_200:
        if current_price > sma_200:
            signals.append(("SMA200", "bullish", "Price above 200-day SMA — long-term uptrend"))
        else:
            signals.append(("SMA200", "bearish", "Price below 200-day SMA — long-term downtrend"))

    # 5. EMA
    ema = indicators.get("ema", {})
    ema_9 = ema.get("EMA_9")
    ema_21 = ema.get("EMA_21")
    if ema_9 and ema_21:
        if ema_9 > ema_21:
            signals.append(("EMA", "bullish", "EMA9 > EMA21 — short-term momentum is up"))
        else:
            signals.append(("EMA", "bearish", "EMA9 < EMA21 — short-term momentum is down"))

    # 6. Supertrend
    st = indicators.get("supertrend", {})
    if st.get("direction"):
        if st["direction"] == "bullish":
            signals.append(("Supertrend", "bullish", "Supertrend is bullish — price above trend line"))
        else:
            signals.append(("Supertrend", "bearish", "Supertrend is bearish — price below trend line"))

    # 7. ADX (trend strength)
    adx = indicators.get("adx")
    if adx is not None:
        if adx > 25:
            signals.append(("ADX", "neutral", f"ADX at {adx:.1f} — strong trend in place (>25)"))
        else:
            signals.append(("ADX", "neutral", f"ADX at {adx:.1f} — weak/no trend (<25), ranging market"))

    # 8. Stochastic
    stoch = indicators.get("stochastic", {})
    if stoch.get("k") is not None:
        if stoch["k"] > 80:
            signals.append(("Stochastic", "bearish", f"Stochastic %K at {stoch['k']:.1f} — overbought (>80)"))
        elif stoch["k"] < 20:
            signals.append(("Stochastic", "bullish", f"Stochastic %K at {stoch['k']:.1f} — oversold (<20)"))
        else:
            signals.append(("Stochastic", "neutral", f"Stochastic %K at {stoch['k']:.1f} — mid range"))

    # 9. VWAP
    vwap = indicators.get("vwap")
    if vwap is not None:
        if current_price > vwap:
            signals.append(("VWAP", "bullish", "Price above VWAP — buying pressure"))
        else:
            signals.append(("VWAP", "bearish", "Price below VWAP — selling pressure"))

    # 10. Elliott Wave
    elliott = indicators.get("elliott_wave", {})
    if elliott.get("signal"):
        sig = elliott["signal"]
        if "bullish" in sig:
            signals.append(("Elliott Wave", "bullish", elliott.get("description", "")))
        elif "bearish" in sig:
            signals.append(("Elliott Wave", "bearish", elliott.get("description", "")))
        else:
            signals.append(("Elliott Wave", "neutral", elliott.get("description", "")))

    # 11. Candlestick Patterns
    patterns = indicators.get("candlestick_patterns", [])
    for p in patterns[:2]:  # Max 2 pattern signals
        signals.append((f"Pattern: {p['pattern']}", p["signal"], p["description"]))

    # 12. Support/Resistance proximity
    sr = indicators.get("support_resistance", {})
    supports = sr.get("support", [])
    resistances = sr.get("resistance", [])
    if supports:
        nearest_support = supports[0]
        dist_pct = ((current_price - nearest_support) / current_price) * 100
        if dist_pct < 2:
            signals.append(("Support", "bullish", f"Price near support at {nearest_support:.2f} ({dist_pct:.1f}% away)"))
    if resistances:
        nearest_resistance = resistances[0]
        dist_pct = ((nearest_resistance - current_price) / current_price) * 100
        if dist_pct < 2:
            signals.append(("Resistance", "bearish", f"Price near resistance at {nearest_resistance:.2f} ({dist_pct:.1f}% away)"))

    # 13. Ichimoku Cloud
    ichimoku = indicators.get("ichimoku", {})
    if ichimoku.get("cloud_position"):
        pos = ichimoku["cloud_position"]
        tk = ichimoku.get("tk_cross", "neutral")
        if pos == "above_cloud" and tk == "bullish":
            signals.append(("Ichimoku", "bullish", "Price above cloud + Tenkan > Kijun — strong bullish"))
        elif pos == "above_cloud":
            signals.append(("Ichimoku", "bullish", "Price above Ichimoku cloud — bullish trend"))
        elif pos == "below_cloud" and tk == "bearish":
            signals.append(("Ichimoku", "bearish", "Price below cloud + Tenkan < Kijun — strong bearish"))
        elif pos == "below_cloud":
            signals.append(("Ichimoku", "bearish", "Price below Ichimoku cloud — bearish trend"))
        else:
            signals.append(("Ichimoku", "neutral", "Price inside Ichimoku cloud — consolidation zone"))

    # 14. Williams %R
    wr = indicators.get("williams_r")
    if wr is not None:
        if wr > -20:
            signals.append(("Williams %R", "bearish", f"Williams %R at {wr:.1f} — overbought (>-20)"))
        elif wr < -80:
            signals.append(("Williams %R", "bullish", f"Williams %R at {wr:.1f} — oversold (<-80)"))
        else:
            signals.append(("Williams %R", "neutral", f"Williams %R at {wr:.1f} — mid range"))

    # 15. CMF (Chaikin Money Flow)
    cmf = indicators.get("cmf")
    if cmf is not None:
        if cmf > 0.1:
            signals.append(("CMF", "bullish", f"CMF at {cmf:.3f} — strong buying pressure"))
        elif cmf > 0:
            signals.append(("CMF", "bullish", f"CMF at {cmf:.3f} — mild buying pressure"))
        elif cmf < -0.1:
            signals.append(("CMF", "bearish", f"CMF at {cmf:.3f} — strong selling pressure"))
        elif cmf < 0:
            signals.append(("CMF", "bearish", f"CMF at {cmf:.3f} — mild selling pressure"))
        else:
            signals.append(("CMF", "neutral", f"CMF at {cmf:.3f} — balanced"))

    # 16. Pivot Points
    pp = indicators.get("pivot_points", {})
    if pp.get("pp"):
        pivot = pp["pp"]
        if current_price > pp.get("r1", 9999999):
            signals.append(("Pivots", "bullish", f"Price above R1 ({pp['r1']}) — strong bullish momentum"))
        elif current_price > pivot:
            signals.append(("Pivots", "bullish", f"Price above pivot ({pivot}) — bullish bias"))
        elif current_price < pp.get("s1", 0):
            signals.append(("Pivots", "bearish", f"Price below S1 ({pp['s1']}) — strong bearish momentum"))
        elif current_price < pivot:
            signals.append(("Pivots", "bearish", f"Price below pivot ({pivot}) — bearish bias"))

    # 17. Fibonacci Signal
    fib = indicators.get("fibonacci", {})
    fib_618 = fib.get("level_618")
    fib_382 = fib.get("level_382")
    if fib_618 and fib_382:
        if current_price <= fib_618 * 1.01 and current_price >= fib_618 * 0.99:
            signals.append(("Fibonacci", "bullish", f"Price at 61.8% retracement (₹{fib_618:.0f}) — key support level"))
        elif current_price <= fib_382 * 1.01 and current_price >= fib_382 * 0.99:
            signals.append(("Fibonacci", "bearish", f"Price at 38.2% retracement (₹{fib_382:.0f}) — potential resistance"))
        elif fib.get("level_236") and current_price > fib.get("level_236"):
            signals.append(("Fibonacci", "bullish", f"Price above 23.6% fib — uptrend intact"))

    # ── Tally ──
    bullish = sum(1 for _, s, _ in signals if s == "bullish")
    bearish = sum(1 for _, s, _ in signals if s == "bearish")
    neutral = sum(1 for _, s, _ in signals if s == "neutral")
    total = len(signals)

    if total == 0:
        return {
            "verdict": "INSUFFICIENT DATA",
            "score": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "total_indicators": 0,
            "signals": [],
            "reasoning": "Not enough data to generate signals.",
        }

    score = ((bullish - bearish) / total) * 100  # -100 to +100

    if score > 30:
        verdict = "BULLISH"
    elif score > 10:
        verdict = "SLIGHTLY BULLISH"
    elif score < -30:
        verdict = "BEARISH"
    elif score < -10:
        verdict = "SLIGHTLY BEARISH"
    else:
        verdict = "NEUTRAL"

    return {
        "verdict": verdict,
        "score": round(score, 1),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "total_indicators": total,
        "signals": [{"indicator": name, "signal": sig, "reason": reason} for name, sig, reason in signals],
        "reasoning": f"{bullish} indicators bullish, {bearish} bearish, {neutral} neutral out of {total} total. Overall signal: {verdict}.",
    }


# ════════════════════════════════════════════════════════
#  PRICE PROJECTIONS (1mo, 3mo, 6mo)
# ════════════════════════════════════════════════════════

def calc_price_projections(df: pd.DataFrame, indicators: dict, signal: dict) -> dict:
    """
    Estimate price ranges for 1mo, 3mo, 6mo BASED ON TECHNICAL ANALYSIS:
    - Support/Resistance as boundaries
    - Fibonacci levels as targets
    - RSI mean-reversion pressure
    - MACD momentum direction
    - Bollinger Band squeeze/expansion
    - Moving average trends
    - Historical price action patterns
    NO hallucinations — only uses data that exists.
    """
    if len(df) < 30:
        return {"1m": None, "3m": None, "6m": None, "reasoning": "Insufficient data (need 30+ candles)."}

    current = float(df["Close"].iloc[-1])
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)

    # ── 1. PRICE ACTION: Historical monthly returns for realistic bounds ──
    if len(closes) >= 22:
        monthly_returns = []
        for i in range(22, len(closes), 22):
            mr = (closes[i] - closes[i-22]) / closes[i-22]
            monthly_returns.append(mr)
        if monthly_returns:
            avg_monthly_return = np.mean(monthly_returns)
            monthly_vol = np.std(monthly_returns)
        else:
            avg_monthly_return = 0
            monthly_vol = 0.05
    else:
        daily_returns = np.diff(closes) / closes[:-1]
        avg_monthly_return = np.mean(daily_returns) * 22
        monthly_vol = np.std(daily_returns) * np.sqrt(22)

    # ── 2. INDICATOR-BASED BIAS (each adds/subtracts from target) ──
    bias_factors = []  # List of (factor_name, monthly_adjustment_pct, explanation)

    # RSI: Mean-reversion pressure
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi > 75:
            bias_factors.append(("RSI", -0.03, f"RSI {rsi:.0f} — severely overbought, mean-reversion pull down likely"))
        elif rsi > 65:
            bias_factors.append(("RSI", -0.01, f"RSI {rsi:.0f} — overbought zone, upside limited"))
        elif rsi < 25:
            bias_factors.append(("RSI", 0.04, f"RSI {rsi:.0f} — severely oversold, bounce likely"))
        elif rsi < 35:
            bias_factors.append(("RSI", 0.02, f"RSI {rsi:.0f} — oversold, recovery potential"))
        else:
            bias_factors.append(("RSI", 0, f"RSI {rsi:.0f} — neutral zone"))

    # MACD: Momentum direction
    macd = indicators.get("macd", {})
    if macd.get("histogram") is not None:
        hist = macd["histogram"]
        if hist > 0 and macd.get("macd", 0) > 0:
            bias_factors.append(("MACD", 0.015, "MACD positive & above signal — bullish momentum"))
        elif hist < 0 and macd.get("macd", 0) < 0:
            bias_factors.append(("MACD", -0.015, "MACD negative & below signal — bearish momentum"))
        elif hist > 0:
            bias_factors.append(("MACD", 0.005, "MACD histogram turning positive — early bullish"))
        else:
            bias_factors.append(("MACD", -0.005, "MACD histogram turning negative — early bearish"))

    # Moving Averages: Trend structure
    sma = indicators.get("sma", {})
    sma_20 = sma.get("SMA_20")
    sma_50 = sma.get("SMA_50")
    sma_200 = sma.get("SMA_200")

    if sma_20 and sma_50 and sma_200:
        if current > sma_20 > sma_50 > sma_200:
            bias_factors.append(("MA Alignment", 0.02, "Perfect bullish alignment: Price > 20 > 50 > 200 SMA"))
        elif current < sma_20 < sma_50 < sma_200:
            bias_factors.append(("MA Alignment", -0.02, "Perfect bearish alignment: Price < 20 < 50 < 200 SMA"))
        elif current > sma_200:
            bias_factors.append(("MA Alignment", 0.01, "Above 200 SMA — long-term uptrend intact"))
        else:
            bias_factors.append(("MA Alignment", -0.01, "Below 200 SMA — long-term downtrend"))
    elif sma_20 and sma_50:
        if current > sma_20 > sma_50:
            bias_factors.append(("MA Alignment", 0.01, "Price above 20 & 50 SMA — short-term bullish"))
        elif current < sma_20 < sma_50:
            bias_factors.append(("MA Alignment", -0.01, "Price below 20 & 50 SMA — short-term bearish"))

    # Supertrend: Trend direction
    st = indicators.get("supertrend", {})
    if st.get("direction") == "bullish":
        bias_factors.append(("Supertrend", 0.01, "Supertrend bullish — trend support below price"))
    elif st.get("direction") == "bearish":
        bias_factors.append(("Supertrend", -0.01, "Supertrend bearish — trend resistance above price"))

    # Bollinger Bands: Squeeze/expansion
    bb = indicators.get("bollinger", {})
    if bb.get("bandwidth") is not None:
        bw = bb["bandwidth"]
        pct_b = bb.get("percent_b", 0.5)
        if bw < 5:
            bias_factors.append(("Bollinger", 0, f"BB squeeze (bandwidth {bw:.1f}%) — big move expected, direction unclear"))
        elif pct_b > 0.9:
            bias_factors.append(("Bollinger", -0.01, "Price near upper BB — overbought pressure"))
        elif pct_b < 0.1:
            bias_factors.append(("Bollinger", 0.01, "Price near lower BB — oversold bounce possible"))

    # ── 3. SUPPORT/RESISTANCE as hard boundaries ──
    sr = indicators.get("support_resistance", {})
    supports = sr.get("support", [])
    resistances = sr.get("resistance", [])

    # Fibonacci as price targets
    fib = indicators.get("fibonacci", {})
    fib_levels = []
    for key in ["level_236", "level_382", "level_500", "level_618"]:
        val = fib.get(key)
        if val is not None:
            fib_levels.append(val)

    # ── 4. CALCULATE PROJECTIONS ──
    total_monthly_bias = sum(f[1] for f in bias_factors)
    projections = {}
    horizons = {"1m": 1, "3m": 3, "6m": 6}

    for label, months in horizons.items():
        # Base: historical average return * months + indicator bias * months
        base_move_pct = (avg_monthly_return * months) + (total_monthly_bias * months)
        vol_range_pct = monthly_vol * np.sqrt(months)

        target = current * (1 + base_move_pct)
        proj_high = current * (1 + base_move_pct + vol_range_pct)
        proj_low = current * (1 + base_move_pct - vol_range_pct)

        # Clamp using support/resistance
        if resistances and proj_high > resistances[0] * 1.05:
            proj_high = min(proj_high, resistances[0] * 1.03)  # Resistance acts as ceiling
        if supports and proj_low < supports[0] * 0.95:
            proj_low = max(proj_low, supports[0] * 0.97)  # Support acts as floor

        # Use Fibonacci levels as target refinement
        if fib_levels:
            fib_above = [f for f in fib_levels if f > current]
            fib_below = [f for f in fib_levels if f < current]
            if base_move_pct > 0 and fib_above:
                nearest_fib_target = min(fib_above)
                target = (target + nearest_fib_target) / 2  # Blend with fib
            elif base_move_pct < 0 and fib_below:
                nearest_fib_target = max(fib_below)
                target = (target + nearest_fib_target) / 2

        proj_low = max(proj_low, current * 0.5)  # Sanity floor
        proj_high = min(proj_high, current * 2.0)  # Sanity ceiling

        change_pct = ((target - current) / current) * 100

        # Confidence from indicator agreement
        total_ind = signal.get("total_indicators", 1)
        bull_pct = signal.get("bullish_count", 0) / max(total_ind, 1)
        bear_pct = signal.get("bearish_count", 0) / max(total_ind, 1)
        agreement = max(bull_pct, bear_pct)
        confidence = "high" if agreement > 0.65 else "medium" if agreement > 0.45 else "low"

        if change_pct > 2:
            direction = "bullish"
        elif change_pct < -2:
            direction = "bearish"
        else:
            direction = "neutral"

        projections[label] = {
            "low": round(proj_low, 2),
            "target": round(target, 2),
            "high": round(proj_high, 2),
            "direction": direction,
            "confidence": confidence,
            "change_pct": round(change_pct, 1),
        }

    # ── 5. BUILD REASONING from actual indicators ──
    reasoning = [f[2] for f in bias_factors if f[1] != 0]  # Only non-neutral factors
    if supports:
        reasoning.append(f"Key support at ₹{supports[0]:.0f}")
    if resistances:
        reasoning.append(f"Key resistance at ₹{resistances[0]:.0f}")
    reasoning.append(f"Historical monthly volatility: {monthly_vol*100:.1f}%")
    reasoning.append(f"Avg monthly return (from price action): {avg_monthly_return*100:.1f}%")

    projections["current"] = round(current, 2)
    projections["reasoning"] = reasoning
    projections["disclaimer"] = "Based on technical indicators, price action, and historical patterns. Not financial advice. Markets can move against projections."

    return projections


# ════════════════════════════════════════════════════════
#  MASTER ANALYSIS FUNCTION
# ════════════════════════════════════════════════════════

def run_full_analysis(df: pd.DataFrame, interval: str = "1d") -> dict:
    """Run all technical indicators and generate complete analysis."""
    if df is None or df.empty or len(df) < 5:
        return {"error": "Insufficient data for analysis"}

    current_price = df["Close"].iloc[-1]

    # Calculate all indicators
    sma = calc_sma(df)
    ema = calc_ema(df)
    rsi = calc_rsi(df)
    macd = calc_macd(df)
    bollinger = calc_bollinger(df)
    stochastic = calc_stochastic(df)
    atr = calc_atr(df)
    adx = calc_adx(df)
    supertrend = calc_supertrend(df)
    vwap = calc_vwap(df)
    obv = calc_obv(df)
    fibonacci = calc_fibonacci_levels(df)
    support_resistance = calc_support_resistance(df)
    candlestick_patterns = detect_candlestick_patterns(df)
    elliott_wave = detect_elliott_wave_basic(df)
    ichimoku = calc_ichimoku(df)
    williams_r = calc_williams_r(df)
    cmf = calc_cmf(df)
    pivot_points = calc_pivot_points(df)

    indicators = {
        "sma": sma,
        "ema": ema,
        "rsi": rsi,
        "macd": macd,
        "bollinger": bollinger,
        "stochastic": stochastic,
        "atr": atr,
        "adx": adx,
        "supertrend": supertrend,
        "vwap": vwap,
        "obv": obv,
        "fibonacci": fibonacci,
        "support_resistance": support_resistance,
        "candlestick_patterns": candlestick_patterns,
        "elliott_wave": elliott_wave,
        "ichimoku": ichimoku,
        "williams_r": williams_r,
        "cmf": cmf,
        "pivot_points": pivot_points,
    }

    # Generate signal
    signal = generate_signal(indicators, current_price)

    # Calculate projected range
    projected = calc_projected_range(df, indicators)

    # Detect chart patterns
    from pattern_detection import detect_all_patterns
    chart_patterns = detect_all_patterns(df, min_confidence=0, interval=interval)

    # Prepare chart data (OHLCV + indicator overlays as JSON-serializable)
    # LIMIT data points: intraday can have thousands — cap at 500 for chart performance
    MAX_CHART_POINTS = 500
    step = max(1, len(df) // MAX_CHART_POINTS)
    df_chart = df.iloc[::step] if len(df) > MAX_CHART_POINTS else df

    chart_data = []
    for idx, row in df_chart.iterrows():
        # Format time string — yfinance returns UTC-aware for intraday, naive for daily
        try:
            if hasattr(idx, 'tzinfo') and idx.tzinfo is not None:
                # Convert to IST (UTC+5:30) 
                try:
                    ist_time = idx.tz_convert('Asia/Kolkata')
                    time_str = ist_time.strftime('%Y-%m-%dT%H:%M:%S')
                except Exception:
                    # Fallback: manual UTC+5:30
                    from datetime import timezone, timedelta
                    ist = timezone(timedelta(hours=5, minutes=30))
                    ist_time = idx.astimezone(ist)
                    time_str = ist_time.strftime('%Y-%m-%dT%H:%M:%S')
            else:
                # Daily data — no timezone, just use as-is
                time_str = idx.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(idx, 'strftime') else str(idx)
        except Exception:
            time_str = str(idx)
        
        point = {
            "time": time_str,
            "open": round(row["Open"], 2) if pd.notna(row["Open"]) else None,
            "high": round(row["High"], 2) if pd.notna(row["High"]) else None,
            "low": round(row["Low"], 2) if pd.notna(row["Low"]) else None,
            "close": round(row["Close"], 2) if pd.notna(row["Close"]) else None,
            "volume": int(row["Volume"]) if pd.notna(row.get("Volume", None)) else 0,
        }
        # Add indicator values to each data point
        for col in ["SMA_20", "SMA_50", "SMA_200", "EMA_9", "EMA_21", "EMA_50",
                     "RSI", "MACD", "MACD_Signal", "MACD_Hist",
                     "BB_Upper", "BB_Middle", "BB_Lower",
                     "Stoch_K", "Stoch_D", "ATR", "ADX",
                     "Supertrend", "VWAP", "OBV"]:
            if col in df_chart.columns:
                val = row.get(col)
                point[col.lower()] = round(float(val), 4) if pd.notna(val) else None

        chart_data.append(point)

    # Round indicator values for display
    def safe_round(val, decimals=2):
        if val is None:
            return None
        try:
            return round(float(val), decimals)
        except (TypeError, ValueError):
            return val

    return {
        "current_price": safe_round(current_price),
        "indicators": {
            "sma": {k: safe_round(v) for k, v in sma.items()},
            "ema": {k: safe_round(v) for k, v in ema.items()},
            "rsi": safe_round(rsi, 1),
            "macd": {k: safe_round(v, 4) for k, v in macd.items()},
            "bollinger": {k: safe_round(v) for k, v in bollinger.items()},
            "stochastic": {k: safe_round(v, 1) for k, v in stochastic.items()},
            "atr": safe_round(atr),
            "adx": safe_round(adx, 1),
            "supertrend": supertrend,
            "vwap": safe_round(vwap),
            "obv": safe_round(obv, 0),
            "fibonacci": {k: safe_round(v) for k, v in fibonacci.items()},
            "support_resistance": support_resistance,
            "candlestick_patterns": candlestick_patterns,
            "elliott_wave": elliott_wave,
            "ichimoku": ichimoku,
            "williams_r": safe_round(williams_r, 2),
            "cmf": safe_round(cmf, 4),
            "pivot_points": pivot_points,
        },
        "signal": signal,
        "projected_range": projected,
        "price_projections": calc_price_projections(df, indicators, signal),
        "chart_patterns": chart_patterns,
        "chart_data": chart_data,
        "data_points": len(chart_data),
    }
