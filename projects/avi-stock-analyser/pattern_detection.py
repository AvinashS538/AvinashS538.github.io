"""
Chart Pattern Detection Engine
===============================
Detects classical chart patterns from OHLCV data with confidence scoring.
Each pattern includes: name, type (bullish/bearish), confidence %, pivot points,
trendlines for chart overlay, and one-liner outcome suggestion.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
#  SWING POINT DETECTION
# ════════════════════════════════════════════════════════

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Find local swing highs and swing lows.
    A swing high = higher than `lookback` bars on each side.
    A swing low = lower than `lookback` bars on each side.
    """
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    
    # Convert times to IST-matching format (same as chart data)
    times = []
    for idx in df.index:
        try:
            if hasattr(idx, 'tzinfo') and idx.tzinfo is not None:
                try:
                    ist = idx.tz_convert('Asia/Kolkata')
                    times.append(ist.strftime('%Y-%m-%dT%H:%M:%S'))
                except Exception:
                    from datetime import timezone, timedelta as td
                    ist_tz = timezone(td(hours=5, minutes=30))
                    times.append(idx.astimezone(ist_tz).strftime('%Y-%m-%dT%H:%M:%S'))
            else:
                times.append(idx.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(idx, 'strftime') else str(idx))
        except Exception:
            times.append(str(idx))

    swing_highs = []  # (index, time, price)
    swing_lows = []

    for i in range(lookback, len(highs) - lookback):
        # Swing high: higher than all neighbors
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            swing_highs.append({"idx": i, "time": times[i], "price": float(highs[i])})
        # Swing low: lower than all neighbors
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            swing_lows.append({"idx": i, "time": times[i], "price": float(lows[i])})

    return {
        "highs": swing_highs,
        "lows": swing_lows,
        "closes": closes,
        "times": times,
        "all_highs": highs,
        "all_lows": lows,
    }


def _price_similar(p1: float, p2: float, tolerance: float = 0.02) -> bool:
    """Check if two prices are within tolerance % of each other."""
    if p1 == 0:
        return False
    return abs(p1 - p2) / p1 <= tolerance


def _slope_angle(p1: float, p2: float, bars: int) -> float:
    """Calculate slope as % per bar."""
    if bars == 0:
        return 0
    return (p2 - p1) / p1 / bars


# ════════════════════════════════════════════════════════
#  INDIVIDUAL PATTERN DETECTORS
# ════════════════════════════════════════════════════════

def detect_double_top(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Two highs at similar level → bearish reversal."""
    highs = sp["highs"]
    if len(highs) < 2:
        return None

    # Check last 2 swing highs
    h1, h2 = highs[-2], highs[-1]
    if h2["idx"] - h1["idx"] < 8:  # Need some distance
        return None

    # Prices should be similar (within 2%)
    if not _price_similar(h1["price"], h2["price"], 0.025):
        return None

    # There should be a dip between them
    between_lows = [l for l in sp["lows"] if h1["idx"] < l["idx"] < h2["idx"]]
    if not between_lows:
        return None

    neckline = min(l["price"] for l in between_lows)
    current = float(closes[-1])
    top_level = (h1["price"] + h2["price"]) / 2

    # Price should be near or below the neckline area
    if current > top_level * 1.02:  # Still making new highs, not a double top
        return None

    # Confidence: how similar are the tops + how clear is the neckline
    price_similarity = 1 - abs(h1["price"] - h2["price"]) / h1["price"]
    dip_depth = (top_level - neckline) / top_level
    conf = min(price_similarity * 70 + dip_depth * 200, 95)

    target = neckline - (top_level - neckline)  # Measured move

    return {
        "name": "Double Top",
        "type": "bearish",
        "confidence": round(conf),
        "points": [h1, between_lows[0], h2],
        "lines": [
            {"x": [h1["time"], h2["time"]], "y": [h1["price"], h2["price"]], "color": "rgba(255,92,92,0.7)", "dash": "solid", "label": "Resistance"},
            {"x": [between_lows[0]["time"], sp["times"][-1]], "y": [neckline, neckline], "color": "rgba(255,92,92,0.4)", "dash": "dash", "label": "Neckline"},
        ],
        "annotation": {"x": h2["time"], "y": h2["price"], "text": "Double Top"},
        "outcome": f"Bearish reversal. Breakdown below neckline ₹{neckline:.0f} targets ₹{target:.0f}. Two failed attempts to break resistance confirms sellers in control.",
        "key_level": round(neckline, 2),
        "target": round(target, 2),
    }


def detect_double_bottom(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Two lows at similar level → bullish reversal."""
    lows = sp["lows"]
    if len(lows) < 2:
        return None

    l1, l2 = lows[-2], lows[-1]
    if l2["idx"] - l1["idx"] < 8:
        return None

    if not _price_similar(l1["price"], l2["price"], 0.025):
        return None

    between_highs = [h for h in sp["highs"] if l1["idx"] < h["idx"] < l2["idx"]]
    if not between_highs:
        return None

    neckline = max(h["price"] for h in between_highs)
    current = float(closes[-1])
    bottom_level = (l1["price"] + l2["price"]) / 2

    if current < bottom_level * 0.98:
        return None

    price_similarity = 1 - abs(l1["price"] - l2["price"]) / l1["price"]
    dip_depth = (neckline - bottom_level) / bottom_level
    conf = min(price_similarity * 70 + dip_depth * 200, 95)

    target = neckline + (neckline - bottom_level)

    return {
        "name": "Double Bottom",
        "type": "bullish",
        "confidence": round(conf),
        "points": [l1, between_highs[0], l2],
        "lines": [
            {"x": [l1["time"], l2["time"]], "y": [l1["price"], l2["price"]], "color": "rgba(0,214,143,0.7)", "dash": "solid", "label": "Support"},
            {"x": [between_highs[0]["time"], sp["times"][-1]], "y": [neckline, neckline], "color": "rgba(0,214,143,0.4)", "dash": "dash", "label": "Neckline"},
        ],
        "annotation": {"x": l2["time"], "y": l2["price"], "text": "Double Bottom"},
        "outcome": f"Bullish reversal. Breakout above neckline ₹{neckline:.0f} targets ₹{target:.0f}. Two bounces from support shows strong buyer interest.",
        "key_level": round(neckline, 2),
        "target": round(target, 2),
    }


def detect_head_shoulders(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Head & Shoulders (bearish) — three peaks, middle highest."""
    highs = sp["highs"]
    if len(highs) < 3:
        return None

    ls, head, rs = highs[-3], highs[-2], highs[-1]

    # Head must be highest
    if head["price"] <= ls["price"] or head["price"] <= rs["price"]:
        return None

    # Shoulders should be roughly similar height (within 5%)
    if not _price_similar(ls["price"], rs["price"], 0.05):
        return None

    # Head should be clearly higher than shoulders (at least 2%)
    if head["price"] < ls["price"] * 1.02:
        return None

    # Find neckline from lows between shoulders
    lows_between = [l for l in sp["lows"] if ls["idx"] < l["idx"] < rs["idx"]]
    if len(lows_between) < 1:
        return None

    neckline = np.mean([l["price"] for l in lows_between])
    current = float(closes[-1])

    shoulder_similarity = 1 - abs(ls["price"] - rs["price"]) / ls["price"]
    head_prominence = (head["price"] - max(ls["price"], rs["price"])) / head["price"]
    conf = min(shoulder_similarity * 50 + head_prominence * 300, 95)

    target = neckline - (head["price"] - neckline)

    nl_start = lows_between[0]
    nl_end_time = sp["times"][min(rs["idx"] + 10, len(sp["times"]) - 1)]

    return {
        "name": "Head & Shoulders",
        "type": "bearish",
        "confidence": round(conf),
        "points": [ls, head, rs],
        "lines": [
            {"x": [ls["time"], head["time"], rs["time"]], "y": [ls["price"], head["price"], rs["price"]], "color": "rgba(255,92,92,0.6)", "dash": "solid", "label": "Pattern"},
            {"x": [nl_start["time"], nl_end_time], "y": [neckline, neckline], "color": "rgba(255,92,92,0.4)", "dash": "dash", "label": "Neckline"},
        ],
        "annotation": {"x": head["time"], "y": head["price"], "text": "H&S"},
        "outcome": f"Classic bearish reversal. Break below neckline ₹{neckline:.0f} targets ₹{target:.0f}. Head is the final push before sellers take control.",
        "key_level": round(neckline, 2),
        "target": round(target, 2),
    }


def detect_inv_head_shoulders(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Inverse Head & Shoulders (bullish) — three troughs, middle lowest."""
    lows = sp["lows"]
    if len(lows) < 3:
        return None

    ls, head, rs = lows[-3], lows[-2], lows[-1]

    if head["price"] >= ls["price"] or head["price"] >= rs["price"]:
        return None
    if not _price_similar(ls["price"], rs["price"], 0.05):
        return None
    if head["price"] > ls["price"] * 0.98:
        return None

    highs_between = [h for h in sp["highs"] if ls["idx"] < h["idx"] < rs["idx"]]
    if not highs_between:
        return None

    neckline = np.mean([h["price"] for h in highs_between])
    current = float(closes[-1])

    shoulder_similarity = 1 - abs(ls["price"] - rs["price"]) / ls["price"]
    head_depth = (min(ls["price"], rs["price"]) - head["price"]) / head["price"]
    conf = min(shoulder_similarity * 50 + head_depth * 300, 95)

    target = neckline + (neckline - head["price"])

    return {
        "name": "Inv Head & Shoulders",
        "type": "bullish",
        "confidence": round(conf),
        "points": [ls, head, rs],
        "lines": [
            {"x": [ls["time"], head["time"], rs["time"]], "y": [ls["price"], head["price"], rs["price"]], "color": "rgba(0,214,143,0.6)", "dash": "solid", "label": "Pattern"},
            {"x": [highs_between[0]["time"], sp["times"][-1]], "y": [neckline, neckline], "color": "rgba(0,214,143,0.4)", "dash": "dash", "label": "Neckline"},
        ],
        "annotation": {"x": head["time"], "y": head["price"], "text": "Inv H&S"},
        "outcome": f"Bullish reversal. Breakout above neckline ₹{neckline:.0f} targets ₹{target:.0f}. Deep head shows sellers exhausted, buyers stepping in.",
        "key_level": round(neckline, 2),
        "target": round(target, 2),
    }


def detect_ascending_triangle(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Flat resistance + rising support → bullish breakout."""
    highs = sp["highs"]
    lows = sp["lows"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
    recent_lows = lows[-3:] if len(lows) >= 3 else lows[-2:]

    # Resistance should be flat (highs at similar levels)
    h_prices = [h["price"] for h in recent_highs]
    h_range = (max(h_prices) - min(h_prices)) / np.mean(h_prices)
    if h_range > 0.03:  # More than 3% variation = not flat
        return None

    # Support should be rising
    l_prices = [l["price"] for l in recent_lows]
    if len(l_prices) >= 2 and l_prices[-1] <= l_prices[0]:
        return None  # Not rising

    resistance = np.mean(h_prices)
    support_slope = (l_prices[-1] - l_prices[0]) / max(len(l_prices) - 1, 1)

    if support_slope <= 0:
        return None

    conf = min((1 - h_range / 0.03) * 40 + (support_slope / l_prices[0] * 100) * 40, 90)
    target = resistance + (resistance - l_prices[-1])

    return {
        "name": "Ascending Triangle",
        "type": "bullish",
        "confidence": round(conf),
        "points": recent_highs + recent_lows,
        "lines": [
            {"x": [recent_highs[0]["time"], recent_highs[-1]["time"]], "y": [resistance, resistance], "color": "rgba(91,140,255,0.7)", "dash": "solid", "label": "Resistance"},
            {"x": [recent_lows[0]["time"], recent_lows[-1]["time"]], "y": [recent_lows[0]["price"], recent_lows[-1]["price"]], "color": "rgba(0,214,143,0.7)", "dash": "solid", "label": "Rising Support"},
        ],
        "annotation": {"x": recent_highs[-1]["time"], "y": resistance, "text": "Asc △"},
        "outcome": f"Bullish continuation. Breakout above ₹{resistance:.0f} targets ₹{target:.0f}. Rising lows show increasing buyer aggression against flat resistance.",
        "key_level": round(resistance, 2),
        "target": round(target, 2),
    }


def detect_descending_triangle(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Flat support + falling resistance → bearish breakdown."""
    highs = sp["highs"]
    lows = sp["lows"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
    recent_lows = lows[-3:] if len(lows) >= 3 else lows[-2:]

    l_prices = [l["price"] for l in recent_lows]
    l_range = (max(l_prices) - min(l_prices)) / np.mean(l_prices)
    if l_range > 0.03:
        return None

    h_prices = [h["price"] for h in recent_highs]
    if len(h_prices) >= 2 and h_prices[-1] >= h_prices[0]:
        return None

    support = np.mean(l_prices)
    target = support - (h_prices[0] - support)

    conf = min((1 - l_range / 0.03) * 40 + 30, 85)

    return {
        "name": "Descending Triangle",
        "type": "bearish",
        "confidence": round(conf),
        "points": recent_highs + recent_lows,
        "lines": [
            {"x": [recent_lows[0]["time"], recent_lows[-1]["time"]], "y": [support, support], "color": "rgba(91,140,255,0.7)", "dash": "solid", "label": "Support"},
            {"x": [recent_highs[0]["time"], recent_highs[-1]["time"]], "y": [recent_highs[0]["price"], recent_highs[-1]["price"]], "color": "rgba(255,92,92,0.7)", "dash": "solid", "label": "Falling Resistance"},
        ],
        "annotation": {"x": recent_lows[-1]["time"], "y": support, "text": "Desc △"},
        "outcome": f"Bearish continuation. Breakdown below ₹{support:.0f} targets ₹{max(target,0):.0f}. Falling highs show sellers getting more aggressive.",
        "key_level": round(support, 2),
        "target": round(max(target, 0), 2),
    }


def detect_symmetrical_triangle(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Converging trendlines — breakout direction unknown."""
    highs = sp["highs"]
    lows = sp["lows"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    rh = highs[-3:] if len(highs) >= 3 else highs[-2:]
    rl = lows[-3:] if len(lows) >= 3 else lows[-2:]

    h_prices = [h["price"] for h in rh]
    l_prices = [l["price"] for l in rl]

    # Highs should be falling
    if h_prices[-1] >= h_prices[0]:
        return None
    # Lows should be rising
    if l_prices[-1] <= l_prices[0]:
        return None

    # They should be converging
    spread_start = h_prices[0] - l_prices[0]
    spread_end = h_prices[-1] - l_prices[-1]
    if spread_end >= spread_start:
        return None

    convergence = (spread_start - spread_end) / spread_start
    conf = min(convergence * 100 + 30, 85)

    current = float(closes[-1])
    breakout_range = spread_end / 2

    return {
        "name": "Symmetrical Triangle",
        "type": "neutral",
        "confidence": round(conf),
        "points": rh + rl,
        "lines": [
            {"x": [rh[0]["time"], rh[-1]["time"]], "y": [rh[0]["price"], rh[-1]["price"]], "color": "rgba(255,184,77,0.7)", "dash": "solid", "label": "Falling Resistance"},
            {"x": [rl[0]["time"], rl[-1]["time"]], "y": [rl[0]["price"], rl[-1]["price"]], "color": "rgba(255,184,77,0.7)", "dash": "solid", "label": "Rising Support"},
        ],
        "annotation": {"x": rh[-1]["time"], "y": (h_prices[-1] + l_prices[-1]) / 2, "text": "Sym △"},
        "outcome": f"Coiling pattern — big move incoming. Breakout above ₹{h_prices[-1]:.0f} = bullish to ₹{current + spread_start:.0f}. Breakdown below ₹{l_prices[-1]:.0f} = bearish to ₹{current - spread_start:.0f}. Wait for direction confirmation.",
        "key_level": round((h_prices[-1] + l_prices[-1]) / 2, 2),
        "target": None,
    }


def detect_cup_and_handle(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """U-shaped recovery + small pullback → bullish continuation."""
    lows = sp["lows"]
    highs = sp["highs"]
    if len(lows) < 3 or len(highs) < 2:
        return None

    # Need a U-shape: price drops, forms a rounded bottom, recovers near the start
    # Then a small pullback (handle)

    # Find the deepest low in recent data
    recent_lows = lows[-5:] if len(lows) >= 5 else lows
    cup_bottom = min(recent_lows, key=lambda l: l["price"])

    # Highs before and after the cup bottom
    pre_cup = [h for h in highs if h["idx"] < cup_bottom["idx"]]
    post_cup = [h for h in highs if h["idx"] > cup_bottom["idx"]]

    if not pre_cup or not post_cup:
        return None

    cup_left = pre_cup[-1]
    cup_right = post_cup[-1] if post_cup else None

    if cup_right is None:
        return None

    # Cup rim should be at similar levels (within 3%)
    if not _price_similar(cup_left["price"], cup_right["price"], 0.04):
        return None

    # Cup should have meaningful depth (at least 5%)
    rim_level = (cup_left["price"] + cup_right["price"]) / 2
    depth = (rim_level - cup_bottom["price"]) / rim_level
    if depth < 0.05:
        return None

    # Handle: small pullback after right rim (last low should be above cup bottom)
    handle_lows = [l for l in lows if l["idx"] > cup_right["idx"]]
    has_handle = len(handle_lows) > 0 and handle_lows[-1]["price"] > cup_bottom["price"]

    conf = min(depth * 300 + (30 if has_handle else 0), 90)
    target = rim_level + (rim_level - cup_bottom["price"])

    return {
        "name": "Cup & Handle" if has_handle else "Cup Formation",
        "type": "bullish",
        "confidence": round(conf),
        "points": [cup_left, cup_bottom, cup_right],
        "lines": [
            {"x": [cup_left["time"], cup_right["time"]], "y": [rim_level, rim_level], "color": "rgba(0,214,143,0.5)", "dash": "dash", "label": "Rim"},
        ],
        "annotation": {"x": cup_bottom["time"], "y": cup_bottom["price"], "text": "Cup" + (" & Handle" if has_handle else "")},
        "outcome": f"Bullish continuation. Breakout above rim ₹{rim_level:.0f} targets ₹{target:.0f}. Cup depth {depth*100:.0f}% shows strong accumulation before next leg up.",
        "key_level": round(rim_level, 2),
        "target": round(target, 2),
    }


def detect_rising_wedge(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Both trendlines rising + converging → bearish."""
    highs = sp["highs"]
    lows = sp["lows"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    rh = highs[-3:] if len(highs) >= 3 else highs[-2:]
    rl = lows[-3:] if len(lows) >= 3 else lows[-2:]

    h_prices = [h["price"] for h in rh]
    l_prices = [l["price"] for l in rl]

    # Both rising
    if h_prices[-1] <= h_prices[0] or l_prices[-1] <= l_prices[0]:
        return None

    # Converging (support rising faster or spread narrowing)
    spread_start = h_prices[0] - l_prices[0]
    spread_end = h_prices[-1] - l_prices[-1]
    if spread_start <= 0 or spread_end >= spread_start:
        return None

    conf = min(((spread_start - spread_end) / spread_start) * 100 + 30, 85)

    return {
        "name": "Rising Wedge",
        "type": "bearish",
        "confidence": round(conf),
        "points": rh + rl,
        "lines": [
            {"x": [rh[0]["time"], rh[-1]["time"]], "y": [rh[0]["price"], rh[-1]["price"]], "color": "rgba(255,92,92,0.7)", "dash": "solid", "label": "Upper"},
            {"x": [rl[0]["time"], rl[-1]["time"]], "y": [rl[0]["price"], rl[-1]["price"]], "color": "rgba(255,92,92,0.5)", "dash": "solid", "label": "Lower"},
        ],
        "annotation": {"x": rh[-1]["time"], "y": rh[-1]["price"], "text": "Rising Wedge"},
        "outcome": f"Bearish reversal pattern. Rising wedges break down 65-70% of the time. Breakdown below ₹{l_prices[-1]:.0f} confirms. Momentum weakening despite higher prices.",
        "key_level": round(l_prices[-1], 2),
        "target": round(l_prices[0], 2),
    }


def detect_falling_wedge(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Both trendlines falling + converging → bullish."""
    highs = sp["highs"]
    lows = sp["lows"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    rh = highs[-3:] if len(highs) >= 3 else highs[-2:]
    rl = lows[-3:] if len(lows) >= 3 else lows[-2:]

    h_prices = [h["price"] for h in rh]
    l_prices = [l["price"] for l in rl]

    if h_prices[-1] >= h_prices[0] or l_prices[-1] >= l_prices[0]:
        return None

    spread_start = h_prices[0] - l_prices[0]
    spread_end = h_prices[-1] - l_prices[-1]
    if spread_start <= 0 or spread_end >= spread_start:
        return None

    conf = min(((spread_start - spread_end) / spread_start) * 100 + 30, 85)

    return {
        "name": "Falling Wedge",
        "type": "bullish",
        "confidence": round(conf),
        "points": rh + rl,
        "lines": [
            {"x": [rh[0]["time"], rh[-1]["time"]], "y": [rh[0]["price"], rh[-1]["price"]], "color": "rgba(0,214,143,0.5)", "dash": "solid", "label": "Upper"},
            {"x": [rl[0]["time"], rl[-1]["time"]], "y": [rl[0]["price"], rl[-1]["price"]], "color": "rgba(0,214,143,0.7)", "dash": "solid", "label": "Lower"},
        ],
        "annotation": {"x": rl[-1]["time"], "y": rl[-1]["price"], "text": "Falling Wedge"},
        "outcome": f"Bullish reversal. Falling wedges break up 65-70% of the time. Breakout above ₹{h_prices[-1]:.0f} confirms. Selling pressure diminishing.",
        "key_level": round(h_prices[-1], 2),
        "target": round(h_prices[0], 2),
    }


def detect_bull_flag(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Sharp move up + small pullback channel → bullish continuation."""
    if len(closes) < 20:
        return None

    # Look for a strong move (pole) followed by a slight pullback (flag)
    recent = closes[-20:]
    mid = len(recent) // 2

    # Pole: first half should have strong upward move (>5%)
    pole_return = (recent[mid] - recent[0]) / recent[0]
    if pole_return < 0.05:
        return None

    # Flag: second half should be slight pullback or consolidation (<3% drop)
    flag_return = (recent[-1] - recent[mid]) / recent[mid]
    if flag_return > 0 or flag_return < -0.05:
        return None

    conf = min(pole_return * 500 + abs(flag_return) * 200, 85)
    target = recent[-1] + (recent[mid] - recent[0])

    t = sp["times"]
    pole_start_t = t[max(0, len(t) - 20)]
    pole_end_t = t[max(0, len(t) - 10)]
    flag_end_t = t[-1]

    return {
        "name": "Bull Flag",
        "type": "bullish",
        "confidence": round(conf),
        "points": [],
        "lines": [
            {"x": [pole_start_t, pole_end_t], "y": [float(recent[0]), float(recent[mid])], "color": "rgba(0,214,143,0.6)", "dash": "solid", "label": "Pole"},
            {"x": [pole_end_t, flag_end_t], "y": [float(recent[mid]), float(recent[-1])], "color": "rgba(0,214,143,0.4)", "dash": "dot", "label": "Flag"},
        ],
        "annotation": {"x": flag_end_t, "y": float(recent[-1]), "text": "Bull Flag"},
        "outcome": f"Bullish continuation. Strong rally ({pole_return*100:.0f}%) followed by healthy pullback ({flag_return*100:.1f}%). Expected breakout target ₹{target:.0f}. Volume should increase on breakout.",
        "key_level": round(float(recent[mid]), 2),
        "target": round(target, 2),
    }


def detect_bear_flag(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Sharp move down + small bounce channel → bearish continuation."""
    if len(closes) < 20:
        return None

    recent = closes[-20:]
    mid = len(recent) // 2

    pole_return = (recent[mid] - recent[0]) / recent[0]
    if pole_return > -0.05:
        return None

    flag_return = (recent[-1] - recent[mid]) / recent[mid]
    if flag_return < 0 or flag_return > 0.05:
        return None

    conf = min(abs(pole_return) * 500 + flag_return * 200, 85)
    target = recent[-1] - (recent[0] - recent[mid])

    t = sp["times"]
    pole_start_t = t[max(0, len(t) - 20)]
    pole_end_t = t[max(0, len(t) - 10)]

    return {
        "name": "Bear Flag",
        "type": "bearish",
        "confidence": round(conf),
        "points": [],
        "lines": [
            {"x": [pole_start_t, pole_end_t], "y": [float(recent[0]), float(recent[mid])], "color": "rgba(255,92,92,0.6)", "dash": "solid", "label": "Pole"},
            {"x": [pole_end_t, t[-1]], "y": [float(recent[mid]), float(recent[-1])], "color": "rgba(255,92,92,0.4)", "dash": "dot", "label": "Flag"},
        ],
        "annotation": {"x": t[-1], "y": float(recent[-1]), "text": "Bear Flag"},
        "outcome": f"Bearish continuation. Sharp drop ({pole_return*100:.0f}%) followed by weak bounce ({flag_return*100:.1f}%). Expected breakdown target ₹{max(target,0):.0f}.",
        "key_level": round(float(recent[mid]), 2),
        "target": round(max(target, 0), 2),
    }


def detect_channel(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Parallel trendlines — ascending or descending channel."""
    highs = sp["highs"]
    lows = sp["lows"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    rh = highs[-4:] if len(highs) >= 4 else highs[-2:]
    rl = lows[-4:] if len(lows) >= 4 else lows[-2:]

    if len(rh) < 2 or len(rl) < 2:
        return None

    # Calculate slopes
    h_slope = (rh[-1]["price"] - rh[0]["price"]) / max(rh[-1]["idx"] - rh[0]["idx"], 1)
    l_slope = (rl[-1]["price"] - rl[0]["price"]) / max(rl[-1]["idx"] - rl[0]["idx"], 1)

    # Slopes should be similar direction and magnitude (parallel)
    if h_slope == 0 and l_slope == 0:
        return None

    # Check parallelism: slopes should be within 50% of each other
    avg_slope = (h_slope + l_slope) / 2
    if avg_slope == 0:
        return None
    slope_diff = abs(h_slope - l_slope) / abs(avg_slope) if avg_slope != 0 else 999
    if slope_diff > 0.6:
        return None

    is_ascending = avg_slope > 0
    name = "Ascending Channel" if is_ascending else "Descending Channel"
    ptype = "bullish" if is_ascending else "bearish"
    color = "rgba(0,214,143,0.5)" if is_ascending else "rgba(255,92,92,0.5)"

    conf = min((1 - slope_diff) * 60 + 25, 80)

    channel_width = np.mean([h["price"] for h in rh]) - np.mean([l["price"] for l in rl])

    return {
        "name": name,
        "type": ptype,
        "confidence": round(conf),
        "points": rh + rl,
        "lines": [
            {"x": [rh[0]["time"], rh[-1]["time"]], "y": [rh[0]["price"], rh[-1]["price"]], "color": color, "dash": "solid", "label": "Upper Channel"},
            {"x": [rl[0]["time"], rl[-1]["time"]], "y": [rl[0]["price"], rl[-1]["price"]], "color": color, "dash": "solid", "label": "Lower Channel"},
        ],
        "annotation": {"x": rh[-1]["time"], "y": (rh[-1]["price"] + rl[-1]["price"]) / 2, "text": name[:3] + " Ch"},
        "outcome": f"{'Bullish' if is_ascending else 'Bearish'} channel with width ₹{channel_width:.0f}. Trade: buy at lower channel, sell at upper. Break {'above' if is_ascending else 'below'} channel = acceleration.",
        "key_level": round(rl[-1]["price"] if is_ascending else rh[-1]["price"], 2),
        "target": None,
    }


# ════════════════════════════════════════════════════════
#  ADDITIONAL PATTERNS
# ════════════════════════════════════════════════════════

def detect_triple_top(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Three highs at similar level → bearish reversal."""
    highs = sp["highs"]
    if len(highs) < 3:
        return None
    h1, h2, h3 = highs[-3], highs[-2], highs[-1]
    if h2["idx"] - h1["idx"] < 5 or h3["idx"] - h2["idx"] < 5:
        return None
    prices = [h1["price"], h2["price"], h3["price"]]
    avg = np.mean(prices)
    spread = (max(prices) - min(prices)) / avg
    if spread > 0.03:
        return None
    between_lows = [l for l in sp["lows"] if h1["idx"] < l["idx"] < h3["idx"]]
    if not between_lows:
        return None
    neckline = min(l["price"] for l in between_lows)
    conf = min((1 - spread / 0.03) * 50 + 30, 90)
    target = neckline - (avg - neckline)
    return {
        "name": "Triple Top", "type": "bearish", "confidence": round(conf),
        "points": [h1, h2, h3],
        "lines": [
            {"x": [h1["time"], h3["time"]], "y": [avg, avg], "color": "rgba(255,92,92,0.7)", "dash": "solid", "label": "Resistance"},
            {"x": [between_lows[0]["time"], sp["times"][-1]], "y": [neckline, neckline], "color": "rgba(255,92,92,0.4)", "dash": "dash", "label": "Neckline"},
        ],
        "annotation": {"x": h3["time"], "y": h3["price"], "text": "Triple Top"},
        "outcome": f"Bearish. Three failed attempts at ₹{avg:.0f}. Breakdown below ₹{neckline:.0f} targets ₹{max(target,0):.0f}.",
        "key_level": round(neckline, 2), "target": round(max(target, 0), 2),
    }


def detect_triple_bottom(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Three lows at similar level → bullish reversal."""
    lows = sp["lows"]
    if len(lows) < 3:
        return None
    l1, l2, l3 = lows[-3], lows[-2], lows[-1]
    if l2["idx"] - l1["idx"] < 5 or l3["idx"] - l2["idx"] < 5:
        return None
    prices = [l1["price"], l2["price"], l3["price"]]
    avg = np.mean(prices)
    spread = (max(prices) - min(prices)) / avg
    if spread > 0.03:
        return None
    between_highs = [h for h in sp["highs"] if l1["idx"] < h["idx"] < l3["idx"]]
    if not between_highs:
        return None
    neckline = max(h["price"] for h in between_highs)
    conf = min((1 - spread / 0.03) * 50 + 30, 90)
    target = neckline + (neckline - avg)
    return {
        "name": "Triple Bottom", "type": "bullish", "confidence": round(conf),
        "points": [l1, l2, l3],
        "lines": [
            {"x": [l1["time"], l3["time"]], "y": [avg, avg], "color": "rgba(0,214,143,0.7)", "dash": "solid", "label": "Support"},
            {"x": [between_highs[0]["time"], sp["times"][-1]], "y": [neckline, neckline], "color": "rgba(0,214,143,0.4)", "dash": "dash", "label": "Neckline"},
        ],
        "annotation": {"x": l3["time"], "y": l3["price"], "text": "Triple Bottom"},
        "outcome": f"Bullish. Three bounces off ₹{avg:.0f}. Breakout above ₹{neckline:.0f} targets ₹{target:.0f}.",
        "key_level": round(neckline, 2), "target": round(target, 2),
    }


def detect_rounding_bottom(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Gradual U-shape recovery → bullish."""
    if len(closes) < 40:
        return None
    lookback = min(60, len(closes))
    segment = closes[-lookback:]
    mid = lookback // 2
    first_q = np.mean(segment[:mid // 2])
    bottom = np.mean(segment[mid - 3:mid + 3])
    last_q = np.mean(segment[-(mid // 2):])
    if bottom >= first_q or bottom >= last_q:
        return None
    left_drop = (first_q - bottom) / first_q
    right_rise = (last_q - bottom) / bottom
    if left_drop < 0.03 or right_rise < 0.03:
        return None
    symmetry = 1 - abs(left_drop - right_rise) / max(left_drop, right_rise)
    conf = min(symmetry * 50 + left_drop * 300, 85)
    current = float(closes[-1])
    target = first_q + (first_q - bottom)
    t = sp["times"]
    return {
        "name": "Rounding Bottom", "type": "bullish", "confidence": round(conf),
        "points": [],
        "lines": [
            {"x": [t[-lookback], t[-1]], "y": [float(first_q), float(first_q)], "color": "rgba(0,214,143,0.4)", "dash": "dash", "label": "Rim Level"},
        ],
        "annotation": {"x": t[max(0, len(t) - mid)], "y": float(bottom), "text": "Rounding Bottom"},
        "outcome": f"Bullish accumulation. Gradual U-shape recovery with {left_drop*100:.0f}% depth. Target ₹{target:.0f} on breakout above rim.",
        "key_level": round(float(first_q), 2), "target": round(target, 2),
    }


def detect_rectangle(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Horizontal consolidation between flat support and resistance."""
    highs = sp["highs"]
    lows = sp["lows"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    rh = highs[-4:] if len(highs) >= 4 else highs[-2:]
    rl = lows[-4:] if len(lows) >= 4 else lows[-2:]
    h_prices = [h["price"] for h in rh]
    l_prices = [l["price"] for l in rl]
    h_spread = (max(h_prices) - min(h_prices)) / np.mean(h_prices)
    l_spread = (max(l_prices) - min(l_prices)) / np.mean(l_prices)
    if h_spread > 0.03 or l_spread > 0.03:
        return None
    resistance = np.mean(h_prices)
    support = np.mean(l_prices)
    range_pct = (resistance - support) / support
    if range_pct < 0.02 or range_pct > 0.15:
        return None
    conf = min((1 - h_spread / 0.03) * 30 + (1 - l_spread / 0.03) * 30 + 20, 82)
    return {
        "name": "Rectangle", "type": "neutral", "confidence": round(conf),
        "points": rh + rl,
        "lines": [
            {"x": [rh[0]["time"], rh[-1]["time"]], "y": [resistance, resistance], "color": "rgba(255,184,77,0.7)", "dash": "solid", "label": "Resistance"},
            {"x": [rl[0]["time"], rl[-1]["time"]], "y": [support, support], "color": "rgba(255,184,77,0.7)", "dash": "solid", "label": "Support"},
        ],
        "annotation": {"x": rh[-1]["time"], "y": (resistance + support) / 2, "text": "Rectangle"},
        "outcome": f"Range-bound between ₹{support:.0f}–₹{resistance:.0f}. Breakout above = bullish to ₹{resistance + (resistance - support):.0f}. Breakdown below = bearish.",
        "key_level": round(support, 2), "target": None,
    }


def detect_broadening(sp: Dict, closes: np.ndarray) -> Optional[Dict]:
    """Expanding trendlines — increasing volatility."""
    highs = sp["highs"]
    lows = sp["lows"]
    if len(highs) < 3 or len(lows) < 3:
        return None
    rh = highs[-3:]
    rl = lows[-3:]
    h_prices = [h["price"] for h in rh]
    l_prices = [l["price"] for l in rl]
    if h_prices[-1] <= h_prices[0] or l_prices[-1] >= l_prices[0]:
        return None
    spread_start = h_prices[0] - l_prices[0]
    spread_end = h_prices[-1] - l_prices[-1]
    if spread_end <= spread_start:
        return None
    expansion = (spread_end - spread_start) / spread_start
    if expansion < 0.2:
        return None
    conf = min(expansion * 100 + 20, 80)
    return {
        "name": "Broadening Formation", "type": "neutral", "confidence": round(conf),
        "points": rh + rl,
        "lines": [
            {"x": [rh[0]["time"], rh[-1]["time"]], "y": [rh[0]["price"], rh[-1]["price"]], "color": "rgba(255,184,77,0.6)", "dash": "solid", "label": "Expanding High"},
            {"x": [rl[0]["time"], rl[-1]["time"]], "y": [rl[0]["price"], rl[-1]["price"]], "color": "rgba(255,184,77,0.6)", "dash": "solid", "label": "Expanding Low"},
        ],
        "annotation": {"x": rh[-1]["time"], "y": (rh[-1]["price"] + rl[-1]["price"]) / 2, "text": "Broadening"},
        "outcome": f"Increasing volatility — higher highs AND lower lows. Unstable. Wait for a decisive breakout in either direction before entering.",
        "key_level": round(rl[-1]["price"], 2), "target": None,
    }


# ════════════════════════════════════════════════════════
#  MASTER PATTERN DETECTION
# ════════════════════════════════════════════════════════

def detect_all_patterns(df: pd.DataFrame, min_confidence: int = 75, interval: str = "1d") -> List[Dict]:
    """
    Run all pattern detectors and return patterns above confidence threshold.
    Returns list sorted by confidence (highest first).
    """
    if df is None or df.empty or len(df) < 15:
        return []

    from datetime import datetime

    # Use multiple lookback periods for swing detection
    all_patterns = []

    for lookback in [3, 5, 7]:
        try:
            sp = find_swing_points(df, lookback=lookback)
            closes = sp["closes"]

            detectors = [
                detect_double_top,
                detect_double_bottom,
                detect_head_shoulders,
                detect_inv_head_shoulders,
                detect_ascending_triangle,
                detect_descending_triangle,
                detect_symmetrical_triangle,
                detect_cup_and_handle,
                detect_rising_wedge,
                detect_falling_wedge,
                detect_bull_flag,
                detect_bear_flag,
                detect_channel,
                detect_triple_top,
                detect_triple_bottom,
                detect_rounding_bottom,
                detect_rectangle,
                detect_broadening,
            ]

            for detector in detectors:
                try:
                    result = detector(sp, closes)
                    if result and result["confidence"] >= min_confidence:
                        # Avoid duplicate pattern types
                        if not any(p["name"] == result["name"] for p in all_patterns):
                            # Add detection metadata
                            result["detected_at"] = datetime.now().strftime("%d %b %Y, %I:%M %p IST")
                            result["timeframe"] = interval
                            # Find when the pattern started forming (earliest point)
                            if result.get("points"):
                                pts = result["points"]
                                if isinstance(pts, list) and len(pts) > 0 and isinstance(pts[0], dict):
                                    result["pattern_start"] = pts[0].get("time", "")
                                    result["pattern_end"] = pts[-1].get("time", "")
                            all_patterns.append(result)
                except Exception as e:
                    logger.debug(f"Pattern detector {detector.__name__} error: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Swing detection error (lookback={lookback}): {e}")
            continue

    # Sort by confidence descending
    all_patterns.sort(key=lambda p: -p["confidence"])

    # Return top 5 patterns max
    return all_patterns[:5]
