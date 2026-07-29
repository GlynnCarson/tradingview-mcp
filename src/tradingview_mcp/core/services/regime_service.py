"""
Market Regime Service — trend/volatility classification and signal gating.

The single biggest failure mode observed in Phase 1 walk-forwards was regime
mismatch: long signals firing in downtrends, breakouts firing in chop. This
module classifies each bar of an anchor symbol (default BTCUSDT — the market
leader) and gates strategy signals accordingly:

    trend "up"   → only long signals pass
    trend "down" → only short signals pass
    trend "chop" → nothing passes

Classification per anchor bar (no lookahead — bar i uses only bars ≤ i):
    trend: EMA(fast) vs EMA(slow) with a dead-band. ratio = fast/slow − 1;
           up if ratio > band, down if ratio < −band, else chop.
    vol:   ATR% of price vs the median of its own PAST values; "high"/"low".
           (Reported for context; gating uses trend only — every extra gate
           is another parameter to overfit.)

Gating uses only anchor bars that have CLOSED by the trade decision moment
(the close of the signal bar), so a backtest can never peek at an in-progress
higher-timeframe bar.

Pure Python — no pandas, no numpy.
"""
from __future__ import annotations

import statistics
from bisect import bisect_right
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.binance_data import (
    INTERVAL_MS,
    fetch_binance_klines,
)
from tradingview_mcp.core.services.indicators_calc import calc_atr, calc_ema

_TREND_FAST = 50
_TREND_SLOW = 200
_TREND_BAND = 0.002  # ±0.2% dead-band around EMA parity → "chop"
_ATR_PERIOD = 14
_VOL_LOOKBACK = 100
_VOL_MIN_HISTORY = 20


# ─── Classification ───────────────────────────────────────────────────────────

def compute_regime_series(candles: list[dict],
                          fast: int = _TREND_FAST,
                          slow: int = _TREND_SLOW,
                          band: float = _TREND_BAND) -> list[Optional[dict]]:
    """Per-bar regime for an anchor series. None until EMA(slow) warmup.

    Each entry: {"trend": "up"|"down"|"chop", "vol": "high"|"low",
                 "ema_fast": float, "ema_slow": float, "atr_pct": float|None}
    """
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    ema_f = calc_ema(closes, fast)
    ema_s = calc_ema(closes, slow)
    atr = calc_atr(highs, lows, closes, _ATR_PERIOD)

    atr_pct: list[Optional[float]] = [
        (atr[i] / closes[i] if atr[i] is not None and closes[i] else None)
        for i in range(len(candles))
    ]

    out: list[Optional[dict]] = []
    for i in range(len(candles)):
        f, s = ema_f[i], ema_s[i]
        if f is None or s is None or s == 0:
            out.append(None)
            continue
        ratio = f / s - 1
        if ratio > band:
            trend = "up"
        elif ratio < -band:
            trend = "down"
        else:
            trend = "chop"

        vol = None
        if atr_pct[i] is not None:
            past = [v for v in atr_pct[max(0, i - _VOL_LOOKBACK):i]
                    if v is not None]
            if len(past) >= _VOL_MIN_HISTORY:
                vol = "high" if atr_pct[i] > statistics.median(past) else "low"

        out.append({
            "trend": trend,
            "vol": vol,
            "ema_fast": round(f, 6),
            "ema_slow": round(s, 6),
            "atr_pct": round(atr_pct[i], 6) if atr_pct[i] is not None else None,
        })
    return out


# ─── Signal gating ────────────────────────────────────────────────────────────

def filter_signals_by_regime(
    signals: list[tuple[int, str]],
    trade_candles: list[dict],
    trade_interval: str,
    anchor_candles: list[dict],
    regimes: list[Optional[dict]],
) -> tuple[list[tuple[int, str]], dict]:
    """Keep only signals aligned with the anchor's trend at decision time.

    Decision time = close of the signal bar. The regime used is the latest
    anchor bar whose CLOSE is at or before that moment.

    Returns (kept_signals, drop_stats).
    """
    inferred = _infer_interval(anchor_candles)
    anchor_step = INTERVAL_MS[inferred] if inferred else 0
    anchor_closes = [c["ts"] + anchor_step for c in anchor_candles]
    trade_step = INTERVAL_MS[trade_interval]

    kept: list[tuple[int, str]] = []
    dropped = {"chop": 0, "counter_trend": 0, "no_regime_data": 0}

    for idx, side in signals:
        decision_ts = trade_candles[idx]["ts"] + trade_step
        k = bisect_right(anchor_closes, decision_ts) - 1
        regime = regimes[k] if 0 <= k < len(regimes) else None
        if regime is None:
            dropped["no_regime_data"] += 1
            continue
        trend = regime["trend"]
        if trend == "chop":
            dropped["chop"] += 1
        elif (trend == "up") == (side == "long"):
            kept.append((idx, side))
        else:
            dropped["counter_trend"] += 1
    return kept, dropped


def _infer_interval(anchor_candles: list[dict]) -> Optional[str]:
    if len(anchor_candles) < 2:
        return None
    gap = anchor_candles[1]["ts"] - anchor_candles[0]["ts"]
    for name, ms in INTERVAL_MS.items():
        if ms == gap:
            return name
    return None


# ─── Public API: current regime snapshot ─────────────────────────────────────

def get_market_regime(symbol: str = "BTCUSDT", interval: str = "4h",
                      days: int = 180) -> dict:
    """Current market regime for an anchor symbol, plus recent history.

    This is what the (future) paper trader will consult before taking trades,
    and what `regime_filter=True` applies bar-by-bar inside backtests.
    """
    symbol = symbol.upper().strip().replace("-", "").replace("/", "")
    interval = interval.lower().strip()
    if interval not in INTERVAL_MS:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"Invalid interval '{interval}'. "
                          f"Choose: {', '.join(INTERVAL_MS)}")
    if not (1 <= days <= 730):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"days must be between 1 and 730, got {days}")

    try:
        candles = fetch_binance_klines(symbol, interval, days)
    except ValueError as e:
        return make_error(ErrorCode.SYMBOL_NOT_FOUND, str(e), symbol=symbol)
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR,
                          f"Failed to fetch Binance klines for '{symbol}': {e}",
                          retryable=True)

    regimes = compute_regime_series(candles)
    current = next((r for r in reversed(regimes) if r is not None), None)
    if current is None:
        return make_error(ErrorCode.NO_DATA,
                          f"Only {len(candles)} bars for {symbol} ({interval}, "
                          f"{days}d) — not enough for EMA{_TREND_SLOW} warmup. "
                          f"Use more days.")

    valid = [(i, r) for i, r in enumerate(regimes) if r is not None]
    last30 = [r for _, r in valid[-30:]]
    counts = {"up": 0, "down": 0, "chop": 0}
    for r in last30:
        counts[r["trend"]] += 1

    transitions = []
    prev_trend = None
    for i, r in valid:
        if r["trend"] != prev_trend:
            transitions.append({"date": candles[i]["date"], "trend": r["trend"]})
            prev_trend = r["trend"]
    transitions = transitions[-4:]

    allowed = {"up": ["long"], "down": ["short"], "chop": []}[current["trend"]]
    advice = {
        "up": "Anchor trend is UP — long setups enabled, shorts gated off.",
        "down": "Anchor trend is DOWN — short setups enabled, longs gated off.",
        "chop": "Anchor is CHOPPING — no directional edge; regime gate blocks all entries.",
    }[current["trend"]]

    return {
        "symbol": symbol,
        "exchange": "BINANCE",
        "interval": interval,
        "days": days,
        "bars_analyzed": len(candles),
        "as_of": candles[-1]["date"],
        "current": {
            "trend": current["trend"],
            "volatility": current["vol"],
            "ema_fast": current["ema_fast"],
            "ema_slow": current["ema_slow"],
            "atr_pct": current["atr_pct"],
            "price": candles[-1]["close"],
        },
        "allowed_directions": allowed,
        "advice": advice,
        "last_30_bars": counts,
        "recent_transitions": transitions,
        "classification": {
            "trend_rule": f"EMA{_TREND_FAST}/EMA{_TREND_SLOW} ratio with "
                          f"±{_TREND_BAND * 100:.1f}% dead-band",
            "vol_rule": f"ATR({_ATR_PERIOD})% vs median of past "
                        f"{_VOL_LOOKBACK} bars",
        },
        "data_source": "Binance public klines API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
