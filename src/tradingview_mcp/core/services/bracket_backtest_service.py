"""
Bracket Backtest Service — long+short backtests on Binance klines, 1m to 1d.

Covers scalping through swing trading with one engine. Fills the gaps the
Yahoo-based backtester can't: shorts, intraday intervals, intrabar bracket
exits, and explicit per-side costs. Data comes straight from Binance's public
klines API (no auth), with data-api.binance.vision as a mirror for regions
where api.binance.com is blocked.

Three public entry points, one shared core:
  - run_bracket_backtest     — one strategy, one parameter set
  - run_bracket_sweep        — grid over (rr, atr_mult) + neighborhood stability
  - run_bracket_walk_forward — per-fold re-optimization, out-of-sample verdict

Design choices that keep the results honest:
  - Exits are simulated intrabar against high/low. When a bar touches both
    the stop and the target, the STOP is assumed to fill first (conservative).
  - Gap-through fills execute at the bar's open, not the bracket price
    (worse fill on stops, better on targets — matches live behaviour).
  - Costs are charged per side (fee + slippage), so the round-trip hurdle is
    explicit in the output as `breakeven_move_pct`.
  - Walk-forward computes indicators over the full series (no lookahead — each
    bar's value uses only prior bars) but simulates each segment in isolation:
    positions are force-closed at the segment boundary so train trades can
    never bleed into the test window.

Pure Python — no pandas, no numpy.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.binance_data import (
    INTERVAL_MIN,
    INTERVAL_MS,
    MAX_CANDLES,
    fetch_binance_klines as _fetch_binance_klines,
)
from tradingview_mcp.core.services.indicators_calc import (
    calc_atr,
    calc_bollinger,
    calc_ema,
)
from tradingview_mcp.core.services.regime_service import (
    compute_regime_series,
    filter_signals_by_regime,
)

_INTERVAL_MS = INTERVAL_MS
_INTERVAL_MIN = INTERVAL_MIN
_MAX_CANDLES = MAX_CANDLES
_MAX_DAYS = 730
_ATR_PERIOD = 14

_VALID_DIRECTIONS = {"both", "long", "short"}

_STRATEGY_LABELS = {
    "squeeze_breakout": "Bollinger Squeeze Breakout (long+short, ATR bracket)",
    "ema_momentum": "EMA 9/21 Momentum with EMA200 Trend Gate (long+short, ATR bracket)",
}

# Bars needed before the strategy can emit its first signal.
_STRATEGY_MIN_BARS = {"squeeze_breakout": 150, "ema_momentum": 300}

_DEFAULT_RR_GRID = [1.0, 1.5, 2.0, 3.0]
_DEFAULT_ATR_GRID = [1.0, 1.5, 2.0, 3.0]
_MAX_GRID_VALUES = 8

_DISCLAIMER = ("Past performance does not guarantee future results. "
               "For educational use only.")


# ─── Signal Generation ────────────────────────────────────────────────────────
# Signals are (bar_index, "long"|"short") pairs; the bracket simulator turns
# them into trades. Both strategies signal at bar close, using only history up
# to that bar — so signals computed over the full series remain valid inside
# any walk-forward segment.

def _signals_squeeze_breakout(candles: list[dict], direction: str,
                              bb_period: int = 20, bb_std: float = 2.0,
                              squeeze_window: int = 60,
                              arm_bars: int = 5) -> list[tuple[int, str]]:
    """Squeeze = band width at a `squeeze_window`-bar low (no lookahead).
    Entry = close crossing outside a band within `arm_bars` of the squeeze.
    """
    closes = [c["close"] for c in candles]
    bb = calc_bollinger(closes, bb_period, bb_std)
    n = len(candles)

    bbw: list[Optional[float]] = [None] * n
    for i in range(n):
        u, m, l = bb["upper"][i], bb["middle"][i], bb["lower"][i]
        if u is not None and m:
            bbw[i] = (u - l) / m

    signals: list[tuple[int, str]] = []
    last_squeeze = None
    for i in range(1, n):
        if bbw[i] is not None and i >= squeeze_window:
            window = [w for w in bbw[i - squeeze_window:i] if w is not None]
            if window and bbw[i] <= min(window):
                last_squeeze = i
        if last_squeeze is None or i - last_squeeze > arm_bars:
            continue
        up, up_prev = bb["upper"][i], bb["upper"][i - 1]
        lo, lo_prev = bb["lower"][i], bb["lower"][i - 1]
        if None in (up, up_prev, lo, lo_prev):
            continue
        cross_up = closes[i - 1] <= up_prev and closes[i] > up
        cross_dn = closes[i - 1] >= lo_prev and closes[i] < lo
        if cross_up and direction != "short":
            signals.append((i, "long"))
            last_squeeze = None
        elif cross_dn and direction != "long":
            signals.append((i, "short"))
            last_squeeze = None
    return signals


def _signals_ema_momentum(candles: list[dict], direction: str,
                          fast: int = 9, slow: int = 21,
                          trend: int = 200) -> list[tuple[int, str]]:
    """EMA fast/slow cross taken only in the direction of the EMA(trend) side."""
    closes = [c["close"] for c in candles]
    ema_f = calc_ema(closes, fast)
    ema_s = calc_ema(closes, slow)
    ema_t = calc_ema(closes, trend)

    signals: list[tuple[int, str]] = []
    for i in range(1, len(candles)):
        f, s, fp, sp, t = ema_f[i], ema_s[i], ema_f[i - 1], ema_s[i - 1], ema_t[i]
        if None in (f, s, fp, sp, t):
            continue
        bull_cross = fp < sp and f >= s
        bear_cross = fp > sp and f <= s
        if bull_cross and closes[i] > t and direction != "short":
            signals.append((i, "long"))
        elif bear_cross and closes[i] < t and direction != "long":
            signals.append((i, "short"))
    return signals


_SIGNAL_MAP = {
    "squeeze_breakout": _signals_squeeze_breakout,
    "ema_momentum": _signals_ema_momentum,
}


# ─── Bracket Simulation ───────────────────────────────────────────────────────

def _simulate_bracket(candles: list[dict], signals: list[tuple[int, str]],
                      atr: list[Optional[float]], atr_mult: float, rr: float,
                      max_hold_bars: int,
                      start_index: int = 0,
                      end_index: Optional[int] = None) -> list[dict]:
    """One position at a time; entry at signal-bar close; exit at stop, target,
    time-stop, or segment end — whichever comes first.

    `start_index`/`end_index` bound the simulation to candles[start:end) so a
    walk-forward segment can never hold a position across its boundary.
    """
    n = len(candles) if end_index is None else min(end_index, len(candles))
    trades: list[dict] = []
    next_free = start_index

    for idx, side in signals:
        if idx < next_free or idx >= n - 1:
            continue
        a = atr[idx]
        if a is None or a <= 0:
            continue
        entry = candles[idx]["close"]
        risk = atr_mult * a
        if side == "long":
            stop, target = entry - risk, entry + rr * risk
        else:
            stop, target = entry + risk, entry - rr * risk

        exit_i, exit_price, exit_reason = None, None, None
        for j in range(idx + 1, min(idx + 1 + max_hold_bars, n)):
            bar = candles[j]
            if side == "long":
                if bar["low"] <= stop:
                    exit_i, exit_reason = j, "stop"
                    exit_price = min(stop, bar["open"])
                    break
                if bar["high"] >= target:
                    exit_i, exit_reason = j, "target"
                    exit_price = max(target, bar["open"])
                    break
            else:
                if bar["high"] >= stop:
                    exit_i, exit_reason = j, "stop"
                    exit_price = max(stop, bar["open"])
                    break
                if bar["low"] <= target:
                    exit_i, exit_reason = j, "target"
                    exit_price = min(target, bar["open"])
                    break
        if exit_i is None:
            exit_i = min(idx + max_hold_bars, n - 1)
            exit_price = candles[exit_i]["close"]
            exit_reason = "time" if exit_i - idx >= max_hold_bars else "end_of_data"

        trades.append({
            "entry_date": candles[idx]["date"],
            "entry_price": round(entry, 6),
            "direction": side,
            "exit_date": candles[exit_i]["date"],
            "exit_price": round(exit_price, 6),
            "exit_reason": exit_reason,
            "bars_held": exit_i - idx,
        })
        next_free = exit_i + 1
    return trades


# ─── Costs & Metrics ──────────────────────────────────────────────────────────

def _apply_bracket_costs(trades: list[dict], fee_pct: float,
                         slippage_pct: float) -> list[dict]:
    round_trip = (fee_pct + slippage_pct) * 2
    out = []
    for t in trades:
        move = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100
        gross = move if t["direction"] == "long" else -move
        out.append({**t,
                    "gross_return_pct": round(gross, 4),
                    "cost_pct": round(-round_trip, 4),
                    "return_pct": round(gross - round_trip, 4)})
    return out


def _bracket_metrics(trades: list[dict], initial_capital: float,
                     span_days: float, interval: str) -> dict:
    if not trades:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate_pct": 0, "final_capital": initial_capital,
            "total_return_pct": 0, "avg_gain_pct": 0, "avg_loss_pct": 0,
            "max_drawdown_pct": 0, "profit_factor": 0, "sharpe_ratio": 0,
            "expectancy_pct": 0, "trades_per_day": 0, "avg_hold_minutes": 0,
            "best_trade": None, "worst_trade": None,
            "exit_reasons": {}, "long_short_breakdown": {},
        }

    winners = [t for t in trades if t["return_pct"] > 0]
    losers = [t for t in trades if t["return_pct"] <= 0]

    capital, peak, max_dd = initial_capital, initial_capital, 0.0
    returns = []
    for t in trades:
        r = t["return_pct"] / 100
        capital *= (1 + r)
        returns.append(r)
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak * 100)

    total_return = (capital - initial_capital) / initial_capital * 100
    avg_gain = sum(t["return_pct"] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t["return_pct"] for t in losers) / len(losers) if losers else 0
    gp = sum(t["return_pct"] for t in winners)
    gl = abs(sum(t["return_pct"] for t in losers))
    profit_factor = round(gp / gl, 2) if gl > 0 else float("inf")

    # Per-trade Sharpe annualized by the observed trade frequency.
    sharpe = 0.0
    span_days = max(span_days, 1e-9)
    if len(returns) > 1:
        std_r = statistics.stdev(returns)
        if std_r > 0:
            trades_per_year = len(trades) / span_days * 365
            sharpe = round(statistics.mean(returns) / std_r
                           * math.sqrt(trades_per_year), 2)

    wr = len(winners) / len(trades)
    expectancy = round(wr * avg_gain + (1 - wr) * avg_loss, 4)
    best = max(trades, key=lambda t: t["return_pct"])
    worst = min(trades, key=lambda t: t["return_pct"])

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

    breakdown = {}
    for side in ("long", "short"):
        side_t = [t for t in trades if t["direction"] == side]
        if not side_t:
            continue
        side_w = [t for t in side_t if t["return_pct"] > 0]
        breakdown[side] = {
            "trades": len(side_t),
            "win_rate_pct": round(len(side_w) / len(side_t) * 100, 1),
            "avg_return_pct": round(
                sum(t["return_pct"] for t in side_t) / len(side_t), 4),
        }

    avg_hold_min = (sum(t["bars_held"] for t in trades) / len(trades)
                    * _INTERVAL_MIN[interval])

    return {
        "total_trades": len(trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_pct": round(wr * 100, 1),
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return, 2),
        "avg_gain_pct": round(avg_gain, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "max_drawdown_pct": round(-max_dd, 2),
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "expectancy_pct": expectancy,
        "trades_per_day": round(len(trades) / span_days, 2),
        "avg_hold_minutes": round(avg_hold_min, 1),
        "best_trade": {k: best[k] for k in
                       ("entry_date", "exit_date", "direction", "return_pct")},
        "worst_trade": {k: worst[k] for k in
                        ("entry_date", "exit_date", "direction", "return_pct")},
        "exit_reasons": reasons,
        "long_short_breakdown": breakdown,
    }


def _span_days(n_bars: int, interval: str) -> float:
    return n_bars * _INTERVAL_MIN[interval] / 1440


# ─── Shared validation & data prep ────────────────────────────────────────────

def _validate_common(strategy: str, interval: str, direction: str, days: int,
                     max_hold_bars: int, fee_pct: float, slippage_pct: float,
                     initial_capital: float) -> Optional[dict]:
    if strategy not in _SIGNAL_MAP:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"Unknown strategy '{strategy}'. "
                          f"Choose: {', '.join(_SIGNAL_MAP)}")
    if interval not in _INTERVAL_MS:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"Invalid interval '{interval}'. "
                          f"Choose: {', '.join(_INTERVAL_MS)}")
    if direction not in _VALID_DIRECTIONS:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"Invalid direction '{direction}'. "
                          f"Choose: {', '.join(sorted(_VALID_DIRECTIONS))}")
    if not (1 <= days <= _MAX_DAYS):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"days must be between 1 and {_MAX_DAYS}, got {days}")
    est_candles = days * 1440 // _INTERVAL_MIN[interval]
    if est_candles > _MAX_CANDLES:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"{days} days of {interval} candles ≈ {est_candles} bars "
                          f"(cap {_MAX_CANDLES}). Use fewer days or a larger interval.")
    min_bars = _STRATEGY_MIN_BARS[strategy]
    if est_candles < min_bars:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"{days} days of {interval} candles ≈ {est_candles} bars, "
                          f"but '{strategy}' needs ≥{min_bars} for indicator warmup. "
                          f"Use more days or a smaller interval.")
    if not (2 <= max_hold_bars <= 1000):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"max_hold_bars must be between 2 and 1000, got {max_hold_bars}")
    if not (0 <= fee_pct <= 5) or not (0 <= slippage_pct <= 5):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          "fee_pct and slippage_pct must be between 0 and 5 (percent per side)")
    if initial_capital <= 0:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"initial_capital must be positive, got {initial_capital}")
    return None


def _validate_bracket_params(rr: float, atr_mult: float) -> Optional[dict]:
    if not (0.2 <= rr <= 10):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"rr must be between 0.2 and 10, got {rr}")
    if not (0.2 <= atr_mult <= 10):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"atr_mult must be between 0.2 and 10, got {atr_mult}")
    return None


def _validate_grid(name: str, values: list[float]) -> Optional[dict]:
    if not values or len(values) > _MAX_GRID_VALUES:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"{name} must contain 1-{_MAX_GRID_VALUES} values")
    for v in values:
        if not (0.2 <= v <= 10):
            return make_error(ErrorCode.INVALID_PARAMETER,
                              f"{name} values must be between 0.2 and 10, got {v}")
    return None


def _validate_regime(regime_filter: bool, regime_anchor: str,
                     regime_interval: str) -> Optional[dict]:
    if not regime_filter:
        return None
    if regime_interval not in _INTERVAL_MS:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"Invalid regime_interval '{regime_interval}'. "
                          f"Choose: {', '.join(_INTERVAL_MS)}")
    if not regime_anchor or not regime_anchor.strip():
        return make_error(ErrorCode.INVALID_PARAMETER,
                          "regime_anchor must be a Binance symbol or 'self'")
    return None


def _prepare(symbol: str, strategy: str, interval: str, days: int,
             direction: str, regime_filter: bool = False,
             regime_anchor: str = "BTCUSDT", regime_interval: str = "4h"):
    """Fetch candles and compute ATR + signals once (regime-gated if asked).
    Returns a (candles, atr, signals, gate) tuple, or an error envelope dict.
    """
    try:
        candles = _fetch_binance_klines(symbol, interval, days)
    except ValueError as e:
        return make_error(ErrorCode.SYMBOL_NOT_FOUND, str(e), symbol=symbol)
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR,
                          f"Failed to fetch Binance klines for '{symbol}': {e}",
                          retryable=True)

    min_bars = _STRATEGY_MIN_BARS[strategy]
    if len(candles) < min_bars:
        return make_error(ErrorCode.NO_DATA,
                          f"Only {len(candles)} bars returned for {symbol} "
                          f"({interval}, {days}d); '{strategy}' needs ≥{min_bars}.")

    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    atr = calc_atr(highs, lows, closes, _ATR_PERIOD)
    signals = _SIGNAL_MAP[strategy](candles, direction)

    gate = {"enabled": False}
    if regime_filter:
        anchor_symbol = (symbol if regime_anchor.lower().strip() == "self"
                         else _normalize_symbol(regime_anchor))
        # Extra history so EMA200 on the anchor is warm from the first trade bar.
        warmup_days = math.ceil(210 * _INTERVAL_MIN[regime_interval] / 1440)
        anchor_days = min(_MAX_DAYS, days + warmup_days)
        try:
            anchor_candles = _fetch_binance_klines(anchor_symbol,
                                                   regime_interval, anchor_days)
        except ValueError as e:
            return make_error(ErrorCode.SYMBOL_NOT_FOUND, str(e),
                              symbol=anchor_symbol)
        except Exception as e:
            return make_error(ErrorCode.UPSTREAM_ERROR,
                              f"Failed to fetch regime anchor "
                              f"'{anchor_symbol}': {e}", retryable=True)
        regimes = compute_regime_series(anchor_candles)
        before = len(signals)
        signals, dropped = filter_signals_by_regime(
            signals, candles, interval, anchor_candles, regimes)
        gate = {
            "enabled": True,
            "anchor": anchor_symbol,
            "anchor_interval": regime_interval,
            "rule": "long only in anchor uptrend, short only in downtrend, "
                    "nothing in chop",
            "signals_before": before,
            "signals_after": len(signals),
            "dropped": dropped,
        }
    return candles, atr, signals, gate


def _run_combo(candles: list[dict], signals: list[tuple[int, str]],
               atr: list[Optional[float]], rr: float, atr_mult: float,
               max_hold_bars: int, fee_pct: float, slippage_pct: float,
               initial_capital: float, interval: str,
               start_index: int = 0,
               end_index: Optional[int] = None) -> tuple[list[dict], dict]:
    raw = _simulate_bracket(candles, signals, atr, atr_mult, rr,
                            max_hold_bars, start_index, end_index)
    trades = _apply_bracket_costs(raw, fee_pct, slippage_pct)
    n_bars = (len(candles) if end_index is None else end_index) - start_index
    metrics = _bracket_metrics(trades, initial_capital,
                               _span_days(n_bars, interval), interval)
    return trades, metrics


def _normalize_symbol(symbol: str) -> str:
    return (symbol.upper().strip()
            .replace("-", "").replace("/", "").replace(" ", ""))


# ─── Public API: single backtest ──────────────────────────────────────────────

def run_bracket_backtest(
    symbol: str,
    strategy: str = "squeeze_breakout",
    interval: str = "1h",
    days: int = 90,
    direction: str = "both",
    rr: float = 1.5,
    atr_mult: float = 1.0,
    max_hold_bars: int = 60,
    fee_pct: float = 0.05,
    slippage_pct: float = 0.02,
    initial_capital: float = 10_000.0,
    include_trade_log: bool = False,
    regime_filter: bool = False,
    regime_anchor: str = "BTCUSDT",
    regime_interval: str = "4h",
) -> dict:
    symbol = _normalize_symbol(symbol)
    strategy = strategy.lower().strip()
    interval = interval.lower().strip()
    direction = direction.lower().strip()
    regime_interval = regime_interval.lower().strip()

    err = (_validate_common(strategy, interval, direction, days, max_hold_bars,
                            fee_pct, slippage_pct, initial_capital)
           or _validate_bracket_params(rr, atr_mult)
           or _validate_regime(regime_filter, regime_anchor, regime_interval))
    if err:
        return err

    prep = _prepare(symbol, strategy, interval, days, direction,
                    regime_filter, regime_anchor, regime_interval)
    if isinstance(prep, dict):
        return prep
    candles, atr, signals, gate = prep

    trades, metrics = _run_combo(candles, signals, atr, rr, atr_mult,
                                 max_hold_bars, fee_pct, slippage_pct,
                                 initial_capital, interval)

    bnh = round((candles[-1]["close"] - candles[0]["close"])
                / candles[0]["close"] * 100, 2)
    breakeven = round((fee_pct + slippage_pct) * 2, 4)

    result = {
        "symbol": symbol,
        "exchange": "BINANCE",
        "strategy": strategy,
        "strategy_label": _STRATEGY_LABELS[strategy],
        "interval": interval,
        "days": days,
        "direction": direction,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "params": {
            "rr": rr,
            "atr_mult": atr_mult,
            "max_hold_bars": max_hold_bars,
            "atr_period": _ATR_PERIOD,
        },
        "costs": {
            "fee_pct_per_side": fee_pct,
            "slippage_pct_per_side": slippage_pct,
            "breakeven_move_pct": breakeven,
            "note": "A trade must move more than breakeven_move_pct in your "
                    "favor before it earns anything.",
        },
        "initial_capital": round(initial_capital, 2),
        "regime_gate": gate,
        **metrics,
        "signals_generated": len(signals),
        "buy_and_hold_return_pct": bnh,
        "vs_buy_and_hold_pct": round(metrics["total_return_pct"] - bnh, 2),
        "recent_trades": trades[-5:],
        "data_source": "Binance public klines API",
        "disclaimer": _DISCLAIMER,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if include_trade_log:
        result["trade_log"] = trades
    return result


# ─── Public API: parameter sweep ──────────────────────────────────────────────

def run_bracket_sweep(
    symbol: str,
    strategy: str = "squeeze_breakout",
    interval: str = "1h",
    days: int = 180,
    direction: str = "both",
    rr_values: Optional[list[float]] = None,
    atr_mult_values: Optional[list[float]] = None,
    max_hold_bars: int = 60,
    fee_pct: float = 0.05,
    slippage_pct: float = 0.02,
    initial_capital: float = 10_000.0,
    regime_filter: bool = False,
    regime_anchor: str = "BTCUSDT",
    regime_interval: str = "4h",
) -> dict:
    """Grid-search (rr × atr_mult) on one symbol and report every cell plus a
    neighborhood-stability read on the best cell. A profitable cell whose
    neighbors all lose is noise, not edge.
    """
    symbol = _normalize_symbol(symbol)
    strategy = strategy.lower().strip()
    interval = interval.lower().strip()
    direction = direction.lower().strip()
    regime_interval = regime_interval.lower().strip()
    rr_values = list(_DEFAULT_RR_GRID) if rr_values is None else rr_values
    atr_mult_values = (list(_DEFAULT_ATR_GRID) if atr_mult_values is None
                       else atr_mult_values)

    err = (_validate_common(strategy, interval, direction, days, max_hold_bars,
                            fee_pct, slippage_pct, initial_capital)
           or _validate_grid("rr_values", rr_values)
           or _validate_grid("atr_mult_values", atr_mult_values)
           or _validate_regime(regime_filter, regime_anchor, regime_interval))
    if err:
        return err

    prep = _prepare(symbol, strategy, interval, days, direction,
                    regime_filter, regime_anchor, regime_interval)
    if isinstance(prep, dict):
        return prep
    candles, atr, signals, gate = prep

    cells = []
    for ri, rr in enumerate(rr_values):
        for ai, am in enumerate(atr_mult_values):
            _, m = _run_combo(candles, signals, atr, rr, am, max_hold_bars,
                              fee_pct, slippage_pct, initial_capital, interval)
            cells.append({
                "rr": rr, "atr_mult": am, "_ri": ri, "_ai": ai,
                "total_trades": m["total_trades"],
                "win_rate_pct": m["win_rate_pct"],
                "total_return_pct": m["total_return_pct"],
                "profit_factor": m["profit_factor"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "expectancy_pct": m["expectancy_pct"],
                "sharpe_ratio": m["sharpe_ratio"],
            })

    ranked = sorted(cells, key=lambda c: c["total_return_pct"], reverse=True)
    best = ranked[0]

    neighbors = [c for c in cells
                 if abs(c["_ri"] - best["_ri"]) + abs(c["_ai"] - best["_ai"]) == 1]
    stability = None
    if neighbors:
        profitable = [c for c in neighbors if c["total_return_pct"] > 0]
        returns = sorted(c["total_return_pct"] for c in neighbors)
        stability = {
            "neighbors_checked": len(neighbors),
            "neighbors_profitable": len(profitable),
            "neighbor_median_return_pct": round(statistics.median(returns), 2),
        }

    if best["total_return_pct"] <= 0:
        verdict = ("NO_EDGE — no parameter combination is profitable in this "
                   "window; do not tune further, change the strategy or add a "
                   "regime filter.")
    elif stability and stability["neighbors_profitable"] / stability["neighbors_checked"] >= 0.5:
        verdict = ("STABLE_CANDIDATE — best cell's neighbors are mostly "
                   "profitable too; worth walk-forward validation.")
    else:
        verdict = ("FRAGILE — best cell is an isolated profitable island; "
                   "likely overfit noise, do not trade it.")

    for c in ranked:
        c.pop("_ri"), c.pop("_ai")

    return {
        "symbol": symbol,
        "exchange": "BINANCE",
        "strategy": strategy,
        "strategy_label": _STRATEGY_LABELS[strategy],
        "interval": interval,
        "days": days,
        "direction": direction,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "grid": {"rr_values": rr_values, "atr_mult_values": atr_mult_values,
                 "combos_tested": len(cells)},
        "costs": {"fee_pct_per_side": fee_pct,
                  "slippage_pct_per_side": slippage_pct,
                  "breakeven_move_pct": round((fee_pct + slippage_pct) * 2, 4)},
        "regime_gate": gate,
        "best": {k: best[k] for k in
                 ("rr", "atr_mult", "total_return_pct", "total_trades",
                  "win_rate_pct", "profit_factor", "max_drawdown_pct")},
        "best_neighborhood": stability,
        "verdict": verdict,
        "results": ranked,
        "buy_and_hold_return_pct": round(
            (candles[-1]["close"] - candles[0]["close"])
            / candles[0]["close"] * 100, 2),
        "data_source": "Binance public klines API",
        "disclaimer": _DISCLAIMER,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Public API: walk-forward with per-fold optimization ─────────────────────

def run_bracket_walk_forward(
    symbol: str,
    strategy: str = "squeeze_breakout",
    interval: str = "1h",
    days: int = 365,
    direction: str = "both",
    n_splits: int = 4,
    train_ratio: float = 0.7,
    rr_values: Optional[list[float]] = None,
    atr_mult_values: Optional[list[float]] = None,
    min_trades: int = 5,
    max_hold_bars: int = 60,
    fee_pct: float = 0.05,
    slippage_pct: float = 0.02,
    initial_capital: float = 10_000.0,
    regime_filter: bool = False,
    regime_anchor: str = "BTCUSDT",
    regime_interval: str = "4h",
) -> dict:
    """True walk-forward: in each fold, pick the best (rr, atr_mult) on the
    train window, then apply ONLY that combo to the unseen test window. The
    aggregated test performance is the honest estimate of live behaviour.
    """
    symbol = _normalize_symbol(symbol)
    strategy = strategy.lower().strip()
    interval = interval.lower().strip()
    direction = direction.lower().strip()
    regime_interval = regime_interval.lower().strip()
    rr_values = list(_DEFAULT_RR_GRID) if rr_values is None else rr_values
    atr_mult_values = (list(_DEFAULT_ATR_GRID) if atr_mult_values is None
                       else atr_mult_values)

    err = (_validate_common(strategy, interval, direction, days, max_hold_bars,
                            fee_pct, slippage_pct, initial_capital)
           or _validate_grid("rr_values", rr_values)
           or _validate_grid("atr_mult_values", atr_mult_values)
           or _validate_regime(regime_filter, regime_anchor, regime_interval))
    if err:
        return err
    if not (2 <= n_splits <= 10):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"n_splits must be between 2 and 10, got {n_splits}")
    if not (0.5 <= train_ratio <= 0.9):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"train_ratio must be between 0.5 and 0.9, got {train_ratio}")
    if not (1 <= min_trades <= 100):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"min_trades must be between 1 and 100, got {min_trades}")

    prep = _prepare(symbol, strategy, interval, days, direction,
                    regime_filter, regime_anchor, regime_interval)
    if isinstance(prep, dict):
        return prep
    candles, atr, signals, gate = prep

    fold_size = len(candles) // n_splits
    if fold_size < 50:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          f"{len(candles)} bars / {n_splits} splits = folds of "
                          f"{fold_size} bars — too small. Use more days or fewer splits.")

    folds = []
    all_test_trades: list[dict] = []
    total_test_bars = 0
    chosen_params: list[tuple[float, float]] = []

    for k in range(n_splits):
        a = k * fold_size
        b = (a + fold_size) if k < n_splits - 1 else len(candles)
        s = a + int((b - a) * train_ratio)

        best_combo, best_train_return, best_train_trades = None, None, 0
        for rr in rr_values:
            for am in atr_mult_values:
                _, m = _run_combo(candles, signals, atr, rr, am, max_hold_bars,
                                  fee_pct, slippage_pct, initial_capital,
                                  interval, start_index=a, end_index=s)
                if m["total_trades"] < min_trades:
                    continue
                if best_train_return is None or m["total_return_pct"] > best_train_return:
                    best_combo = (rr, am)
                    best_train_return = m["total_return_pct"]
                    best_train_trades = m["total_trades"]

        if best_combo is None:
            folds.append({
                "fold": k + 1,
                "train_from": candles[a]["date"], "train_to": candles[s - 1]["date"],
                "skipped": f"no combo produced ≥{min_trades} train trades",
            })
            continue

        rr, am = best_combo
        test_trades, test_m = _run_combo(candles, signals, atr, rr, am,
                                         max_hold_bars, fee_pct, slippage_pct,
                                         initial_capital, interval,
                                         start_index=s, end_index=b)
        all_test_trades.extend(test_trades)
        total_test_bars += b - s
        chosen_params.append(best_combo)

        folds.append({
            "fold": k + 1,
            "train_from": candles[a]["date"], "train_to": candles[s - 1]["date"],
            "test_from": candles[s]["date"], "test_to": candles[b - 1]["date"],
            "best_params": {"rr": rr, "atr_mult": am},
            "train_return_pct": best_train_return,
            "train_trades": best_train_trades,
            "test_return_pct": test_m["total_return_pct"],
            "test_trades": test_m["total_trades"],
            "test_win_rate_pct": test_m["win_rate_pct"],
        })

    evaluated = [f for f in folds if "skipped" not in f]
    if not evaluated:
        return make_error(ErrorCode.NO_DATA,
                          "Every fold was skipped — the strategy produced too few "
                          "trades to optimize. Use more days, a smaller interval, "
                          "or min_trades=1.")

    oos = _bracket_metrics(all_test_trades, initial_capital,
                           _span_days(total_test_bars, interval), interval)
    positive_folds = sum(1 for f in evaluated if f["test_return_pct"] > 0)
    avg_train = round(statistics.mean(f["train_return_pct"] for f in evaluated), 2)

    distinct = len(set(chosen_params))
    if distinct == 1:
        param_stability = "STABLE — every fold chose the same parameters"
    elif distinct <= max(2, len(evaluated) // 2):
        param_stability = "MODERATE — parameter choice drifts between folds"
    else:
        param_stability = "UNSTABLE — each fold wants different parameters (bad sign)"

    oos_ret = oos["total_return_pct"]
    if oos_ret > 0 and positive_folds / len(evaluated) >= 0.6:
        verdict = ("PASSED — edge survives out-of-sample across most folds; "
                   "candidate for paper trading.")
    elif oos_ret > 0:
        verdict = ("MIXED — positive out-of-sample overall but inconsistent "
                   "across folds; treat with suspicion.")
    elif avg_train > 0:
        verdict = ("OVERFIT — profitable in-sample, loses on unseen data. "
                   "The optimizer is fitting noise; do not trade this.")
    else:
        verdict = ("NO_EDGE — loses both in-sample and out-of-sample in this "
                   "window. Change strategy, timeframe, or add a regime filter.")

    return {
        "symbol": symbol,
        "exchange": "BINANCE",
        "strategy": strategy,
        "strategy_label": _STRATEGY_LABELS[strategy],
        "interval": interval,
        "days": days,
        "direction": direction,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "n_splits": n_splits,
        "train_ratio": train_ratio,
        "grid": {"rr_values": rr_values, "atr_mult_values": atr_mult_values},
        "costs": {"fee_pct_per_side": fee_pct,
                  "slippage_pct_per_side": slippage_pct,
                  "breakeven_move_pct": round((fee_pct + slippage_pct) * 2, 4)},
        "regime_gate": gate,
        "folds": folds,
        "folds_evaluated": len(evaluated),
        "folds_positive_oos": positive_folds,
        "avg_train_return_pct": avg_train,
        "oos_total_return_pct": oos_ret,
        "oos_total_trades": oos["total_trades"],
        "oos_win_rate_pct": oos["win_rate_pct"],
        "oos_profit_factor": oos["profit_factor"],
        "oos_max_drawdown_pct": oos["max_drawdown_pct"],
        "oos_long_short_breakdown": oos["long_short_breakdown"],
        "param_stability": param_stability,
        "verdict": verdict,
        "buy_and_hold_return_pct": round(
            (candles[-1]["close"] - candles[0]["close"])
            / candles[0]["close"] * 100, 2),
        "data_source": "Binance public klines API",
        "disclaimer": _DISCLAIMER,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
