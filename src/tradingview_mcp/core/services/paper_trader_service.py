"""
Paper Trader Service — Phase 3 of the automated-trading pipeline.

Runs the walk-forward-validated strategies against LIVE Binance data with fake
money, using the exact same bracket semantics as the backtester (stop-first
intrabar checks, gap fills at open, per-side fee+slippage). Its purpose is to
generate cheap additional out-of-sample evidence before any real capital is
risked — and its trade log doubles as the dataset for a future ML trade filter.

State lives in a JSON file (default ~/.tradingview-mcp/paper_account.json).
Each `paper_step()` call processes every candle that CLOSED since the last
step, so the account catches up correctly even if the daemon was down.
The in-progress candle is always ignored — decisions use closed bars only.

Risk rules are enforced in code, not left to discipline:
  - risk_pct_per_trade   — position sized so a stop-out loses ~1% of capital
                           (notional capped at 1x capital, no leverage)
  - daily_loss_halt_pct  — realized loss beyond this % of the day's starting
                           capital blocks NEW entries until the next UTC day
  - max_drawdown_kill_pct — drawdown from peak capital beyond this freezes the
                           account entirely until a manual paper_reset

Run one tick:      python -m tradingview_mcp.core.services.paper_trader_service
Run as a daemon:   python -m ...paper_trader_service --loop 3600

Pure Python — no pandas, no numpy.
"""
from __future__ import annotations

import json
import math
import os
import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.binance_data import (
    INTERVAL_MIN,
    INTERVAL_MS,
    fetch_binance_klines,
)
from tradingview_mcp.core.services.bracket_backtest_service import (
    _SIGNAL_MAP,
    _STRATEGY_MIN_BARS,
)
from tradingview_mcp.core.services.indicators_calc import calc_atr
from tradingview_mcp.core.services.regime_service import compute_regime_series

_ATR_PERIOD = 14

# The two Phase 1/2 walk-forward survivors. Edit via paper_reset(strategies=...).
_DEFAULT_STRATEGIES = [
    {"id": "btc_squeeze_1h", "symbol": "BTCUSDT", "strategy": "squeeze_breakout",
     "interval": "1h", "direction": "both", "rr": 1.5, "atr_mult": 2.0,
     "max_hold_bars": 60, "regime_filter": False},
    {"id": "eth_squeeze_1h_gated", "symbol": "ETHUSDT",
     "strategy": "squeeze_breakout", "interval": "1h", "direction": "both",
     "rr": 2.0, "atr_mult": 2.0, "max_hold_bars": 60, "regime_filter": True,
     "regime_anchor": "BTCUSDT", "regime_interval": "4h"},
]

_DEFAULT_CONFIG = {
    "initial_capital": 10_000.0,
    "risk_pct_per_trade": 1.0,
    "daily_loss_halt_pct": 3.0,
    "max_drawdown_kill_pct": 15.0,
    "fee_pct": 0.05,
    "slippage_pct": 0.02,
}


def _default_state_path() -> str:
    override = os.environ.get("TVMCP_PAPER_STATE")
    if override:
        return override
    return str(Path.home() / ".tradingview-mcp" / "paper_account.json")


def _utc_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _date_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ─── State I/O ────────────────────────────────────────────────────────────────

def _load_state(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: str, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, p)


# ─── Account lifecycle ────────────────────────────────────────────────────────

def paper_reset(
    confirm: bool = False,
    initial_capital: float = 10_000.0,
    risk_pct_per_trade: float = 1.0,
    daily_loss_halt_pct: float = 3.0,
    max_drawdown_kill_pct: float = 15.0,
    fee_pct: float = 0.05,
    slippage_pct: float = 0.02,
    strategies: Optional[list[dict]] = None,
    state_path: Optional[str] = None,
) -> dict:
    """Create (or wipe and recreate) the paper-trading account."""
    path = state_path or _default_state_path()
    existing = _load_state(path)
    if existing is not None and not confirm:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"A paper account already exists at {path} with "
            f"{len(existing.get('closed_trades', []))} closed trades and "
            f"capital {existing.get('capital')}. Pass confirm=true to wipe it.")

    if initial_capital <= 0:
        return make_error(ErrorCode.INVALID_PARAMETER,
                          "initial_capital must be positive")
    if not (0.1 <= risk_pct_per_trade <= 5):
        return make_error(ErrorCode.INVALID_PARAMETER,
                          "risk_pct_per_trade must be between 0.1 and 5")

    strategies = strategies or [dict(s) for s in _DEFAULT_STRATEGIES]
    for s in strategies:
        for key in ("id", "symbol", "strategy", "interval"):
            if key not in s:
                return make_error(ErrorCode.INVALID_PARAMETER,
                                  f"strategy config missing '{key}': {s}")
        if s["strategy"] not in _SIGNAL_MAP:
            return make_error(ErrorCode.INVALID_PARAMETER,
                              f"unknown strategy '{s['strategy']}'")
        if s["interval"] not in INTERVAL_MS:
            return make_error(ErrorCode.INVALID_PARAMETER,
                              f"invalid interval '{s['interval']}'")

    state = {
        "created": datetime.now(timezone.utc).isoformat(),
        "config": {
            "initial_capital": initial_capital,
            "risk_pct_per_trade": risk_pct_per_trade,
            "daily_loss_halt_pct": daily_loss_halt_pct,
            "max_drawdown_kill_pct": max_drawdown_kill_pct,
            "fee_pct": fee_pct,
            "slippage_pct": slippage_pct,
            "strategies": strategies,
        },
        "capital": initial_capital,
        "peak_capital": initial_capital,
        "killed": False,
        "open_positions": [],
        "closed_trades": [],
        "daily": {},
        "last_processed": {},
    }
    _save_state(path, state)
    return {"status": "created", "state_path": path,
            "capital": initial_capital,
            "strategies": [s["id"] for s in strategies],
            "risk_rules": {
                "risk_pct_per_trade": risk_pct_per_trade,
                "daily_loss_halt_pct": daily_loss_halt_pct,
                "max_drawdown_kill_pct": max_drawdown_kill_pct,
            }}


# ─── Core mechanics ───────────────────────────────────────────────────────────

def _closed_candles(symbol: str, interval: str, min_bars: int,
                    now_ms: int) -> list[dict]:
    """Fetch and return only candles that have CLOSED by now_ms."""
    days = math.ceil(min_bars * INTERVAL_MIN[interval] / 1440) + 4
    candles = fetch_binance_klines(symbol, interval, days)
    step = INTERVAL_MS[interval]
    return [c for c in candles if c["ts"] + step <= now_ms]


def _regime_allows(side: str, cfg: dict, now_ms: int) -> tuple[bool, str]:
    """Check the anchor's trend for a gated strategy at decision time."""
    anchor = cfg.get("regime_anchor", "BTCUSDT")
    interval = cfg.get("regime_interval", "4h")
    candles = _closed_candles(anchor, interval, 220, now_ms)
    regimes = compute_regime_series(candles)
    current = next((r for r in reversed(regimes) if r is not None), None)
    if current is None:
        return False, "no_regime_data"
    trend = current["trend"]
    if trend == "chop":
        return False, "regime_chop"
    if (trend == "up") == (side == "long"):
        return True, trend
    return False, f"counter_trend_{trend}"


def _close_position(state: dict, pos: dict, exit_price: float, exit_ts: int,
                    exit_reason: str) -> dict:
    cfg = state["config"]
    move = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
    gross = move if pos["direction"] == "long" else -move
    round_trip = (cfg["fee_pct"] + cfg["slippage_pct"]) * 2
    net_pct = gross - round_trip
    pnl = pos["notional"] * net_pct / 100

    state["capital"] += pnl
    state["peak_capital"] = max(state["peak_capital"], state["capital"])

    day = _utc_day(exit_ts)
    daily = state["daily"].setdefault(
        day, {"start_capital": state["capital"] - pnl, "realized_pnl": 0.0})
    daily["realized_pnl"] += pnl

    trade = {
        **{k: pos[k] for k in ("strategy_id", "symbol", "direction",
                               "entry_date", "entry_price", "notional",
                               "stop", "target")},
        "exit_date": _date_str(exit_ts),
        "exit_price": round(exit_price, 6),
        "exit_reason": exit_reason,
        "bars_held": pos["bars_held"],
        "gross_return_pct": round(gross, 4),
        "net_return_pct": round(net_pct, 4),
        "pnl": round(pnl, 2),
        "capital_after": round(state["capital"], 2),
    }
    state["closed_trades"].append(trade)
    return trade


def _daily_halted(state: dict, ts_ms: int) -> bool:
    day = state["daily"].get(_utc_day(ts_ms))
    if not day:
        return False
    limit = state["config"]["daily_loss_halt_pct"] / 100 * day["start_capital"]
    return day["realized_pnl"] <= -limit


def _check_kill_switch(state: dict) -> bool:
    if state["killed"]:
        return True
    dd_limit = state["config"]["max_drawdown_kill_pct"] / 100
    if state["capital"] <= state["peak_capital"] * (1 - dd_limit):
        state["killed"] = True
        return True
    return False


# ─── The tick ─────────────────────────────────────────────────────────────────

def paper_step(state_path: Optional[str] = None,
               now_ms: Optional[int] = None) -> dict:
    """Process one tick: manage open positions and take new entries on every
    candle that closed since the last step. Safe to call as often as you like;
    idempotent for bars already processed.
    """
    path = state_path or _default_state_path()
    state = _load_state(path)
    if state is None:
        return make_error(ErrorCode.NO_DATA,
                          f"No paper account at {path}. Run paper_reset first.")
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    cfg = state["config"]
    events: list[dict] = []

    for strat in cfg["strategies"]:
        sid = strat["id"]
        try:
            min_bars = _STRATEGY_MIN_BARS[strat["strategy"]]
            candles = _closed_candles(strat["symbol"], strat["interval"],
                                      min_bars, now_ms)
        except Exception as e:
            events.append({"strategy": sid, "event": "error",
                           "detail": f"data fetch failed: {e}"})
            continue
        if len(candles) < min_bars:
            events.append({"strategy": sid, "event": "error",
                           "detail": f"only {len(candles)} closed bars"})
            continue

        step = INTERVAL_MS[strat["interval"]]
        last_ts = state["last_processed"].get(sid)
        if last_ts is None:
            # First run: start from the newest bar; no history replay.
            state["last_processed"][sid] = candles[-1]["ts"] - step

        ts_list = [c["ts"] for c in candles]
        start_i = bisect_right(ts_list, state["last_processed"][sid])
        new_bars = list(range(start_i, len(candles)))

        atr = calc_atr([c["high"] for c in candles],
                       [c["low"] for c in candles],
                       [c["close"] for c in candles], _ATR_PERIOD)
        signal_set = {i: side for i, side in
                      _SIGNAL_MAP[strat["strategy"]](
                          candles, strat.get("direction", "both"))}

        pos = next((p for p in state["open_positions"]
                    if p["strategy_id"] == sid), None)

        for i in new_bars:
            bar = candles[i]
            bar_close_ts = bar["ts"] + step

            # 1. Manage the open position against this bar (stop first).
            if pos is not None:
                closed = None
                if pos["direction"] == "long":
                    if bar["low"] <= pos["stop"]:
                        closed = (min(pos["stop"], bar["open"]), "stop")
                    elif bar["high"] >= pos["target"]:
                        closed = (max(pos["target"], bar["open"]), "target")
                else:
                    if bar["high"] >= pos["stop"]:
                        closed = (max(pos["stop"], bar["open"]), "stop")
                    elif bar["low"] <= pos["target"]:
                        closed = (min(pos["target"], bar["open"]), "target")
                pos["bars_held"] += 1
                pos["mark_price"] = bar["close"]
                if closed is None and pos["bars_held"] >= pos["max_hold_bars"]:
                    closed = (bar["close"], "time")
                if closed is not None:
                    trade = _close_position(state, pos, closed[0],
                                            bar_close_ts, closed[1])
                    state["open_positions"].remove(pos)
                    pos = None
                    events.append({"strategy": sid, "event": "closed",
                                   "trade": trade})

            # 2. Entry check on this bar.
            side = signal_set.get(i)
            if side is None or pos is not None:
                continue
            if _check_kill_switch(state):
                events.append({"strategy": sid, "event": "skipped_entry",
                               "reason": "kill_switch_active"})
                continue
            if _daily_halted(state, bar_close_ts):
                events.append({"strategy": sid, "event": "skipped_entry",
                               "reason": "daily_loss_halt"})
                continue
            if strat.get("regime_filter"):
                try:
                    ok, why = _regime_allows(side, strat, bar_close_ts)
                except Exception as e:
                    ok, why = False, f"regime fetch failed: {e}"
                if not ok:
                    events.append({"strategy": sid, "event": "skipped_entry",
                                   "reason": why})
                    continue
            a = atr[i]
            if a is None or a <= 0:
                continue

            entry = bar["close"]
            risk_dist = strat["atr_mult"] * a
            if side == "long":
                stop, target = entry - risk_dist, entry + strat["rr"] * risk_dist
            else:
                stop, target = entry + risk_dist, entry - strat["rr"] * risk_dist

            risk_amount = state["capital"] * cfg["risk_pct_per_trade"] / 100
            notional = min(risk_amount * entry / risk_dist, state["capital"])
            pos = {
                "strategy_id": sid,
                "symbol": strat["symbol"],
                "direction": side,
                "entry_ts": bar["ts"],
                "entry_date": bar["date"],
                "entry_price": round(entry, 6),
                "stop": round(stop, 6),
                "target": round(target, 6),
                "notional": round(notional, 2),
                "risk_amount": round(risk_amount, 2),
                "max_hold_bars": strat["max_hold_bars"],
                "bars_held": 0,
                "mark_price": entry,
            }
            state["open_positions"].append(pos)
            events.append({"strategy": sid, "event": "opened",
                           "position": dict(pos)})

        state["last_processed"][sid] = candles[-1]["ts"]

    _check_kill_switch(state)
    _save_state(path, state)

    if not events:
        events.append({"event": "no_action",
                       "detail": "no new closed bars or signals"})
    return {
        "as_of": _date_str(now_ms),
        "events": events,
        "capital": round(state["capital"], 2),
        "peak_capital": round(state["peak_capital"], 2),
        "open_positions": state["open_positions"],
        "closed_trades_total": len(state["closed_trades"]),
        "killed": state["killed"],
        "state_path": path,
    }


# ─── Status report ────────────────────────────────────────────────────────────

def paper_status(state_path: Optional[str] = None) -> dict:
    """Account summary: capital, risk state, per-strategy performance, recent trades."""
    path = state_path or _default_state_path()
    state = _load_state(path)
    if state is None:
        return make_error(ErrorCode.NO_DATA,
                          f"No paper account at {path}. Run paper_reset first.")

    trades = state["closed_trades"]
    cfg = state["config"]
    winners = [t for t in trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)

    per_strategy = {}
    for s in cfg["strategies"]:
        st = [t for t in trades if t["strategy_id"] == s["id"]]
        if not st:
            per_strategy[s["id"]] = {"trades": 0}
            continue
        sw = [t for t in st if t["pnl"] > 0]
        per_strategy[s["id"]] = {
            "trades": len(st),
            "win_rate_pct": round(len(sw) / len(st) * 100, 1),
            "pnl": round(sum(t["pnl"] for t in st), 2),
        }

    unrealized = 0.0
    for p in state["open_positions"]:
        move = (p["mark_price"] - p["entry_price"]) / p["entry_price"]
        unrealized += p["notional"] * (move if p["direction"] == "long" else -move)

    dd = 0.0
    if state["peak_capital"] > 0:
        dd = (state["peak_capital"] - state["capital"]) / state["peak_capital"] * 100

    return {
        "state_path": path,
        "created": state["created"],
        "capital": round(state["capital"], 2),
        "initial_capital": cfg["initial_capital"],
        "total_return_pct": round(
            (state["capital"] - cfg["initial_capital"])
            / cfg["initial_capital"] * 100, 2),
        "realized_pnl": round(total_pnl, 2),
        "unrealized_pnl": round(unrealized, 2),
        "drawdown_from_peak_pct": round(-dd, 2),
        "killed": state["killed"],
        "risk_rules": {
            "risk_pct_per_trade": cfg["risk_pct_per_trade"],
            "daily_loss_halt_pct": cfg["daily_loss_halt_pct"],
            "max_drawdown_kill_pct": cfg["max_drawdown_kill_pct"],
        },
        "closed_trades": len(trades),
        "win_rate_pct": round(len(winners) / len(trades) * 100, 1) if trades else 0,
        "per_strategy": per_strategy,
        "open_positions": state["open_positions"],
        "recent_trades": trades[-10:],
        "daily": dict(list(state["daily"].items())[-7:]),
    }


# ─── CLI runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="tradingview-mcp paper trader")
    parser.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                        help="run forever, stepping every SECONDS (0 = one tick)")
    parser.add_argument("--state", default=None, help="state file path override")
    args = parser.parse_args()

    while True:
        result = paper_step(state_path=args.state)
        print(json.dumps(result, indent=1))
        if not args.loop:
            break
        time.sleep(args.loop)
