"""Unit tests for bracket_backtest_service — no network access.

Covers the parts where money math can silently go wrong:
  - bracket simulation (stop-first conservatism, gap fills, time-stop, shorts,
    segment bounds used by walk-forward)
  - per-side cost application for both directions
  - input validation envelopes (must reject before any fetch)
  - end-to-end pipeline, sweep, and walk-forward with a monkeypatched fetch
"""
from __future__ import annotations

import pytest

from tradingview_mcp.core.errors import is_error
import tradingview_mcp.core.services.bracket_backtest_service as svc
from tradingview_mcp.core.services.bracket_backtest_service import (
    _apply_bracket_costs,
    _signals_ema_momentum,
    _simulate_bracket,
    run_bracket_backtest,
    run_bracket_sweep,
    run_bracket_walk_forward,
)


def bar(o, h, l, c, date="2026-01-01 00:00", ts=0):
    return {"ts": ts, "date": date, "open": o, "high": h, "low": l, "close": c,
            "volume": 1.0}


def flat_atr(candles, value=1.0):
    return [value] * len(candles)


# ─── Bracket simulation ───────────────────────────────────────────────────────

class TestBracket:
    def test_long_target_hit(self):
        candles = [
            bar(100, 100.5, 99.5, 100),   # 0: signal bar, entry 100
            bar(100, 101.0, 99.8, 100.8),  # 1: high 101 < target 101.5
            bar(100.8, 102.0, 100.5, 101.9),  # 2: high >= 101.5 → target
        ]
        trades = _simulate_bracket(candles, [(0, "long")], flat_atr(candles),
                                   atr_mult=1.0, rr=1.5, max_hold_bars=10)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "target"
        assert t["exit_price"] == pytest.approx(101.5)
        assert t["bars_held"] == 2

    def test_stop_first_when_bar_hits_both(self):
        # Bar 1 spans both the stop (99) and target (101.5): stop must win.
        candles = [
            bar(100, 100.5, 99.5, 100),
            bar(100, 102.0, 98.0, 101.0),
        ]
        trades = _simulate_bracket(candles, [(0, "long")], flat_atr(candles),
                                   atr_mult=1.0, rr=1.5, max_hold_bars=10)
        assert trades[0]["exit_reason"] == "stop"
        assert trades[0]["exit_price"] == pytest.approx(99.0)

    def test_gap_through_stop_fills_at_open(self):
        # Long stop at 99, next bar opens at 97 → fill at 97, not 99.
        candles = [
            bar(100, 100.5, 99.5, 100),
            bar(97.0, 97.5, 96.0, 96.5),
        ]
        trades = _simulate_bracket(candles, [(0, "long")], flat_atr(candles),
                                   atr_mult=1.0, rr=1.5, max_hold_bars=10)
        assert trades[0]["exit_reason"] == "stop"
        assert trades[0]["exit_price"] == pytest.approx(97.0)

    def test_short_target_and_stop_sides(self):
        # Short entry 100: stop 101, target 98.5.
        candles = [
            bar(100, 100.5, 99.5, 100),
            bar(100, 100.9, 98.0, 98.2),  # low <= 98.5 → target
        ]
        trades = _simulate_bracket(candles, [(0, "short")], flat_atr(candles),
                                   atr_mult=1.0, rr=1.5, max_hold_bars=10)
        assert trades[0]["exit_reason"] == "target"
        assert trades[0]["exit_price"] == pytest.approx(98.5)

        candles2 = [
            bar(100, 100.5, 99.5, 100),
            bar(100, 101.5, 99.9, 101.2),  # high >= 101 → stop
        ]
        trades2 = _simulate_bracket(candles2, [(0, "short")], flat_atr(candles2),
                                    atr_mult=1.0, rr=1.5, max_hold_bars=10)
        assert trades2[0]["exit_reason"] == "stop"
        assert trades2[0]["exit_price"] == pytest.approx(101.0)

    def test_time_stop(self):
        candles = [bar(100, 100.4, 99.6, 100) for _ in range(6)]
        trades = _simulate_bracket(candles, [(0, "long")], flat_atr(candles),
                                   atr_mult=1.0, rr=1.5, max_hold_bars=3)
        assert trades[0]["exit_reason"] == "time"
        assert trades[0]["bars_held"] == 3

    def test_no_overlapping_positions(self):
        candles = [bar(100, 100.4, 99.6, 100) for _ in range(10)]
        signals = [(0, "long"), (1, "long"), (2, "long"), (4, "long")]
        trades = _simulate_bracket(candles, signals, flat_atr(candles),
                                   atr_mult=1.0, rr=1.5, max_hold_bars=3)
        # Trade 1 holds bars 0→3, so signals at 1 and 2 are skipped;
        # the signal at 4 re-enters after the position is free.
        assert len(trades) == 2
        assert trades[0]["entry_date"] == candles[0]["date"]
        assert trades[1]["bars_held"] == 3

    def test_skips_zero_atr_and_last_bar(self):
        candles = [bar(100, 100.4, 99.6, 100) for _ in range(3)]
        atr = [0.0, 1.0, 1.0]
        trades = _simulate_bracket(candles, [(0, "long"), (2, "long")], atr,
                                   atr_mult=1.0, rr=1.5, max_hold_bars=3)
        assert trades == []

    def test_segment_bounds_force_close_and_filter(self):
        candles = [bar(100, 100.4, 99.6, 100) for _ in range(8)]
        signals = [(0, "long"), (5, "long")]
        # end_index=4 → the bar-5 signal is outside the segment; the bar-0
        # trade is force-closed at bar 3 (last bar inside the segment).
        trades = _simulate_bracket(candles, signals, flat_atr(candles),
                                   atr_mult=1.0, rr=1.5, max_hold_bars=10,
                                   start_index=0, end_index=4)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "end_of_data"
        assert trades[0]["bars_held"] == 3
        # start_index=5 → only the bar-5 signal is eligible.
        trades2 = _simulate_bracket(candles, signals, flat_atr(candles),
                                    atr_mult=1.0, rr=1.5, max_hold_bars=10,
                                    start_index=5)
        assert len(trades2) == 1
        assert trades2[0]["entry_date"] == candles[5]["date"]


# ─── Cost application ─────────────────────────────────────────────────────────

class TestCosts:
    def test_long_and_short_net_returns(self):
        trades = [
            {"entry_price": 100.0, "exit_price": 101.0, "direction": "long"},
            {"entry_price": 100.0, "exit_price": 99.0, "direction": "short"},
            {"entry_price": 100.0, "exit_price": 101.0, "direction": "short"},
        ]
        out = _apply_bracket_costs(trades, fee_pct=0.05, slippage_pct=0.02)
        # Round trip = 2 * (0.05 + 0.02) = 0.14
        assert out[0]["gross_return_pct"] == pytest.approx(1.0)
        assert out[0]["return_pct"] == pytest.approx(0.86)
        assert out[1]["gross_return_pct"] == pytest.approx(1.0)  # short profits on drop
        assert out[1]["return_pct"] == pytest.approx(0.86)
        assert out[2]["gross_return_pct"] == pytest.approx(-1.0)  # short loses on rise
        assert out[2]["return_pct"] == pytest.approx(-1.14)


# ─── Validation (must not touch the network) ─────────────────────────────────

class TestValidation:
    def _no_fetch(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("network fetch must not be reached")
        monkeypatch.setattr(svc, "_fetch_binance_klines", boom)

    @pytest.mark.parametrize("kwargs", [
        {"strategy": "hodl"},
        {"interval": "30s"},
        {"direction": "sideways"},
        {"days": 0},
        {"days": 9999},
        {"rr": 0.05},
        {"atr_mult": 50},
        {"max_hold_bars": 1},
        {"fee_pct": -1},
        {"initial_capital": 0},
        {"interval": "1m", "days": 30},  # 43200 bars > 30k cap
        {"interval": "1d", "days": 30},  # 30 bars < squeeze warmup (150)
    ])
    def test_bad_inputs_return_envelope(self, monkeypatch, kwargs):
        self._no_fetch(monkeypatch)
        result = run_bracket_backtest("BTCUSDT", **kwargs)
        assert is_error(result)
        assert result["error"]["code"] == "INVALID_PARAMETER"

    @pytest.mark.parametrize("kwargs", [
        {"rr_values": []},
        {"rr_values": [1.0] * 9},
        {"atr_mult_values": [0.05]},
    ])
    def test_bad_grids_return_envelope(self, monkeypatch, kwargs):
        self._no_fetch(monkeypatch)
        result = run_bracket_sweep("BTCUSDT", **kwargs)
        assert is_error(result)
        assert result["error"]["code"] == "INVALID_PARAMETER"

    @pytest.mark.parametrize("kwargs", [
        {"n_splits": 1},
        {"train_ratio": 0.3},
        {"min_trades": 0},
    ])
    def test_bad_walk_forward_params(self, monkeypatch, kwargs):
        self._no_fetch(monkeypatch)
        result = run_bracket_walk_forward("BTCUSDT", **kwargs)
        assert is_error(result)
        assert result["error"]["code"] == "INVALID_PARAMETER"


# ─── Signals ──────────────────────────────────────────────────────────────────

class TestSignals:
    def test_ema_momentum_long_only_filter(self):
        # Rising series → any crosses that occur must respect direction filter.
        closes = ([100 - i * 0.5 for i in range(30)]
                  + [85 + i * 0.8 for i in range(60)])
        candles = [bar(c, c + 0.1, c - 0.1, c) for c in closes]
        both = _signals_ema_momentum(candles, "both", fast=3, slow=7, trend=20)
        longs = _signals_ema_momentum(candles, "long", fast=3, slow=7, trend=20)
        assert all(s in both for s in longs)
        assert all(side == "long" for _, side in longs)


# ─── End-to-end with fake data ───────────────────────────────────────────────

def _quiet_block(n, start_minute=0):
    """Tight oscillation around 100 — arms the squeeze."""
    out = []
    for i in range(n):
        c = 100.0 + (0.1 if i % 2 == 0 else -0.1)
        m = start_minute + i
        out.append(bar(c, c + 0.15, c - 0.15, c,
                       date=f"2026-01-{m // 1440 + 1:02d} "
                            f"{m % 1440 // 60:02d}:{m % 60:02d}",
                       ts=m * 60_000))
    return out


def _run_block(n, start_minute=0):
    """Hard upward breakout — close jumps above the squeezed upper band."""
    out = []
    price = 103.0
    for i in range(n):
        m = start_minute + i
        out.append(bar(price - 0.4, price + 0.2, price - 0.5, price,
                       date=f"2026-01-{m // 1440 + 1:02d} "
                            f"{m % 1440 // 60:02d}:{m % 60:02d}",
                       ts=m * 60_000))
        price += 0.6
    return out


def _squeeze_then_breakout(n_quiet=320, n_run=30):
    return _quiet_block(n_quiet) + _run_block(n_run, start_minute=n_quiet)


def _repeating_breakouts(blocks=6, quiet=100, run=20):
    candles = []
    for b in range(blocks):
        start = b * (quiet + run)
        candles += _quiet_block(quiet, start_minute=start)
        candles += _run_block(run, start_minute=start + quiet)
    return candles


class TestEndToEnd:
    def test_pipeline_produces_trades_and_metrics(self, monkeypatch):
        monkeypatch.setattr(svc, "_fetch_binance_klines",
                            lambda *a, **k: _squeeze_then_breakout())
        result = run_bracket_backtest("BTCUSDT", strategy="squeeze_breakout",
                                      interval="5m", days=2,
                                      include_trade_log=True)
        assert not is_error(result)
        assert result["strategy"] == "squeeze_breakout"
        assert result["total_trades"] >= 1
        assert result["costs"]["breakeven_move_pct"] == pytest.approx(0.14)
        # The breakout run is upward, so the first trade should be a winning long.
        first = result["trade_log"][0]
        assert first["direction"] == "long"
        assert first["return_pct"] > 0

    def test_short_data_returns_no_data_envelope(self, monkeypatch):
        monkeypatch.setattr(svc, "_fetch_binance_klines",
                            lambda *a, **k: [bar(100, 101, 99, 100)] * 50)
        result = run_bracket_backtest("BTCUSDT", interval="5m", days=2)
        assert is_error(result)
        assert result["error"]["code"] == "NO_DATA"

    def test_invalid_symbol_maps_to_symbol_not_found(self, monkeypatch):
        def raise_invalid(*a, **k):
            raise ValueError("Invalid Binance symbol 'NOPEUSDT'")
        monkeypatch.setattr(svc, "_fetch_binance_klines", raise_invalid)
        result = run_bracket_backtest("NOPEUSDT", interval="5m", days=2)
        assert is_error(result)
        assert result["error"]["code"] == "SYMBOL_NOT_FOUND"


class TestSweep:
    def test_sweep_reports_full_grid_and_stability(self, monkeypatch):
        monkeypatch.setattr(svc, "_fetch_binance_klines",
                            lambda *a, **k: _squeeze_then_breakout())
        result = run_bracket_sweep("BTCUSDT", strategy="squeeze_breakout",
                                   interval="5m", days=2,
                                   rr_values=[1.0, 2.0],
                                   atr_mult_values=[1.0, 2.0, 3.0])
        assert not is_error(result)
        assert result["grid"]["combos_tested"] == 6
        assert len(result["results"]) == 6
        # Results are ranked best-first and internal grid indices are stripped.
        returns = [c["total_return_pct"] for c in result["results"]]
        assert returns == sorted(returns, reverse=True)
        assert "_ri" not in result["results"][0]
        assert set(result["best"]) >= {"rr", "atr_mult", "total_return_pct"}
        assert result["best_neighborhood"]["neighbors_checked"] >= 2
        assert any(word in result["verdict"]
                   for word in ("STABLE_CANDIDATE", "FRAGILE", "NO_EDGE"))


class TestWalkForward:
    def test_walk_forward_folds_and_verdict(self, monkeypatch):
        monkeypatch.setattr(svc, "_fetch_binance_klines",
                            lambda *a, **k: _repeating_breakouts(blocks=6))
        result = run_bracket_walk_forward("BTCUSDT", strategy="squeeze_breakout",
                                          interval="5m", days=3,
                                          direction="long", n_splits=2,
                                          train_ratio=0.7, min_trades=1,
                                          rr_values=[1.0, 2.0],
                                          atr_mult_values=[1.0, 2.0])
        assert not is_error(result)
        assert len(result["folds"]) == 2
        assert result["folds_evaluated"] >= 1
        evaluated = [f for f in result["folds"] if "skipped" not in f]
        for f in evaluated:
            assert set(f["best_params"]) == {"rr", "atr_mult"}
            # Test window must start after the train window ends.
            assert f["test_from"] > f["train_to"]
        assert "oos_total_return_pct" in result
        assert any(word in result["verdict"]
                   for word in ("PASSED", "MIXED", "OVERFIT", "NO_EDGE"))
        assert result["param_stability"].split(" ")[0] in (
            "STABLE", "MODERATE", "UNSTABLE")

    def test_all_folds_skipped_returns_envelope(self, monkeypatch):
        # Pure quiet data → no breakouts → no trades in any train window.
        monkeypatch.setattr(svc, "_fetch_binance_klines",
                            lambda *a, **k: _quiet_block(600))
        result = run_bracket_walk_forward("BTCUSDT", strategy="squeeze_breakout",
                                          interval="5m", days=3, n_splits=2,
                                          min_trades=5)
        assert is_error(result)
        assert result["error"]["code"] == "NO_DATA"


# ─── Regime gate integration ─────────────────────────────────────────────────

_H4 = 14_400_000


def _anchor_series(closes):
    """4h anchor candles starting far enough back for EMA200 warmup to finish
    before the trade window (trade ts start at 0)."""
    start = -210 * _H4
    return [bar(c, c + 0.5, c - 0.5, c, ts=start + i * _H4)
            for i, c in enumerate(closes)]


def _branching_fetch(anchor_closes):
    trade = _squeeze_then_breakout()

    def fetch(symbol, interval, days):
        if interval == "4h":
            return _anchor_series(anchor_closes)
        return trade
    return fetch


class TestRegimeGate:
    def test_uptrend_anchor_lets_longs_through(self, monkeypatch):
        # Steadily rising anchor → trend "up" → the breakout longs survive.
        monkeypatch.setattr(svc, "_fetch_binance_klines",
                            _branching_fetch([100 + i for i in range(220)]))
        result = run_bracket_backtest("ETHUSDT", strategy="squeeze_breakout",
                                      interval="5m", days=2,
                                      regime_filter=True)
        assert not is_error(result)
        gate = result["regime_gate"]
        assert gate["enabled"] is True
        assert gate["anchor"] == "BTCUSDT"
        assert gate["signals_after"] >= 1
        assert result["total_trades"] >= 1

    def test_chop_anchor_blocks_everything(self, monkeypatch):
        # Flat anchor → trend "chop" → every signal is gated off.
        monkeypatch.setattr(svc, "_fetch_binance_klines",
                            _branching_fetch([100.0] * 220))
        result = run_bracket_backtest("ETHUSDT", strategy="squeeze_breakout",
                                      interval="5m", days=2,
                                      regime_filter=True)
        assert not is_error(result)
        gate = result["regime_gate"]
        assert gate["signals_before"] >= 1
        assert gate["signals_after"] == 0
        assert gate["dropped"]["chop"] >= 1
        assert result["total_trades"] == 0

    def test_self_anchor_uses_traded_symbol(self, monkeypatch):
        seen = []

        def fetch(symbol, interval, days):
            seen.append((symbol, interval))
            if interval == "4h":
                return _anchor_series([100 + i for i in range(220)])
            return _squeeze_then_breakout()
        monkeypatch.setattr(svc, "_fetch_binance_klines", fetch)
        result = run_bracket_backtest("ETHUSDT", strategy="squeeze_breakout",
                                      interval="5m", days=2,
                                      regime_filter=True, regime_anchor="self")
        assert not is_error(result)
        assert ("ETHUSDT", "4h") in seen

    def test_gate_disabled_by_default(self, monkeypatch):
        monkeypatch.setattr(svc, "_fetch_binance_klines",
                            lambda *a, **k: _squeeze_then_breakout())
        result = run_bracket_backtest("BTCUSDT", interval="5m", days=2)
        assert result["regime_gate"] == {"enabled": False}

    def test_bad_regime_interval_rejected(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("must not fetch")
        monkeypatch.setattr(svc, "_fetch_binance_klines", boom)
        result = run_bracket_backtest("BTCUSDT", interval="5m", days=2,
                                      regime_filter=True, regime_interval="2h")
        assert is_error(result)
        assert result["error"]["code"] == "INVALID_PARAMETER"
