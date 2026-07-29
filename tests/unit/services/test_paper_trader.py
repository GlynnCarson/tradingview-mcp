"""Unit tests for paper_trader_service — no network, state in tmp files.

The paper trader is the last gate before real money, so what must be provably
correct: position sizing off the ATR stop, stop-first exit handling, PnL and
capital compounding, the daily-loss halt, the drawdown kill switch, the
closed-bars-only rule, and the regime gate at entry time.
"""
from __future__ import annotations

import pytest

from tradingview_mcp.core.errors import is_error
import tradingview_mcp.core.services.paper_trader_service as svc
from tradingview_mcp.core.services.paper_trader_service import (
    paper_reset,
    paper_status,
    paper_step,
)

H1 = 3_600_000
H4 = 14_400_000


def bar(ts, o, h, l, c):
    return {"ts": ts, "date": f"t{ts // H1}", "open": o, "high": h,
            "low": l, "close": c, "volume": 1.0}


def quiet_1h(n, base=100.0):
    out = []
    for i in range(n):
        c = base + (0.1 if i % 2 == 0 else -0.1)
        out.append(bar(i * H1, c, c + 0.15, c - 0.15, c))
    return out


def one_strategy(**over):
    s = {"id": "test_squeeze", "symbol": "BTCUSDT",
         "strategy": "squeeze_breakout", "interval": "1h",
         "direction": "both", "rr": 1.5, "atr_mult": 2.0,
         "max_hold_bars": 60, "regime_filter": False}
    s.update(over)
    return [s]


def make_account(tmp_path, **over):
    path = str(tmp_path / "paper.json")
    kwargs = dict(confirm=True, strategies=one_strategy(), state_path=path)
    kwargs.update(over)
    result = paper_reset(**kwargs)
    assert result.get("status") == "created", result
    return path


def force_signal_on_last_bar(monkeypatch):
    monkeypatch.setitem(svc._SIGNAL_MAP, "squeeze_breakout",
                        lambda candles, direction: [(len(candles) - 1, "long")])


# ─── Lifecycle & guards ───────────────────────────────────────────────────────

class TestLifecycle:
    def test_reset_guard(self, tmp_path):
        path = make_account(tmp_path)
        again = paper_reset(state_path=path)
        assert is_error(again)
        assert "confirm" in again["error"]["message"]
        assert paper_reset(confirm=True, state_path=path)["status"] == "created"

    def test_step_without_account(self, tmp_path):
        result = paper_step(state_path=str(tmp_path / "missing.json"))
        assert is_error(result)
        assert result["error"]["code"] == "NO_DATA"


# ─── Entries ──────────────────────────────────────────────────────────────────

class TestEntry:
    def test_real_squeeze_breakout_opens_long(self, tmp_path, monkeypatch):
        candles = quiet_1h(200)
        ts = 200 * H1
        candles.append(bar(ts, 102.6, 103.2, 102.5, 103.0))  # breakout bar
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            lambda *a, **k: candles)
        path = make_account(tmp_path)
        result = paper_step(state_path=path, now_ms=ts + H1)
        opened = [e for e in result["events"] if e["event"] == "opened"]
        assert len(opened) == 1
        pos = opened[0]["position"]
        assert pos["direction"] == "long"
        assert pos["entry_price"] == pytest.approx(103.0)
        assert pos["stop"] < 103.0 < pos["target"]
        # rr=1.5: reward distance = 1.5 × risk distance.
        assert (pos["target"] - 103.0) == pytest.approx(
            1.5 * (103.0 - pos["stop"]), rel=1e-6)
        # 1% risk sizing: notional = risk_amount × entry / stop_distance,
        # capped at capital.
        risk_dist = 103.0 - pos["stop"]
        expected = min(100.0 * 103.0 / risk_dist, 10_000.0)
        assert pos["notional"] == pytest.approx(expected, rel=1e-4)

    def test_in_progress_candle_is_ignored(self, tmp_path, monkeypatch):
        candles = quiet_1h(200)
        ts = 200 * H1
        candles.append(bar(ts, 102.6, 103.2, 102.5, 103.0))
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            lambda *a, **k: candles)
        path = make_account(tmp_path)
        # now is 1ms BEFORE the breakout bar closes → bar must be invisible.
        result = paper_step(state_path=path, now_ms=ts + H1 - 1)
        assert not any(e["event"] == "opened" for e in result["events"])


# ─── Exits & PnL ─────────────────────────────────────────────────────────────

class TestExitAndPnl:
    def test_stop_hit_realizes_loss_and_compounds_capital(self, tmp_path,
                                                          monkeypatch):
        force_signal_on_last_bar(monkeypatch)
        candles = quiet_1h(200)
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            lambda *a, **k: list(candles))
        path = make_account(tmp_path)
        r1 = paper_step(state_path=path, now_ms=200 * H1)
        pos = [e for e in r1["events"] if e["event"] == "opened"][0]["position"]

        # Next bar crashes straight through the stop (opens below it).
        crash_open = pos["stop"] - 1.0
        candles.append(bar(200 * H1, crash_open, crash_open + 0.2,
                           crash_open - 1.0, crash_open - 0.5))
        r2 = paper_step(state_path=path, now_ms=201 * H1)
        closed = [e for e in r2["events"] if e["event"] == "closed"]
        assert len(closed) == 1
        trade = closed[0]["trade"]
        assert trade["exit_reason"] == "stop"
        # Gap through the stop → filled at the bar's open, not the stop price.
        assert trade["exit_price"] == pytest.approx(crash_open)
        # Hand-computed PnL must match capital movement exactly.
        gross = (crash_open - trade["entry_price"]) / trade["entry_price"] * 100
        net = gross - 2 * (0.05 + 0.02)
        expected_pnl = trade["notional"] * net / 100
        assert trade["pnl"] == pytest.approx(expected_pnl, abs=0.01)
        assert r2["capital"] == pytest.approx(10_000 + expected_pnl, abs=0.01)

    def test_time_stop(self, tmp_path, monkeypatch):
        force_signal_on_last_bar(monkeypatch)
        candles = quiet_1h(200)
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            lambda *a, **k: list(candles))
        path = make_account(tmp_path, strategies=one_strategy(max_hold_bars=2,
                                                              atr_mult=9.9))
        paper_step(state_path=path, now_ms=200 * H1)
        # Two flat bars later the time-stop must fire (brackets far away).
        candles.append(bar(200 * H1, 100, 100.2, 99.8, 100))
        candles.append(bar(201 * H1, 100, 100.2, 99.8, 100))
        r = paper_step(state_path=path, now_ms=202 * H1)
        closed = [e for e in r["events"] if e["event"] == "closed"]
        assert closed and closed[0]["trade"]["exit_reason"] == "time"


# ─── Risk rules ───────────────────────────────────────────────────────────────

class TestRiskRules:
    def _blow_up(self, tmp_path, monkeypatch, **reset_over):
        """Open, then crash through the stop; forced signal on the crash bar
        immediately attempts re-entry in the same step."""
        force_signal_on_last_bar(monkeypatch)
        candles = quiet_1h(200)
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            lambda *a, **k: list(candles))
        path = make_account(tmp_path, **reset_over)
        r1 = paper_step(state_path=path, now_ms=200 * H1)
        pos = [e for e in r1["events"] if e["event"] == "opened"][0]["position"]
        crash = pos["stop"] - 3.0
        candles.append(bar(200 * H1, crash, crash + 0.2, crash - 1, crash))
        return path, paper_step(state_path=path, now_ms=201 * H1)

    def test_daily_loss_halt_blocks_reentry(self, tmp_path, monkeypatch):
        path, r = self._blow_up(tmp_path, monkeypatch,
                                daily_loss_halt_pct=0.01,
                                max_drawdown_kill_pct=99.0)
        reasons = [e.get("reason") for e in r["events"]
                   if e["event"] == "skipped_entry"]
        assert "daily_loss_halt" in reasons
        assert r["killed"] is False

    def test_kill_switch_freezes_account(self, tmp_path, monkeypatch):
        path, r = self._blow_up(tmp_path, monkeypatch,
                                daily_loss_halt_pct=99.0,
                                max_drawdown_kill_pct=0.01)
        reasons = [e.get("reason") for e in r["events"]
                   if e["event"] == "skipped_entry"]
        assert "kill_switch_active" in reasons
        assert r["killed"] is True
        status = paper_status(state_path=path)
        assert status["killed"] is True


# ─── Regime gate at entry ────────────────────────────────────────────────────

class TestRegimeGateEntry:
    def _fetch_branching(self, anchor_closes):
        trade = quiet_1h(200)

        def fetch(symbol, interval, days):
            if interval == "4h":
                start = -230 * H4
                return [bar(start + i * H4, c, c + 0.5, c - 0.5, c)
                        for i, c in enumerate(anchor_closes)]
            return list(trade)
        return fetch

    def test_chop_anchor_blocks_entry(self, tmp_path, monkeypatch):
        force_signal_on_last_bar(monkeypatch)
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            self._fetch_branching([100.0] * 240))
        path = make_account(tmp_path, strategies=one_strategy(
            regime_filter=True, regime_anchor="BTCUSDT",
            regime_interval="4h"))
        r = paper_step(state_path=path, now_ms=200 * H1)
        reasons = [e.get("reason") for e in r["events"]
                   if e["event"] == "skipped_entry"]
        assert "regime_chop" in reasons
        assert not r["open_positions"]

    def test_uptrend_anchor_allows_long(self, tmp_path, monkeypatch):
        force_signal_on_last_bar(monkeypatch)
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            self._fetch_branching(
                                [100 + i for i in range(240)]))
        path = make_account(tmp_path, strategies=one_strategy(
            regime_filter=True, regime_anchor="BTCUSDT",
            regime_interval="4h"))
        r = paper_step(state_path=path, now_ms=200 * H1)
        assert any(e["event"] == "opened" for e in r["events"])


# ─── Status ───────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_after_trading(self, tmp_path, monkeypatch):
        force_signal_on_last_bar(monkeypatch)
        candles = quiet_1h(200)
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            lambda *a, **k: list(candles))
        path = make_account(tmp_path)
        paper_step(state_path=path, now_ms=200 * H1)
        status = paper_status(state_path=path)
        assert status["capital"] == pytest.approx(10_000.0)
        assert len(status["open_positions"]) == 1
        assert status["per_strategy"]["test_squeeze"]["trades"] == 0
        assert status["killed"] is False
        assert status["risk_rules"]["risk_pct_per_trade"] == 1.0
