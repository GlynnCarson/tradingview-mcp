"""Unit tests for regime_service — no network access.

The regime gate decides which trades exist at all, so the classification and
the closed-bar mapping must be provably correct:
  - trend labels on synthetic up/down/flat series
  - no-lookahead property (prefix invariance)
  - signal gating: direction alignment, chop blocking, closed-bar boundary
  - get_market_regime snapshot shape and error envelopes
"""
from __future__ import annotations

import pytest

from tradingview_mcp.core.errors import is_error
import tradingview_mcp.core.services.regime_service as svc
from tradingview_mcp.core.services.regime_service import (
    compute_regime_series,
    filter_signals_by_regime,
    get_market_regime,
)

H4 = 14_400_000  # 4h in ms
H1 = 3_600_000


def candle(ts, close, spread=0.5):
    return {"ts": ts, "date": str(ts), "open": close, "high": close + spread,
            "low": close - spread, "close": close, "volume": 1.0}


def series(closes, step=H4, start_ts=0):
    return [candle(start_ts + i * step, c) for i, c in enumerate(closes)]


# ─── Classification ───────────────────────────────────────────────────────────

class TestClassification:
    def test_uptrend_downtrend_chop(self):
        n = 60
        up = compute_regime_series(series([100 + 2 * i for i in range(n)]),
                                   fast=5, slow=20)
        down = compute_regime_series(series([300 - 2 * i for i in range(n)]),
                                     fast=5, slow=20)
        flat = compute_regime_series(series([100.0] * n), fast=5, slow=20)
        assert up[-1]["trend"] == "up"
        assert down[-1]["trend"] == "down"
        assert flat[-1]["trend"] == "chop"

    def test_warmup_is_none(self):
        regimes = compute_regime_series(series([100.0] * 30), fast=5, slow=20)
        assert regimes[0] is None
        assert regimes[-1] is not None

    def test_no_lookahead_prefix_invariance(self):
        # Regime at bar i must not change when future bars are appended.
        closes = [100 + (i % 7) + i * 0.3 for i in range(80)]
        full = compute_regime_series(series(closes), fast=5, slow=20)
        prefix = compute_regime_series(series(closes[:50]), fast=5, slow=20)
        for i in range(50):
            if prefix[i] is None:
                assert full[i] is None
            else:
                assert full[i]["trend"] == prefix[i]["trend"]


# ─── Signal gating ────────────────────────────────────────────────────────────

def _anchor_with_regimes(trends):
    """Anchor candles at 4h spacing + hand-built regime list (bypasses EMA)."""
    anchor = series([100.0] * len(trends), step=H4, start_ts=0)
    regimes = [{"trend": t, "vol": "low", "ema_fast": 1, "ema_slow": 1,
                "atr_pct": 0.01} if t else None for t in trends]
    return anchor, regimes


class TestGating:
    def test_direction_alignment_and_chop(self):
        # Anchor: bar0 up, bar1 down, bar2 chop (each 4h).
        anchor, regimes = _anchor_with_regimes(["up", "down", "chop"])
        # 1h trade bars; decision time = ts + 1h. Bar at ts=4h+1h → decision
        # 4h+2h → last closed anchor bar is bar0 (closes 4h)... bar1 closes 8h.
        trade = series([100.0] * 13, step=H1, start_ts=0)
        signals = [
            (4, "long"),   # decision 5h → anchor bar0 (up) → keep long
            (5, "short"),  # decision 6h → anchor bar0 (up) → drop counter
            (8, "short"),  # decision 9h → anchor bar1 (down) → keep short
            (9, "long"),   # decision 10h → anchor bar1 (down) → drop counter
            (11, "long"),  # decision 12h → anchor bar2 (chop) → drop chop
        ]
        kept, dropped = filter_signals_by_regime(signals, trade, "1h",
                                                 anchor, regimes)
        assert kept == [(4, "long"), (8, "short")]
        assert dropped == {"chop": 1, "counter_trend": 2, "no_regime_data": 0}

    def test_only_closed_anchor_bars_are_used(self):
        # Single anchor bar [0, 4h) labelled "up". A signal whose decision
        # time is BEFORE 4h must not see it (bar not closed yet).
        anchor, regimes = _anchor_with_regimes(["up", "up"])
        trade = series([100.0] * 9, step=H1, start_ts=0)
        early = [(1, "long")]   # decision 2h < first close 4h
        late = [(4, "long")]    # decision 5h ≥ 4h
        kept_e, dropped_e = filter_signals_by_regime(early, trade, "1h",
                                                     anchor, regimes)
        kept_l, _ = filter_signals_by_regime(late, trade, "1h",
                                             anchor, regimes)
        assert kept_e == [] and dropped_e["no_regime_data"] == 1
        assert kept_l == late

    def test_none_regime_drops_signal(self):
        anchor, regimes = _anchor_with_regimes([None, "up"])
        trade = series([100.0] * 9, step=H1, start_ts=0)
        # decision 5h → anchor bar0 closed but regime None (warmup).
        kept, dropped = filter_signals_by_regime([(4, "long")], trade, "1h",
                                                 anchor, regimes)
        assert kept == [] and dropped["no_regime_data"] == 1


# ─── Snapshot tool ────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_uptrend_snapshot(self, monkeypatch):
        candles = series([100 + i for i in range(260)], step=H4)
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            lambda *a, **k: candles)
        result = get_market_regime("BTCUSDT", "4h", 180)
        assert not is_error(result)
        assert result["current"]["trend"] == "up"
        assert result["allowed_directions"] == ["long"]
        assert result["last_30_bars"]["up"] == 30
        assert result["recent_transitions"][-1]["trend"] == "up"

    def test_validation_and_no_data(self, monkeypatch):
        monkeypatch.setattr(svc, "fetch_binance_klines",
                            lambda *a, **k: series([100.0] * 50, step=H4))
        assert get_market_regime("BTCUSDT", "2h")["error"]["code"] == \
            "INVALID_PARAMETER"
        assert get_market_regime("BTCUSDT", "4h", 999)["error"]["code"] == \
            "INVALID_PARAMETER"
        # 50 bars < EMA200 warmup → NO_DATA
        assert get_market_regime("BTCUSDT", "4h", 30)["error"]["code"] == \
            "NO_DATA"
