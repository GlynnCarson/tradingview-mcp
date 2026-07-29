"""
Binance public klines fetcher — shared by the bracket backtester, the regime
service, and (eventually) the paper trader.

No auth required. api.binance.com returns HTTP 451 in some regions;
data-api.binance.vision is Binance's official public market-data mirror and
stays reachable there.

Pure Python — no pandas, no numpy.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

_UA = "tradingview-mcp/0.7.0 binance-data"

_BINANCE_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}
INTERVAL_MIN = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15,
    "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}

MAX_CANDLES = 30_000
_MAX_PAGES = MAX_CANDLES // 1000 + 2


def _fetch_from_host(host: str, symbol: str, interval: str,
                     start_ms: int, end_ms: int) -> list[dict]:
    step = INTERVAL_MS[interval]
    candles: list[dict] = []
    cursor = start_ms
    for _ in range(_MAX_PAGES):
        url = (f"{host}/api/v3/klines?symbol={symbol}&interval={interval}"
               f"&startTime={cursor}&endTime={end_ms}&limit=1000")
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 400 and "-1121" in body:
                raise ValueError(f"Invalid Binance symbol '{symbol}'")
            raise RuntimeError(f"HTTP {e.code} from {host}: {body[:200]}")
        if isinstance(rows, dict):
            if rows.get("code") == -1121:
                raise ValueError(f"Invalid Binance symbol '{symbol}'")
            raise RuntimeError(f"Binance error payload: {rows}")
        if not rows:
            break
        for r in rows:
            candles.append({
                "ts": r[0],
                "date": datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc)
                        .strftime("%Y-%m-%d %H:%M"),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            })
        cursor = rows[-1][0] + step
        if len(rows) < 1000 or cursor >= end_ms:
            break
    return candles


def fetch_binance_klines(symbol: str, interval: str, days: int) -> list[dict]:
    """Fetch `days` of `interval` candles for `symbol`, paginated, mirror-aware.

    Each candle: {ts (epoch ms open time), date (UTC string), open, high,
    low, close, volume}.

    Raises ValueError for an invalid symbol, RuntimeError when every host fails.
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    last_err: Optional[Exception] = None
    for host in _BINANCE_HOSTS:
        try:
            return _fetch_from_host(host, symbol, interval, start_ms, end_ms)
        except ValueError:
            raise  # invalid symbol — the mirror will reject it too
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All Binance endpoints failed: {last_err}")
