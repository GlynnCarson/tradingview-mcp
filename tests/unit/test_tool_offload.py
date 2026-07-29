"""Every tool must be non-blocking (issue #77).

mcp 1.12.x runs sync tool functions bare on the event loop, so any sync tool
stalls every concurrent session for its full duration. server.py wraps every
still-sync tool in asyncio.to_thread at import time; these tests pin that
blanket guarantee so a future tool can't silently reintroduce a blocker.
"""
import asyncio
import inspect

from tradingview_mcp import server
from tradingview_mcp.server import mcp


def _tools():
    return mcp._tool_manager.list_tools()


def test_every_tool_is_async_after_offload():
    still_sync = [t.name for t in _tools() if not t.is_async or not inspect.iscoroutinefunction(t.fn)]
    assert still_sync == [], f"tools still blocking the event loop: {still_sync}"
    assert server._OFFLOADED_TOOL_COUNT >= 25  # the historical sync set


def test_wrapped_tool_still_returns_the_original_result(monkeypatch):
    # top_losers was the poster child in #77 (sync twin of async top_gainers).
    monkeypatch.setattr(server, "fetch_trending_analysis",
                        lambda exchange, timeframe, limit, ascending=True: [{"symbol": "X", "changePercent": -3.2, "indicators": {"RSI": 41.0}}])
    tool = next(t for t in _tools() if t.name == "top_losers")
    out = asyncio.run(tool.fn(exchange="KUCOIN", timeframe="15m", limit=5))
    assert isinstance(out, (list, dict))
    text = str(out)
    assert "X" in text and "RSI" in text, f"unexpected result shape: {text[:120]}"


def test_offload_preserves_annotations_and_metadata():
    # paper_reset/paper_step honestly declare readOnlyHint=False (they mutate
    # the local simulation state file) — see test_tool_annotations.py.
    local_state_tools = {"paper_reset", "paper_step"}
    for t in _tools():
        assert t.annotations is not None and t.annotations.title
        expected_ro = t.name not in local_state_tools
        assert t.annotations.readOnlyHint is expected_ro
        assert t.fn_metadata is not None  # validation schema still from the original signature
