"""One paper-trader tick for CI: step the account, then rebuild the dashboard.

Run hourly by .github/workflows/paper-trader.yml. State lives in the repo at
docs/paper_state.json (override with TVMCP_PAPER_STATE); the dashboard is
written to docs/index.html for GitHub Pages.

Exits non-zero only on a hard failure (no account, render crash) so the
workflow surfaces real problems but ignores routine no-signal ticks.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

STATE_PATH = os.environ.get("TVMCP_PAPER_STATE",
                            str(REPO_ROOT / "docs" / "paper_state.json"))
DASHBOARD_PATH = str(REPO_ROOT / "docs" / "index.html")


def main() -> int:
    from tradingview_mcp.core.services.paper_trader_service import paper_step
    from tradingview_mcp.core.services.dashboard_service import write_dashboard

    result = paper_step(state_path=STATE_PATH)
    print(json.dumps(result, indent=1))
    if isinstance(result.get("error"), dict):
        print("paper_step failed", file=sys.stderr)
        return 1

    out = write_dashboard(DASHBOARD_PATH, state_path=STATE_PATH)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
