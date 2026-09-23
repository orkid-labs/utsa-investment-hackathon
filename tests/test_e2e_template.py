#!/usr/bin/env python3
"""End-to-end: boot the starter template on the fixture dataset, then
run the actual rubric checker against it. Expects 100/100 — the
starter app is the baseline every contestant builds on.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PORT = 8155
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def server(sv_fixture_root):
    env = dict(os.environ, SV_DATA_ROOT=str(sv_fixture_root))
    env.pop("SV_DATA_TOKEN", None)  # local mode — no token
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT)],
        cwd=REPO / "launchpad/template", env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"{BASE}/health", timeout=1) as r:
                    if json.loads(r.read()).get("ok"):
                        break
            except Exception:
                time.sleep(0.2)
        else:
            proc.kill()
            pytest.fail("template app did not come up")
        yield BASE
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_e2e_rubric_scores_100(server, sv_fixture_root):
    env = dict(os.environ, SV_DATA_ROOT=str(sv_fixture_root))
    out = subprocess.run(
        [sys.executable, "check.py", "--base-url", server],
        cwd=REPO / "launchpad/rubric", env=env,
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout)
    assert result["score"] == 100, json.dumps(result, indent=1)


def test_e2e_endpoints_directly(server):
    with urllib.request.urlopen(f"{server}/portfolio/holdings") as r:
        h = json.loads(r.read())
    assert h["holdings"] and all(x["weight"] >= 0 for x in h["holdings"])
    assert abs(sum(x["weight"] for x in h["holdings"]) - 1.0) < 0.01

    req = urllib.request.Request(
        f"{server}/backtest", method="POST",
        data=json.dumps({"tickers": ["AAPL", "MSFT"],
                         "start": "2020-01-02", "end": "2020-06-30"}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        bt = json.loads(r.read())
    assert {"n_days", "total_return", "sharpe", "max_drawdown"} <= set(bt)
    assert bt["n_days"] > 60


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
