#!/usr/bin/env python3
"""Remote smoke tests — exercise the thin-client path the way a
contestant hits it. Skipped unless SV_TEST_BASE + SV_TEST_TOKEN are set.

Run:
    SV_TEST_BASE=https://pop-os.tail01ad.ts.net \
    SV_TEST_TOKEN=$(cat /tmp/.sv_test_token) \
    python -m pytest tests/test_remote_smoke.py -x -q
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

BASE = os.environ.get("SV_TEST_BASE")
TOKEN = os.environ.get("SV_TEST_TOKEN")

pytestmark = pytest.mark.skipif(
    not (BASE and TOKEN), reason="SV_TEST_BASE/SV_TEST_TOKEN not set"
)


@pytest.fixture(scope="module")
def ds():
    from statevector import Dataset
    return Dataset(BASE, token=TOKEN)


def _dates(df):
    """Extract a date column as date objects regardless of frame flavor."""
    col = df["date"] if "date" in df.columns else df["Date"]
    try:
        return [d.date() if hasattr(d, "date") else d for d in col.to_list()]
    except AttributeError:
        return [d.date() if hasattr(d, "date") else d for d in col]


def test_universe_size(ds):
    uni = ds.universe()
    assert len(uni) > 1000
    assert "AAPL" in uni


def test_prices_bounded(ds):
    df = ds.prices("AAPL", start="2020-01-01", end="2020-01-31")
    assert len(df) >= 15  # ~21 trading days
    cols = set(df.columns)
    assert {"date", "close"} <= cols or {"date", "Close"} <= cols
    ds_dates = _dates(df)
    assert min(ds_dates).isoformat() >= "2020-01-01"
    assert max(ds_dates).isoformat() <= "2020-01-31"


def test_fundamentals_asof(ds):
    df = ds.fundamentals("AAPL", asof="2024-03-31")
    assert len(df) > 0


def test_asof_no_lookahead(ds):
    df = ds.asof("AAPL", on="2024-03-31")
    assert len(df) > 0
    for d in _dates(df):
        assert d.isoformat() <= "2024-03-31"


def test_state_vector_columns(ds):
    from statevector.pca import NUMERIC
    df = ds.state_vector("AAPL", start="2024-01-02", end="2024-01-10")
    assert len(df) > 0
    missing = [c for c in NUMERIC if c not in df.columns]
    assert not missing, f"missing numeric features: {missing}"


def test_holdout_cutoff(ds):
    cutoff = ds.holdout_cutoff()
    assert cutoff is not None


def test_options_chain(ds):
    df = ds.options_chain("AAPL", on="2024-06-03", limit=100)
    assert len(df) > 0


def test_trading_days(ds):
    days = ds.trading_days(start="2024-01-01", end="2024-01-31")
    assert len(days) >= 18  # Jan 2024 had 21 trading days


def test_bad_token_rejected():
    from statevector import Dataset
    with pytest.raises(Exception):
        Dataset(BASE, token="definitely-wrong-token").universe()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
