#!/usr/bin/env python3
"""Integration tests — the SDK against a real (synthetic) dataset root.
No network, no mocks: exercises the actual scan/filter/PIT code paths
that contestants hit, on data we control.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset  # noqa: E402


@pytest.fixture(scope="module")
def ds(sv_fixture_root):
    return Dataset(sv_fixture_root)


def test_universe_from_fixture(ds):
    assert sorted(ds.universe()) == ["AAPL", "MSFT", "NVDA"]


def test_prices_respects_window(ds):
    df = ds.prices("AAPL", start="2020-03-01", end="2020-03-31")
    assert len(df) == 22  # March 2020 had 22 business days
    assert str(df["date"].min())[:10] >= "2020-03-01"
    assert str(df["date"].max())[:10] <= "2020-03-31"


def test_get_panel_ticker_filter(ds):
    df = ds.get("stocks_daily", ticker="MSFT",
                start="2020-01-02", end="2020-01-31")
    assert set(df["ticker"].unique()) == {"MSFT"}
    assert len(df) == 22


def test_fundamentals_pit_visible_after_filing(ds):
    df = ds.fundamentals("AAPL", asof="2024-03-31")
    assert len(df) == 1
    assert df["date"].iloc[0] == date(2024, 2, 1) or \
        str(df["date"].iloc[0])[:10] == "2024-02-01"


def test_fundamentals_before_any_filing_falls_back(ds):
    # documented PIT caveat: no filings before 2024-02-10, so the
    # period-end fallback applies — row is visible by date alone
    df = ds.fundamentals("AAPL", asof="2024-02-20")
    assert len(df) == 1


def test_panels_lists_fixture(ds):
    panels = ds.panels()
    assert "stocks_daily" in panels and panels["stocks_daily"] > 0


def test_holdout_cutoff_type(ds):
    c = ds.holdout_cutoff()
    assert isinstance(c, date)


def test_unknown_panel_raises(ds):
    with pytest.raises(KeyError):
        ds.get("not_a_panel")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
