#!/usr/bin/env python3
"""Unit tests — pure logic, no data, no network. Covers the helpers
and edge paths the data tests don't reach.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector.dataset import _file_in_window, _as_date, parse_occ  # noqa: E402
from statevector.pit import holdout_cutoff, holdout_mask  # noqa: E402


# -- _file_in_window: the remote partition-pruning guard ---------------------

@pytest.mark.parametrize("rel,start,end,want", [
    # day-partitioned files
    ("raw/massive/stocks_daily/2020-06-15.parquet", "2020-01-01", "2020-12-31", True),
    ("raw/massive/stocks_daily/2019-06-15.parquet", "2020-01-01", "2020-12-31", False),
    ("raw/massive/stocks_daily/2021-06-15.parquet", "2020-01-01", "2020-12-31", False),
    ("raw/massive/stocks_daily/2020-06-15.parquet", "2020-06-15", "2020-06-15", True),
    ("raw/massive/stocks_daily/2020-06-15.parquet", None, "2020-06-14", False),
    ("raw/massive/stocks_daily/2020-06-15.parquet", "2020-06-16", None, False),
    # year-partitioned files (overlap semantics, not exact containment)
    ("canonical/state_vector/2020.parquet", "2020-06-01", "2021-03-01", True),
    ("canonical/state_vector/2019.parquet", "2020-06-01", "2021-03-01", False),
    ("canonical/state_vector/2022.parquet", "2020-06-01", "2021-03-01", False),
    ("canonical/state_vector/2020.parquet", None, "2020-01-15", True),   # year overlaps
    # non-date names always kept
    ("canonical/fundamentals_actuals.parquet", "2020-01-01", "2020-12-31", True),
    ("structural/ticker_map.parquet", "2020-01-01", None, True),
    # no window — everything kept
    ("raw/massive/stocks_daily/1999-01-01.parquet", None, None, True),
])
def test_file_in_window(rel, start, end, want):
    assert _file_in_window(rel, start, end) is want


# -- _as_date ----------------------------------------------------------------

def test_as_date_accepts_str_and_date():
    d = _as_date("2024-03-31")
    assert d == date(2024, 3, 31)
    assert _as_date(date(2024, 3, 31)) == date(2024, 3, 31)
    assert _as_date(None) is None


# -- OCC parsing (dict form, dataset.parse_occ) --------------------------------

def test_parse_occ_golden():
    c = parse_occ("O:AAPL260923C00245000")
    assert c == {"underlying": "AAPL", "expiry": "2026-09-23",
                 "side": "call", "strike": 245.0}


def test_parse_occ_put_strike_scaling():
    c = parse_occ("O:SPY250117P00560000")
    assert c["side"] == "put" and c["strike"] == 560.0


def test_parse_occ_rejects_non_occ():
    with pytest.raises(ValueError):
        parse_occ("AAPL")


# -- holdout arithmetic --------------------------------------------------------

def test_holdout_cutoff_is_30_calendar_days():
    end = date(2024, 6, 30)
    assert holdout_cutoff(30, end=end) == date(2024, 5, 31)


def test_holdout_mask_marks_trailing_rows():
    import polars as pl
    end = date(2024, 6, 30)
    df = pl.DataFrame({"date": [date(2024, 5, 30), date(2024, 6, 1)]})
    mask = df.select(holdout_mask("date", 30, end=end))["date"].to_list()
    assert mask == [False, True]


# -- template helpers (imported, not re-implemented) ----------------------------

def _load_template(sv_root):
    """Import the template app module with env pointed at the fixture."""
    import importlib.util
    import os
    os.environ["SV_DATA_ROOT"] = str(sv_root)
    spec = importlib.util.spec_from_file_location(
        "hackathon_app",
        Path(__file__).parent.parent / "launchpad/template/app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hackathon_app"] = mod  # pydantic needs the module registered
    spec.loader.exec_module(mod)
    return mod


def test_is_option(sv_fixture_root):
    app = _load_template(sv_fixture_root)
    assert app.is_option("O:AAPL260923C00245000")
    assert not app.is_option("AAPL")


def test_clean_converts_nonfinite(sv_fixture_root):
    app = _load_template(sv_fixture_root)
    out = app.clean([{"x": float("nan"), "y": float("inf"), "z": 1.5}])
    assert out == [{"x": None, "y": None, "z": 1.5}]


def test_backtest_validation_rejects_shorting(sv_fixture_root):
    app = _load_template(sv_fixture_root)
    from fastapi import HTTPException
    req = app.BacktestRequest(
        tickers=["AAPL", "MSFT"], weights=[1.2, -0.2],
        start=date(2020, 1, 2), end=date(2020, 6, 30))
    with pytest.raises(HTTPException):
        app.backtest(req)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
