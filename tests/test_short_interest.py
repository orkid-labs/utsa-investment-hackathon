#!/usr/bin/env python3
"""Unit tests for Dataset.short_interest() — the FINRA biweekly
short-interest structural panel."""

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset  # noqa: E402


def _mkroot(tmp_path: Path) -> Path:
    (tmp_path / "data" / "structural").mkdir(parents=True)
    return tmp_path


def _write_si(root: Path):
    pl.DataFrame([
        {"ticker": "GME", "settlement_date": date(2021, 1, 15),
         "short_interest": 61782730, "avg_daily_volume": 29385884,
         "days_to_cover": 2.1},
        {"ticker": "GME", "settlement_date": date(2021, 1, 29),
         "short_interest": 21409004, "avg_daily_volume": 96789540,
         "days_to_cover": 1.0},
        {"ticker": "XYZ", "settlement_date": date(2021, 1, 15),
         "short_interest": 100, "avg_daily_volume": 50,
         "days_to_cover": 0.5},
    ]).write_parquet(
        root / "data" / "structural" / "short_interest.parquet")


def test_short_interest_empty_when_panel_absent(tmp_path):
    ds = Dataset(str(_mkroot(tmp_path)))
    out = ds.short_interest()
    assert out.height == 0
    assert "settlement_date" in out.columns


def test_short_interest_filters_ticker_and_dates(tmp_path):
    root = _mkroot(tmp_path)
    _write_si(root)
    ds = Dataset(str(root))
    gme = ds.short_interest("GME")
    assert gme.height == 2
    assert set(gme["ticker"]) == {"GME"}
    first = ds.short_interest("GME", end="2021-01-20")
    assert first.height == 1
    assert first["short_interest"][0] == 61782730
    assert ds.short_interest(start="2022-01-01").height == 0


def test_short_interest_scalar_ticker_not_char_exploded(tmp_path):
    """is_in("GME") must not match ticker 'G' — the splits() scalar
    footgun pattern."""
    root = _mkroot(tmp_path)
    _write_si(root)
    ds = Dataset(str(root))
    assert ds.short_interest("GME")["ticker"].unique().to_list() == \
        ["GME"]
