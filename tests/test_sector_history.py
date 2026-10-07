#!/usr/bin/env python3
"""Unit tests for sector_history() + sector_map(asof=) — the PIT
sector surface over SEC FSDS history."""

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset  # noqa: E402

HIST_SCHEMA = {"ticker": pl.Utf8, "cik": pl.Utf8, "filed": pl.Date,
               "sic_code": pl.Utf8, "sector": pl.Utf8, "form": pl.Utf8}


def _mkroot(tmp_path: Path) -> Path:
    """Isolated minimal root (never the shared session fixture)."""
    mdir = tmp_path / "data" / "raw" / "massive"
    mdir.mkdir(parents=True)
    pl.DataFrame([
        {"ticker": "OLD", "name": "Old Co", "primary_exchange": "XNYS",
         "type": "CS"},
        {"ticker": "NEWCO", "name": "New Co", "primary_exchange": "XNYS",
         "type": "CS"},
        {"ticker": "ETF1", "name": "Fund", "primary_exchange": "XNYS",
         "type": "ETF"},
    ]).write_parquet(mdir / "reference_tickers.parquet")
    return tmp_path


def _write_history(root: Path):
    sdir = root / "data" / "structural"
    sdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([
        {"ticker": "OLD", "cik": "0000000001",
         "filed": date(2010, 3, 1), "sic_code": "6021",
         "sector": "financials", "form": "10-K"},
        {"ticker": "OLD", "cik": "0000000001",
         "filed": date(2020, 3, 1), "sic_code": "3571",
         "sector": "industrials_manufacturing", "form": "10-K"},
        {"ticker": "NEWCO", "cik": "0000000002",
         "filed": date(2024, 3, 1), "sic_code": "7372",
         "sector": "services_tech_health", "form": "10-K"},
    ], schema=HIST_SCHEMA).write_parquet(
        sdir / "sector_history.parquet")


def test_sector_history_accessor(tmp_path):
    root = _mkroot(tmp_path)
    _write_history(root)
    ds = Dataset(str(root))
    df = ds.sector_history()
    assert df.height == 3
    assert ds.sector_history(tickers="OLD").height == 2
    assert ds.sector_history(end="2015-01-01").height == 1
    assert ds.sector_history(start="2015-01-01").height == 2


def test_sector_history_absent_panel(tmp_path):
    root = _mkroot(tmp_path)
    df = Dataset(str(root)).sector_history()
    assert df.height == 0
    assert set(df.columns) == set(HIST_SCHEMA)


def test_sector_map_asof_picks_era(tmp_path):
    root = _mkroot(tmp_path)
    _write_history(root)
    ds = Dataset(str(root))
    assert ds.sector_map(asof="2015-06-01")["OLD"] == "financials"
    assert ds.sector_map(asof="2021-01-01")["OLD"] == \
        "industrials_manufacturing"
    assert ds.sector_map(asof="2025-01-01")["NEWCO"] == \
        "services_tech_health"


def test_sector_map_asof_honest_nulls(tmp_path):
    root = _mkroot(tmp_path)
    _write_history(root)
    ds = Dataset(str(root))
    # pre-first-filing: never leak the future sector
    assert ds.sector_map(asof="2015-06-01")["NEWCO"] is None
    assert ds.sector_map(asof="2005-01-01")["OLD"] is None


def test_sector_map_asof_immutable_class(tmp_path):
    root = _mkroot(tmp_path)
    _write_history(root)
    sdir = root / "data" / "structural"
    pl.DataFrame({"ticker": ["ETF1"], "sector": ["fund"]}
                 ).write_parquet(sdir / "ticker_details.parquet")
    ds = Dataset(str(root))
    assert ds.sector_map(asof="2015-06-01")["ETF1"] == "fund"
    assert ds.sector_map(asof="2005-01-01")["ETF1"] == "fund"


def test_sector_map_asof_absent_history(tmp_path):
    """No sector_history panel: asof mode still yields instrument
    classes, SIC sectors become None."""
    root = _mkroot(tmp_path)
    sdir = root / "data" / "structural"
    sdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ticker": ["ETF1", "OLD"],
                  "sector": ["fund", "financials"]}
                 ).write_parquet(sdir / "ticker_details.parquet")
    ds = Dataset(str(root))
    m = ds.sector_map(asof="2020-01-01")
    assert m["ETF1"] == "fund"
    assert m["OLD"] is None
