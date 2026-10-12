#!/usr/bin/env python3
"""Unit tests for anchor panels — anchor_daily, futures_anchor_map,
futures_anchors accessors + builder helpers."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset  # noqa: E402


def _mkroot(tmp_path: Path) -> Path:
    (tmp_path / "data" / "structural").mkdir(parents=True)
    return tmp_path


def test_anchor_panels_empty_when_absent(tmp_path):
    ds = Dataset(str(_mkroot(tmp_path)))
    assert ds.anchor_daily().height == 0
    assert ds.futures_anchor_map().height == 0
    assert ds.futures_anchors().height == 0


def test_anchor_daily_filters(tmp_path):
    root = _mkroot(tmp_path)
    pl.DataFrame([
        {"ticker": "I:SPX", "date": "2025-03-10", "open": 5700.0,
         "high": 5750.0, "low": 5690.0, "close": 5740.0,
         "volume": None, "vwap": None, "transactions": None},
        {"ticker": "C:EURUSD", "date": "2025-03-10",
         "open": 1.09, "high": 1.10, "low": 1.08, "close": 1.095,
         "volume": 200000.0, "vwap": 1.094, "transactions": 200000.0},
        {"ticker": "I:SPX", "date": "2025-03-11", "open": 5741.0,
         "high": 5760.0, "low": 5730.0, "close": 5755.0,
         "volume": None, "vwap": None, "transactions": None},
    ]).write_parquet(
        root / "data" / "structural" / "anchor_daily.parquet")
    ds = Dataset(str(root))
    spx = ds.anchor_daily("I:SPX")
    assert spx.height == 2
    assert ds.anchor_daily("I:SPX", end="2025-03-10").height == 1
    eur = ds.anchor_daily("C:EURUSD")
    assert eur["close"][0] == 1.095


def test_futures_anchors_filters(tmp_path):
    root = _mkroot(tmp_path)
    pl.DataFrame([
        {"product_code": "ES", "session_end_date": "2025-03-10",
         "front_ticker": "ESH5", "front_rule": "min_dtm",
         "is_roll_day": False, "fut_close": 5810.0,
         "fut_settle": 5811.0, "volume": 1e6,
         "anchor_ticker": "I:SPX", "anchor_kind": "index",
         "direction": "direct", "is_proxy": False,
         "anchor_close": 5800.0, "anchor_open": 5700.0,
         "basis_bps": 17.24},
        {"product_code": "NKD", "session_end_date": "2025-03-10",
         "front_ticker": "NKDH5", "front_rule": "volume_argmax",
         "is_roll_day": False, "fut_close": 39000.0,
         "fut_settle": 39010.0, "volume": 500.0,
         "anchor_ticker": "EWJ", "anchor_kind": "etf",
         "direction": "direct", "is_proxy": True,
         "anchor_close": 75.0, "anchor_open": 74.5,
         "basis_bps": None},
    ]).write_parquet(
        root / "data" / "structural" / "futures_anchors.parquet")
    ds = Dataset(str(root))
    es = ds.futures_anchors(roots="ES")
    assert es.height == 1
    assert es["front_ticker"][0] == "ESH5"
    assert es["basis_bps"][0] == 17.24
    nkd = ds.futures_anchors(roots="NKD")
    assert nkd["is_proxy"][0] is True
    assert ds.futures_anchors(roots="ES",
                              start="2025-03-11").height == 0


def test_anchor_map_shape(tmp_path):
    root = _mkroot(tmp_path)
    pl.DataFrame([
        {"product_code": "6J", "anchor_ticker": "C:USDJPY",
         "anchor_kind": "forex", "direction": "invert",
         "is_proxy": False, "note": ""},
        {"product_code": "NKD", "anchor_ticker": "EWJ",
         "anchor_kind": "etf", "direction": "direct",
         "is_proxy": True, "note": "proxy"},
    ]).write_parquet(
        root / "data" / "structural" / "futures_anchor_map.parquet")
    ds = Dataset(str(root))
    m = ds.futures_anchor_map()
    j = m.filter(pl.col("product_code") == "6J")
    assert j["direction"][0] == "invert"
