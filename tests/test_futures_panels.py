#!/usr/bin/env python3
"""Unit tests for the futures Dataset accessors — Apex universe
panels under data/structural/futures_*.parquet."""

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset  # noqa: E402


def _mkroot(tmp_path: Path) -> Path:
    (tmp_path / "data" / "structural").mkdir(parents=True)
    return tmp_path


def _write_sessions(root: Path):
    pl.DataFrame([
        {"ticker": "ESH5", "product_code": "ES",
         "session_end_date": "2025-03-10",
         "window_start": 1_741_596_000_000_000_000,
         "open": 5800.0, "high": 5820.0, "low": 5790.0,
         "close": 5810.0, "volume": 1_500_000.0,
         "transactions": 900_000.0, "dollar_volume": 8.7e12,
         "settlement_price": 5811.0},
        {"ticker": "ESH5", "product_code": "ES",
         "session_end_date": "2025-03-11",
         "window_start": 1_741_680_000_000_000_000,
         "open": 5812.0, "high": 5830.0, "low": 5805.0,
         "close": 5825.0, "volume": 1_400_000.0,
         "transactions": 880_000.0, "dollar_volume": 8.1e12,
         "settlement_price": 5824.5},
        {"ticker": "ESH5-ESM5", "product_code": "ES",
         "session_end_date": "2025-03-10",
         "window_start": 1_741_596_000_000_000_000,
         "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
         "volume": 10.0, "transactions": 5.0,
         "dollar_volume": 1.0, "settlement_price": None},
        {"ticker": "MESH5", "product_code": "MES",
         "session_end_date": "2025-03-10",
         "window_start": 1_741_596_000_000_000_000,
         "open": 5801.0, "high": 5821.0, "low": 5791.0,
         "close": 5811.0, "volume": 900_000.0,
         "transactions": 400_000.0, "dollar_volume": 5.2e11,
         "settlement_price": 5811.0},
    ]).write_parquet(
        root / "data" / "structural" / "futures_sessions.parquet")


def _write_contracts(root: Path):
    pl.DataFrame([
        {"ticker": "ESH5", "product_code": "ES",
         "date": "2025-03-10", "active": True,
         "first_trade_date": "2024-06-01",
         "last_trade_date": "2025-03-21",
         "settlement_date": "2025-03-21", "days_to_maturity": 11,
         "trade_tick_size": 0.25, "settlement_tick_size": 0.25,
         "spread_tick_size": 0.05, "trading_venue": "XCME",
         "group_code": "ES", "type": "single", "name": "E-mini",
         "min_order_quantity": 1, "max_order_quantity": None},
        {"ticker": "ESH5-ESM5", "product_code": "ES",
         "date": "2025-03-10", "active": True,
         "first_trade_date": "2024-06-01",
         "last_trade_date": "2025-03-21",
         "settlement_date": "2025-03-21", "days_to_maturity": None,
         "trade_tick_size": 0.05, "settlement_tick_size": 0.05,
         "spread_tick_size": 0.05, "trading_venue": "XCME",
         "group_code": "ES", "type": "single", "name": "spread",
         "min_order_quantity": 1, "max_order_quantity": None},
        {"ticker": "6EH5", "product_code": "6E",
         "date": "2025-03-10", "active": True,
         "first_trade_date": "2024-06-01",
         "last_trade_date": "2025-03-17",
         "settlement_date": "2025-03-17", "days_to_maturity": 7,
         "trade_tick_size": 0.00005, "settlement_tick_size": 0.00005,
         "spread_tick_size": None, "trading_venue": "XCME",
         "group_code": "6E", "type": "single", "name": "Euro FX",
         "min_order_quantity": 1, "max_order_quantity": None},
    ]).write_parquet(
        root / "data" / "structural" / "futures_contracts.parquet")


def _write_labels(root: Path):
    pl.DataFrame([
        {"ticker": "ESH5", "product_code": "ES",
         "session_end_date": "2025-03-10", "y_cc": 0.00258,
         "y_oc": 0.00224, "y_settle": 0.00232,
         "horizon_hours": 23.5, "is_last_session": False,
         "expires_next": False, "is_roll_boundary": False,
         "next_ticker": "ESH5"},
        {"ticker": "ESH5", "product_code": "ES",
         "session_end_date": "2025-03-11", "y_cc": None,
         "y_oc": None, "y_settle": None, "horizon_hours": None,
         "is_last_session": True, "expires_next": False,
         "is_roll_boundary": True, "next_ticker": None},
    ]).write_parquet(
        root / "data" / "structural" / "futures_labels.parquet")


def _write_minute(root: Path):
    pl.DataFrame([
        {"ticker": "ESH5", "product_code": "ES",
         "window_start": 1_741_596_000_000_000_000,
         "session_end_date": "2025-03-10", "open": 5800.0,
         "high": 5801.0, "low": 5799.0, "close": 5800.5,
         "volume": 3000.0, "transactions": 800.0,
         "is_front_day": True},
        {"ticker": "ESH5", "product_code": "ES",
         "window_start": 1_741_596_060_000_000_000,
         "session_end_date": "2025-03-10", "open": 5800.5,
         "high": 5802.0, "low": 5800.0, "close": 5801.0,
         "volume": 2500.0, "transactions": 700.0,
         "is_front_day": False},
    ]).write_parquet(
        root / "data" / "structural" / "futures_minute.parquet")


def _write_roll(root: Path):
    pl.DataFrame([
        {"product_code": "ES", "date": "2025-03-10",
         "front_ticker": "ESH5", "front_dtm": 11,
         "next_ticker": "ESM5", "is_roll_day": False,
         "front_near_expiry": False},
        {"product_code": "ES", "date": "2025-03-11",
         "front_ticker": "ESM5", "front_dtm": 101,
         "next_ticker": "ESU5", "is_roll_day": True,
         "front_near_expiry": False},
    ]).write_parquet(
        root / "data" / "structural" / "futures_roll.parquet")


def _write_products(root: Path):
    pl.DataFrame([
        {"product_code": "ES", "date": "2025-01-01",
         "asset_sub_class": "Equity Index", "trading_venue": "XCME",
         "trade_currency_code": "USD", "type": "financial",
         "unit_of_measure": "index", "unit_of_measure_qty": 50.0,
         "name": "E-mini S&P"},
        {"product_code": "ES", "date": "2025-06-01",
         "asset_sub_class": "Equity Index", "trading_venue": "XCME",
         "trade_currency_code": "USD", "type": "financial",
         "unit_of_measure": "index", "unit_of_measure_qty": 50.0,
         "name": "E-mini S&P v2"},
    ]).write_parquet(
        root / "data" / "structural" / "futures_products.parquet")


def test_empty_frames_when_panels_absent(tmp_path):
    ds = Dataset(str(_mkroot(tmp_path)))
    for f in (ds.futures_products, ds.futures_contracts,
              ds.futures_schedules, ds.futures_sessions,
              ds.futures_minute, ds.futures_meta,
              ds.futures_roll, ds.futures_labels):
        assert f().height == 0


def test_sessions_filters_and_combos_excluded_from_fixture(tmp_path):
    root = _mkroot(tmp_path)
    _write_sessions(root)
    ds = Dataset(str(root))
    es = ds.futures_sessions(roots="ES")
    assert set(es["product_code"]) == {"ES"}
    one = ds.futures_sessions(tickers="ESH5",
                              start="2025-03-11",
                              end="2025-03-11")
    assert one.height == 1
    assert one["close"][0] == 5825.0
    # settlement never conflated with close
    assert one["settlement_price"][0] == 5824.5


def test_contracts_singles_only_drops_dash_tickers(tmp_path):
    root = _mkroot(tmp_path)
    _write_contracts(root)
    ds = Dataset(str(root))
    singles = ds.futures_contracts(roots="ES")
    assert set(singles["ticker"]) == {"ESH5"}
    allr = ds.futures_contracts(roots="ES", singles_only=False)
    assert "ESH5-ESM5" in set(allr["ticker"])


def test_labels_labeled_only_and_flags(tmp_path):
    root = _mkroot(tmp_path)
    _write_labels(root)
    ds = Dataset(str(root))
    lab = ds.futures_labels(roots="ES")
    assert lab.height == 2
    got = ds.futures_labels(roots="ES", labeled_only=True)
    assert got.height == 1
    assert got["y_cc"][0] == 0.00258
    last = lab.filter(pl.col("session_end_date") == "2025-03-11")
    assert last["is_last_session"][0] is True
    assert last["is_roll_boundary"][0] is True


def test_minute_front_only_and_window(tmp_path):
    root = _mkroot(tmp_path)
    _write_minute(root)
    ds = Dataset(str(root))
    front = ds.futures_minute(tickers="ESH5", front_only=True)
    assert front.height == 1
    assert front["is_front_day"][0] is True
    after = ds.futures_minute(tickers="ESH5",
                              start="2025-03-11")
    assert after.height == 0
    day = ds.futures_minute(tickers="ESH5",
                            start="2025-03-09", end="2025-03-10")
    assert day.height == 2


def test_roll_and_meta(tmp_path):
    root = _mkroot(tmp_path)
    _write_roll(root)
    pl.DataFrame([
        {"product_code": "ES", "parent_code": "ES",
         "is_micro": False},
        {"product_code": "MES", "parent_code": "ES",
         "is_micro": True},
    ]).write_parquet(
        root / "data" / "structural" / "futures_meta.parquet")
    ds = Dataset(str(root))
    roll = ds.futures_roll(roots="ES")
    assert roll.height == 2
    assert roll.filter(pl.col("is_roll_day"))["front_ticker"][0] \
        == "ESM5"
    meta = ds.futures_meta()
    mes = meta.filter(pl.col("product_code") == "MES")
    assert mes["parent_code"][0] == "ES"
    assert mes["is_micro"][0] is True


def test_products_asof_picks_latest_prior_row(tmp_path):
    root = _mkroot(tmp_path)
    _write_products(root)
    ds = Dataset(str(root))
    latest = ds.futures_products(roots="ES", asof="2025-07-01")
    assert latest.height == 1
    assert latest["name"][0] == "E-mini S&P v2"
    early = ds.futures_products(roots="ES", asof="2025-02-01")
    assert early["name"][0] == "E-mini S&P"
