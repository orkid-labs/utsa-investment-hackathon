"""Phase-5 panel accessor tests (synthetic root — public repo)."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset  # noqa: E402


def _mkroot(tmp_path: Path) -> Path:
    (tmp_path / "data" / "structural").mkdir(parents=True)
    return tmp_path


def test_etf_flows(tmp_path):
    root = _mkroot(tmp_path)
    pl.DataFrame([
        {"ticker": "SPY", "processed_date": "2026-10-08",
         "effective_date": "2026-10-08", "nav": 773.7,
         "fund_flow": -4.66e8, "staleness_days": 0},
    ]).write_parquet(root / "data/structural/etf_flows.parquet")
    ds = Dataset(str(root))
    f = ds.etf_flows(ticker="SPY")
    assert f.height == 1
    assert f["staleness_days"][0] == 0
    assert ds.etf_flows(start="2026-10-09").height == 0


def test_etf_constituents(tmp_path):
    root = _mkroot(tmp_path)
    pl.DataFrame([
        {"ticker": "SPY", "processed_date": "2026-10-09",
         "n_holdings": 350, "top1_weight": 0.083,
         "top5_weight": 0.278, "top10_weight": 0.362,
         "hhi": 0.0206},
    ]).write_parquet(
        root / "data/structural/etf_constituents.parquet")
    c = Dataset(str(root)).etf_constituents(ticker="SPY")
    assert c.height == 1
    assert c["n_holdings"][0] == 350


def test_apex_constraints(tmp_path):
    root = _mkroot(tmp_path)
    pl.DataFrame([
        {"scope": "account", "product_code": None,
         "rule": "daily_loss_limit", "value": 1000.0,
         "unit": "usd"},
        {"scope": "product", "product_code": "ES",
         "rule": "max_contracts", "value": 6.0,
         "unit": "contracts"},
    ]).write_parquet(
        root / "data/structural/apex_constraints.parquet")
    ds = Dataset(str(root))
    a = ds.apex_constraints(scope="account")
    assert a.height == 1
    p = ds.apex_constraints(product_code="ES")
    assert p["value"][0] == 6.0


def test_options_snapshots(tmp_path):
    root = _mkroot(tmp_path)
    pl.DataFrame([
        {"captured_date": "2026-10-12", "underlying": "SPY",
         "contract": "O:SPY261012C00550000",
         "strike": 550.0, "open_interest": 4},
    ]).write_parquet(
        root / "data/structural/options_snapshots.parquet")
    ds = Dataset(str(root))
    o = ds.options_snapshots(underlying="SPY")
    assert o.height == 1
    assert o["open_interest"][0] == 4


def test_micro_and_ticks_empty_absent(tmp_path):
    ds = Dataset(str(_mkroot(tmp_path)))
    assert ds.futures_ticks().height == 0
    assert ds.futures_micro().height == 0
