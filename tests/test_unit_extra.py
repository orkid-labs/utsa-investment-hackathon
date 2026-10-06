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


# -- CASHHOLDING sleeve --------------------------------------------------------

def test_cashholding_zero_return_free_leg():
    import polars as pl
    from statevector.backtest import run_backtest, CASH_TICKER
    rets = pl.DataFrame({"AAPL": [0.10, -0.05, 0.02, 0.03]})
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
             date(2024, 1, 5)]
    all_in = run_backtest(rets, dates, ["AAPL"], [1.0])
    half = run_backtest(rets, dates, ["AAPL", CASH_TICKER], [0.5, 0.5])
    assert all_in["n_days"] == half["n_days"] == 4
    # cash dilutes the path but never trades: roughly half the cost drag
    assert half["cost_total"] < all_in["cost_total"]
    assert half["cost_total"] == pytest.approx(
        all_in["cost_total"] / 2, abs=1e-4)
    assert half["total_return"] != all_in["total_return"]
    # turnover counts the cash sleeve (weight moved) even though it pays 0 bps
    assert half["turnover"] == pytest.approx(all_in["turnover"])


def test_cashholding_template_roundtrip(sv_fixture_root):
    app = _load_template(sv_fixture_root)
    req = app.BacktestRequest(
        tickers=["AAPL", "CASHHOLDING"], weights=[0.5, 0.5],
        start=date(2020, 1, 2), end=date(2020, 6, 30))
    out = app.backtest(req)
    assert out["n_days"] > 60
    assert "cost_total" in out and "turnover" in out
    # 50% cash book should cost ~half of the all-in round trip
    req2 = app.BacktestRequest(
        tickers=["AAPL"], weights=[1.0],
        start=date(2020, 1, 2), end=date(2020, 6, 30))
    out2 = app.backtest(req2)
    assert out["cost_total"] < out2["cost_total"]


# -- split adjustment -----------------------------------------------------------

def test_apply_split_factors_corrects_ex_date():
    """A 4-for-1 split prints a raw -75% return; the adjusted return is
    the true residual move, not the phantom crash."""
    import polars as pl
    from statevector.backtest import apply_split_factors
    dates = [date(2020, 8, 27), date(2020, 8, 28), date(2020, 8, 31),
             date(2020, 9, 1)]
    # raw closes 500 -> 125 on split day: raw ret -0.75; true move +3%
    rets = pl.DataFrame({"AAPL": [0.01, 0.02, -0.742, 0.005],
                         "MSFT": [0.01, -0.01, 0.005, 0.0]})
    splits = pl.DataFrame({"ticker": ["AAPL"],
                           "date": [date(2020, 8, 31)],
                           "factor": [4.0]})
    out = apply_split_factors(rets, dates, splits)
    assert out["AAPL"][2] == pytest.approx((1 - 0.742) * 4.0 - 1.0)
    assert out["AAPL"][2] == pytest.approx(0.032, abs=1e-9)
    # untouched cells and other legs unchanged
    assert out["AAPL"][1] == 0.02
    assert out["MSFT"][2] == 0.005
    # no splits -> identity
    assert apply_split_factors(rets, dates, None) is rets


def test_split_day_backtest_does_not_crash():
    """A book holding a split name through ex-date earns the residual,
    not -37.5% on a 50% weight."""
    import polars as pl
    from statevector.backtest import apply_split_factors, run_backtest
    dates = [date(2020, 8, 28), date(2020, 8, 31), date(2020, 9, 1)]
    rets = pl.DataFrame({"AAPL": [0.02, -0.742, 0.005],
                         "MSFT": [-0.01, 0.005, 0.01]})
    splits = pl.DataFrame({"ticker": ["AAPL"],
                           "date": [date(2020, 8, 31)],
                           "factor": [4.0]})
    raw = run_backtest(rets, dates, ["AAPL", "MSFT"], [0.5, 0.5])
    adj = run_backtest(apply_split_factors(rets, dates, splits),
                       dates, ["AAPL", "MSFT"], [0.5, 0.5])
    # raw path loses ~19% to the phantom; adjusted earns ~+2%
    assert raw["total_return"] < -0.15
    assert adj["total_return"] > 0.0



# -- decision-series format: validation + event-driven engine ----------------

def _series_frame():
    import polars as pl
    """5 trading days: A +10%/day, B flat."""
    days = [date(2026, 9, d) for d in (1, 2, 3, 4, 7)]
    return pl.DataFrame({"A": [0.10] * 5, "B": [0.0] * 5}), days


def _rec(i, cutoff, decide, execute, holds, action="rebalance"):
    return {
        "schema_version": "1.0",
        "team_id": "t",
        "model_id": "m",
        "decision_id": i,
        "information_cutoff": cutoff,
        "decision_time": decide,
        "execution_time": execute,
        "action": action,
        "target_holdings": holds,
    }


def test_decision_series_validates_good_series():
    from statevector.backtest import validate_decision_series
    recs = [
        _rec(1, "2026-08-31T16:00:00-04:00", "2026-08-31T16:15:00-04:00",
             "2026-09-01T09:30:00-04:00", [{"ticker": "A", "weight": 1.0}]
             if False else [{"ticker": "A", "weight": 0.2}] * 1),
    ]
    # 1.0 weight exceeds the 20% cap — use a 5-name book instead
    recs[0]["target_holdings"] = [{"ticker": t, "weight": 0.2}
                                  for t in "ABCDE"]
    decisions, errors = validate_decision_series(recs)
    assert errors == [] and len(decisions) == 1
    assert decisions[0]["date"] == date(2026, 9, 1)
    assert decisions[0]["weights"]["A"] == 0.2


def test_decision_series_rejects_bad_records():
    from statevector.backtest import validate_decision_series
    base = dict(cutoff="2026-08-31T16:00:00-04:00",
                decide="2026-08-31T16:15:00-04:00",
                execute="2026-09-01T09:30:00-04:00")
    legs = [{"ticker": t, "weight": 0.2} for t in "ABCDE"]

    # decision_id not increasing
    _, e = validate_decision_series(
        [_rec(2, **base, holds=legs), _rec(1, **base, holds=legs)])
    assert any("decision_id" in x for x in e)

    # lookahead: decision before information_cutoff
    _, e = validate_decision_series([_rec(
        1, "2026-08-31T16:00:00-04:00", "2026-08-31T15:00:00-04:00",
        "2026-09-01T09:30:00-04:00", legs)])
    assert any("information_cutoff" in x for x in e)

    # instant fill: execution <= decision
    _, e = validate_decision_series([_rec(
        1, "2026-08-31T16:00:00-04:00", "2026-09-01T09:30:00-04:00",
        "2026-09-01T09:30:00-04:00", legs)])
    assert any("execution_time" in x for x in e)

    # sum != 1
    _, e = validate_decision_series([_rec(
        1, **base, holds=[{"ticker": "A", "weight": 0.2}])])
    assert any("sum" in x for x in e)

    # >20% single name
    _, e = validate_decision_series([_rec(
        1, **base, holds=[{"ticker": t, "weight": w}
                          for t, w in (("A", 0.4), ("B", 0.2),
                                       ("C", 0.2), ("D", 0.1), ("E", 0.1))])])
    assert any("cap" in x for x in e)

    # duplicate execution dates
    _, e = validate_decision_series(
        [_rec(1, **base, holds=legs), _rec(2, **base, holds=legs)])
    assert any("duplicate" in x for x in e)


def test_decision_series_matches_static_backtest():
    """Golden: a 1-decision series executed at window start is exactly a
    buy-and-hold run_backtest on the same book."""
    from statevector.backtest import run_backtest, run_backtest_series
    rets, days = _series_frame()
    w = {"A": 0.6, "B": 0.4} if False else {"A": 1.0}
    static = run_backtest(rets, days, ["A"], [1.0])
    series = run_backtest_series(
        rets, days, [{"date": days[0], "weights": {"A": 1.0}}])
    for k in ("total_return", "sharpe", "max_drawdown", "turnover",
              "cost_total"):
        assert abs(static[k] - series[k]) < 1e-12, (k, static[k], series[k])


def test_decision_series_transition_math():
    """All-A -> all-B switch mid-window: turnover 1+2+1, cost 10bp per unit."""
    from statevector.backtest import run_backtest_series
    rets, days = _series_frame()
    m = run_backtest_series(rets, days, [
        {"date": days[0], "weights": {"A": 1.0}},
        {"date": days[2], "weights": {"B": 1.0}},
    ])
    # entry 1.0 |A|, A->B switch 2.0, exit 1.0 |B|
    assert abs(m["turnover"] - 4.0) < 1e-9
    # net days: +0.1-0.001, +0.1, +0.1-0.002, 0, 0-0.001
    want = ((1.099) * (1.10) * (1.098) * 1.0 * (0.999)) - 1.0
    assert abs(m["total_return"] - want) < 1e-9
    assert m["n_decisions"] == 2
    # gross (cost-free) curve earns three +10% days on A then flat B
    assert abs(m["cost_total"] - ((1.10 ** 3 - 1.0) - want)) < 1e-9


def test_decision_series_weekend_exec_clamps_forward():
    """An execution_time landing off-calendar executes the next trading day."""
    from statevector.backtest import run_backtest_series
    rets, days = _series_frame()   # 9/4 is Fri, 9/5-6 weekend, 9/7 Mon
    m = run_backtest_series(rets, days, [
        {"date": days[0], "weights": {"A": 1.0}},
        {"date": date(2026, 9, 5), "weights": {"B": 1.0}},  # Saturday
    ])
    # switch lands on Monday d4: A earns day-4's +10% on drifted weights,
    # then pays transition (0.002) + liquidation (0.001) -> port4 = 0.097
    want = (1.099) * (1.10) ** 3 * (1.097) - 1.0
    assert abs(m["total_return"] - want) < 1e-9

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))


# -- sectors(): reduced reference_tickers schema ----------------------------

def test_sectors_tolerates_reduced_schema(sv_fixture_root):
    """The shipped build has no sic_description / market_cap —
    sectors() must return what exists rather than raising
    ColumnNotFoundError (contestants hit this)."""
    from statevector import Dataset
    df = Dataset(sv_fixture_root).sectors()
    assert "ticker" in df.columns
    assert "sic_description" not in df.columns
    assert len(df) == 3


# -- holdout_cutoff / _last_trading_day: stems, not a full scan -------------

def test_last_trading_day_falls_back_to_scan(sv_fixture_root):
    """Single-file canonical build: no partitioned dir -> trading_days()."""
    from statevector import Dataset
    ds = Dataset(sv_fixture_root)
    assert ds._last_trading_day() == max(ds.trading_days())


def test_last_trading_day_uses_partition_stems(tmp_path):
    """Stems are parsed without scanning file contents."""
    from statevector import Dataset
    part = tmp_path / "data" / "raw" / "massive" / "stocks_daily"
    part.mkdir(parents=True)
    for d in ("2020-01-02", "2020-06-30", "2020-03-16"):
        (part / f"{d}.parquet").touch()
    ds = Dataset(tmp_path)
    assert ds._last_trading_day() == date(2020, 6, 30)


def test_last_trading_day_uses_remote_manifest(sv_fixture_root):
    """Remote mode reads index.json file stems — no HTTP."""
    from statevector import Dataset
    ds = Dataset(sv_fixture_root)
    ds.base = "https://example.invalid"
    ds._index = {"panels": {"stocks_daily": {"files": [
        "raw/massive/stocks_daily/2021-03-05.parquet",
        "raw/massive/stocks_daily/2020-11-25.parquet",
    ]}}}
    assert ds._last_trading_day() == date(2021, 3, 5)
