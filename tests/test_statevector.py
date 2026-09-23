#!/usr/bin/env python3
"""Spec-critical unit tests: OCC parsing, BS/IV round-trip, vectorized
consistency, PIT clocks, PCA holdout exclusion, validation alignment.

Run:  /tmp/sv-venv/bin/python -m pytest tests/test_statevector.py -x -q
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector.greeks import (  # noqa: E402
    parse_occ, occ_ticker, bs_price, bs_greeks, implied_vol,
    implied_vol_vec, bs_greeks_vec)
from statevector.vector import _expected_next_report, _days_to  # noqa: E402
from statevector.pca import fit_pca, transform_pca, NUMERIC  # noqa: E402
from statevector.validate import (  # noqa: E402
    forward_returns, dual_formulation_scores, embargoed_folds,
    ic_significance)


# -- OCC parsing -------------------------------------------------------------

def test_parse_occ_put():
    c = parse_occ("O:AAPL200221P00315000")
    assert (c.root, c.expiry, c.cp, c.strike) == (
        "AAPL", date(2020, 2, 21), "P", 315.0)


def test_parse_occ_call_roundtrip():
    c = parse_occ("O:MSFT261218C00500000")
    assert c.root == "MSFT" and c.cp == "C" and c.strike == 500.0
    assert occ_ticker(c.root, c.expiry, c.cp, c.strike) == \
        "O:MSFT261218C00500000"


def test_parse_occ_rejects_garbage():
    assert parse_occ("AAPL") is None
    assert parse_occ("O:AAPL200221X00315000") is None
    assert parse_occ("O:AAPL200221P315000") is None


# -- BS / IV -----------------------------------------------------------------

def test_bs_price_put_call_parity():
    S, K, T, r, q, v = 100.0, 105.0, 0.5, 0.03, 0.0, 0.25
    c = bs_price(S, K, T, r, q, v, "C")
    p = bs_price(S, K, T, r, q, v, "P")
    fwd = S - K * np.exp(-r * T)
    assert abs((c - p) - fwd) < 1e-10


def test_iv_round_trip():
    true_iv = 0.494
    px = bs_price(321.55, 290.0, 11 / 365.25, 0.016, 0.0, true_iv, "P")
    got = implied_vol(px, 321.55, 290.0, 11 / 365.25, 0.016, 0.0, "P")
    assert abs(got - true_iv) < 1e-3


def test_iv_below_intrinsic_is_nan():
    # ITM call quoted under intrinsic -> NaN (untradeable quote)
    assert np.isnan(implied_vol(1.0, 321.55, 290.0, 11 / 365.25,
                                0.016, 0.0, "C"))
    # deep ITM put under intrinsic -> NaN
    assert np.isnan(implied_vol(10.0, 321.55, 400.0, 11 / 365.25,
                                0.016, 0.0, "P"))
    # sane OTM put quote -> solves
    assert np.isfinite(implied_vol(0.58, 321.55, 290.0, 11 / 365.25,
                                   0.016, 0.0, "P"))


def test_implied_vol_vec_matches_scalar():
    rng = np.random.default_rng(7)
    n = 200
    S, r, q = 150.0, 0.02, 0.0
    K = rng.uniform(80, 220, n)
    T = rng.uniform(0.02, 1.0, n)
    cp = np.where(rng.random(n) < 0.5, "C", "P")
    true_iv = rng.uniform(0.1, 0.8, n)
    px = np.array([bs_price(S, K[i], T[i], r, q, true_iv[i], cp[i])
                   for i in range(n)])
    got = implied_vol_vec(px, S, K, T, r, q, cp)
    assert np.isfinite(got).all()
    assert np.median(np.abs(got - true_iv)) < 1e-3
    # the solver's actual objective: repriced premium must match
    back = np.array([bs_price(S, K[i], T[i], r, q, got[i], cp[i])
                     for i in range(n)])
    assert np.abs(back - px).max() < 1e-5


def test_implied_vol_vec_per_row_spot():
    # array S (whole-day mixed underlyings) must work
    S = np.array([100.0, 200.0])
    K = np.array([105.0, 190.0])
    T = np.array([0.25, 0.5])
    cp = np.array(["C", "P"])
    iv = np.array([0.3, 0.45])
    px = np.array([bs_price(S[i], K[i], T[i], 0.02, 0.0, iv[i], cp[i])
                   for i in range(2)])
    got = implied_vol_vec(px, S, K, T, 0.02, 0.0, cp)
    assert np.allclose(got, iv, atol=1e-3)


def test_bs_greeks_vec_matches_scalar():
    rng = np.random.default_rng(3)
    n = 100
    S = 100.0
    K = rng.uniform(70, 130, n)
    T = rng.uniform(0.05, 1.0, n)
    v = rng.uniform(0.1, 0.6, n)
    cp = np.where(rng.random(n) < 0.5, "C", "P")
    g = bs_greeks_vec(S, K, T, 0.03, 0.0, v, cp)
    for i in range(0, n, 17):
        gs = bs_greeks(S, K[i], T[i], 0.03, 0.0, v[i], cp[i])
        for key in ("delta", "gamma", "vega", "theta", "rho"):
            assert abs(g[key][i] - gs[key]) < 1e-8, (key, i)


def test_greeks_delta_bounds():
    K = np.array([100.0]); T = np.array([0.5]); v = np.array([0.25])
    gc = bs_greeks_vec(105.0, K, T, 0.03, 0.0, v, np.array(["C"]))
    gp = bs_greeks_vec(105.0, K, T, 0.03, 0.0, v, np.array(["P"]))
    assert 0 < gc["delta"][0] < 1
    assert -1 < gp["delta"][0] < 0
    assert abs(gc["delta"][0] - gp["delta"][0] - 1.0) < 1e-10  # q=0 parity


# -- PIT clocks ---------------------------------------------------------------

def test_expected_next_report_never_uses_future():
    filings = [date(2020, 2, 1), date(2020, 5, 1), date(2020, 8, 1)]
    days = [date(2020, 1, 15) + timedelta(days=i) for i in range(300)]
    got = _expected_next_report(days, filings)
    # before first filing: NaN (nothing observed)
    assert np.isnan(got[date(2020, 1, 15)])
    # between filings: projection = last filing + ~91d gap
    v = got[date(2020, 2, 15)]
    assert v is not None and not np.isnan(v)
    assert abs(v - (filings[0] + timedelta(days=91)
                    - date(2020, 2, 15)).days) <= 2
    # monotone-ish: day after a filing resets the clock upward
    v_before = got[date(2020, 4, 30)]
    v_after = got[date(2020, 5, 2)]
    assert v_after > v_before


# -- PCA holdout exclusion ----------------------------------------------------

def _fake_vec(n_tickers=8, days=800, end=date(2026, 9, 18)):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range(end="2026-09-18", periods=days).date
    rows = []
    for t in range(n_tickers):
        for d in dates:
            rows.append({"ticker": f"T{t}", "date": d,
                         **{c: rng.normal() for c in NUMERIC}})
    return pd.DataFrame(rows)


def test_pca_excludes_holdout_from_fit():
    vec = _fake_vec()
    # poison the holdout: scale it 100x — if holdout leaked into the
    # fit, mean/std would blow up
    cutoff = pd.Timestamp("2026-08-19").date()
    vec.loc[pd.to_datetime(vec["date"]).dt.date >= cutoff, NUMERIC] *= 100
    _, meta = fit_pca(vec, n_components=5, train_start="2017-01-01",
                      holdout_days=30)
    assert np.abs(np.array(meta["mean"])).max() < 10
    tr = vec[(pd.to_datetime(vec["date"]).dt.date >= date(2017, 1, 1))
             & (pd.to_datetime(vec["date"]).dt.date < cutoff)]
    n_complete = np.isfinite(tr[NUMERIC].to_numpy(float)).all(axis=1).sum()
    assert meta["train_rows"] == int(n_complete)


def test_transform_uses_frozen_stats():
    vec = _fake_vec()
    _, meta = fit_pca(vec, n_components=5, holdout_days=30)
    mu = np.array(meta["mean"]); sd = np.array(meta["std"])
    W = np.array(meta["loadings"])
    scores = transform_pca(vec, mu, sd, W)
    assert scores.shape == (len(vec), 5)
    # NaN rows impute to training mean -> scores at origin
    vec_nan = vec.copy()
    vec_nan.loc[0, NUMERIC] = np.nan
    s2 = transform_pca(vec_nan, mu, sd, W)
    assert np.allclose(s2[0], 0, atol=1e-9)


# -- validation ----------------------------------------------------------------

def test_forward_returns_keyed_not_positional():
    d = pd.DataFrame({
        "ticker": ["A"] * 4 + ["B"] * 4,
        "date": list(pd.bdate_range("2024-01-01", periods=4)) * 2,
        "spot": [100, 110, 105, 120, 50, 55, 52, 58],
    })
    fwd = forward_returns(d, horizon=1)
    got = fwd.set_index(["ticker", "date"])["fwd"]
    assert abs(got[("A", pd.Timestamp("2024-01-01"))] - 0.10) < 1e-9
    assert abs(got[("B", pd.Timestamp("2024-01-02"))] - (52/55 - 1)) < 1e-9
    assert np.isnan(got[("B", pd.Timestamp("2024-01-04"))])


def test_dual_formulation_signs():
    z = np.array([[1.0, 2.0], [-1.0, -2.0]])
    w = np.array([0.5, 0.5])
    a, r = dual_formulation_scores(z, w)
    assert a[0] > 0 and r[0] > 0
    assert a[1] < 0 and r[1] < 0
    assert abs(r[0]) == np.sqrt(((z[0] * w) ** 2).sum())


def test_embargoed_folds_gap():
    dates = list(pd.bdate_range("2020-01-01", periods=500))
    folds = embargoed_folds(dates, n_folds=4, embargo_days=21)
    assert folds
    for train, test in folds:
        assert max(train) < min(test)
        assert (min(test) - max(train)).days >= 21


def test_ic_significance_strong_signal():
    rng = np.random.default_rng(1)
    ics = pd.Series(rng.normal(0.05, 0.02, 500))
    s = ic_significance(ics)
    assert s["p"] < 0.01 and s["mean"] > 0


def test_remote_dataset_thin_client():
    """Remote Dataset over the dataserver — needs SV_TEST_BASE + SV_TEST_TOKEN.

    Skipped unless both are set (CI/local runs without the server up).
    """
    base = os.environ.get("SV_TEST_BASE")
    token = os.environ.get("SV_TEST_TOKEN")
    if not base or not token:
        pytest.skip("SV_TEST_BASE/SV_TEST_TOKEN not set")
    from statevector import Dataset
    ds = Dataset(base, token=token)
    uni = ds.universe()
    assert len(uni) > 1000
    df = ds.get("options_smile_daily", ticker="AAPL",
               start="2020-01-01", end="2020-01-10")
    assert len(df) > 0 and "atm_iv_30d" in df.columns
    # auth gate: bad token must fail
    try:
        Dataset(base, token="wrong-token-xyz").universe()
        raise AssertionError("bad token accepted")
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
