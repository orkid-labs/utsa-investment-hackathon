#!/usr/bin/env python3
"""Regression tests — pinned golden values. These don't just check
properties, they pin the actual numbers so silent math drift fails
loudly. Values verified against BS put-call parity and IV round-trips.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector.greeks import bs_price, bs_greeks, implied_vol  # noqa: E402
from statevector.validate import forward_returns, dual_formulation_scores  # noqa: E402
from statevector.pca import NUMERIC  # noqa: E402


# -- Black-Scholes golden values ------------------------------------------------

def test_bs_price_golden():
    # S=K=100, T=0.25, r=5%, q=0, vol=20%
    assert bs_price(100., 100., 0.25, 0.05, 0., 0.20, "C") == pytest.approx(
        4.614997129602855, abs=1e-9)
    assert bs_price(100., 100., 0.25, 0.05, 0., 0.20, "P") == pytest.approx(
        3.372777178991008, abs=1e-9)


def test_bs_greeks_golden():
    g = bs_greeks(100., 100., 0.25, 0.05, 0., 0.20, "C")
    assert g["delta"] == pytest.approx(0.569460, abs=1e-6)
    assert g["gamma"] == pytest.approx(0.039288, abs=1e-6)
    assert g["vega"] == pytest.approx(0.196440, abs=1e-6)
    assert g["theta"] == pytest.approx(-0.050224, abs=1e-6)
    assert g["rho"] == pytest.approx(0.130828, abs=1e-6)


def test_iv_roundtrip_golden():
    px = bs_price(100., 95., 0.5, 0.03, 0., 0.35, "P")
    assert implied_vol(px, 100., 95., 0.5, 0.03, 0., "P") == pytest.approx(
        0.35, abs=1e-4)


# -- forward returns -------------------------------------------------------------

def test_forward_returns_golden():
    d = pd.DataFrame({
        "ticker": ["X"] * 5,
        "date": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03",
             "2024-01-04", "2024-01-05"]),
        "spot": [100., 101., 103., 102., 105.],
    })
    fwd = forward_returns(d, horizon=1)["fwd"].tolist()
    assert fwd[:4] == pytest.approx(
        [0.01, 103 / 101 - 1, 102 / 103 - 1, 105 / 102 - 1], abs=1e-9)
    assert np.isnan(fwd[4])  # last row has no forward


# -- dual formulation -------------------------------------------------------------

def test_dual_formulation_golden():
    z = np.array([[1.0, -0.5], [0.3, 0.2]])
    w = np.array([0.8, 0.6])
    lin, route = dual_formulation_scores(z, w)
    assert lin.tolist() == pytest.approx([0.5, 0.36], abs=1e-9)
    assert route.tolist() == pytest.approx([0.85440037, 0.26832816],
                                           abs=1e-6)


# -- spec-conformance pins ---------------------------------------------------------

def test_state_vector_feature_count():
    """The spec calls for 27 numeric state-vector features."""
    assert len(NUMERIC) == 27


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
