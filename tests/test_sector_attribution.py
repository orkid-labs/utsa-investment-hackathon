#!/usr/bin/env python3
"""Unit tests for sector_attribution + Dataset.sector_map — the
institutional attribution surface over ticker_details sectors."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset, sector_attribution  # noqa: E402


def _rets(cols_returns: dict, n_days: int = 5):
    data = {}
    for t, r in cols_returns.items():
        arr = np.full(n_days, r, dtype=float)
        data[t] = arr
    return pl.DataFrame(data)


def _mkroot(tmp_path: Path) -> Path:
    """Isolated minimal dataset root — does NOT touch the shared
    session fixture (this file sorts before the suite's sector tests)."""
    mdir = tmp_path / "data" / "raw" / "massive"
    mdir.mkdir(parents=True)
    pl.DataFrame([{
        "ticker": t, "name": f"{t} Inc", "primary_exchange": "XNYS",
    } for t in ["A", "B", "C"]]).write_parquet(
        mdir / "reference_tickers.parquet")
    return tmp_path


def test_sector_attribution_basic():
    rets = _rets({"A": 0.01, "B": 0.02, "C": -0.01})
    sec = {"A": "tech", "B": "tech", "C": "energy"}
    out = sector_attribution(rets, ["A", "B", "C"], [0.5, 0.3, 0.2], sec)
    assert out["exposure"] == pytest.approx({"tech": 0.8, "energy": 0.2})
    want_tech = 0.5 * (1.01 ** 5 - 1) + 0.3 * (1.02 ** 5 - 1)
    want_nrg = 0.2 * (0.99 ** 5 - 1)
    assert out["contribution"]["tech"] == pytest.approx(want_tech,
                                                      abs=1e-6)
    assert out["contribution"]["energy"] == pytest.approx(want_nrg,
                                                        abs=1e-6)
    assert out["unmapped_weight"] == 0.0
    assert out["n_unmapped"] == 0


def test_sector_attribution_unmapped_buckets():
    rets = _rets({"A": 0.01, "B": 0.02})
    # B missing from map, C maps to None, D maps to nan
    sec = {"A": "tech", "C": None, "D": float("nan")}
    out = sector_attribution(
        rets, ["A", "B", "C", "D"], [0.4, 0.3, 0.2, 0.1], sec)
    assert out["exposure"]["unmapped"] == pytest.approx(0.6)
    assert out["unmapped_weight"] == pytest.approx(0.6)
    assert out["n_unmapped"] == 3
    assert out["exposure"]["tech"] == pytest.approx(0.4)


def test_sector_attribution_unpriced_leg_contributes_zero():
    rets = _rets({"A": 0.01})
    out = sector_attribution(rets, ["A", "B"], [0.6, 0.4],
                             {"A": "tech", "B": "energy"})
    assert out["exposure"] == pytest.approx({"tech": 0.6, "energy": 0.4})
    assert out["contribution"]["energy"] == 0.0
    assert out["contribution"]["tech"] > 0


def test_sector_attribution_nan_leg_returns():
    data = pl.DataFrame({"A": [0.01, None, 0.02], "B": [0.0, 0.0, 0.0]})
    out = sector_attribution(data, ["A", "B"], [0.5, 0.5],
                             {"A": "x", "B": "x"})
    want = 0.5 * (1.01 * 1.02 - 1)
    assert out["contribution"]["x"] == pytest.approx(want, abs=1e-6)


def test_sector_attribution_empty():
    out = sector_attribution(pl.DataFrame(), [], [], {})
    assert out == {"exposure": {}, "contribution": {},
                   "unmapped_weight": 0.0, "n_unmapped": 0}


def test_dataset_sector_map(tmp_path):
    """sector_map(): absent sector column -> None; real labels join
    through; unclassified tickers stay None."""
    root = _mkroot(tmp_path)
    m = Dataset(str(root)).sector_map()
    assert set(m) == {"A", "B", "C"}
    assert all(v is None for v in m.values())

    sdir = root / "data" / "structural"
    sdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "ticker": ["A", "B"],
        "sector": ["tech", "energy"],
    }).write_parquet(sdir / "ticker_details.parquet")
    m2 = Dataset(str(root)).sector_map()
    assert m2 == {"A": "tech", "B": "energy", "C": None}
