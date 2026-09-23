"""Shared fixtures: a synthetic point-in-time dataset root so the SDK,
template app, and rubric checker can all run locally with no network
and no real data.

Layout mirrors the real dataset root:

    root/
      data/canonical/stocks_daily.parquet        (ticker,date,close,volume)
      data/canonical/fundamentals_actuals.parquet
      data/structural/report_calendar_us.parquet
      data/structural/universe_us.txt
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

TICKERS = ["AAPL", "MSFT", "NVDA"]


def _bizdays(start: date, end: date) -> list[date]:
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@pytest.fixture(scope="session")
def sv_fixture_root(tmp_path_factory) -> Path:
    """A tiny but structurally real dataset root."""
    import polars as pl

    root = tmp_path_factory.mktemp("svroot")
    (root / "data/canonical").mkdir(parents=True)
    (root / "data/structural").mkdir(parents=True)

    # stocks_daily: 2020 (for the rubric backtest window) + a recent
    # 60-day window (for ADV20-style holdings/screen queries).
    days = _bizdays(date(2020, 1, 1), date(2020, 12, 31)) + \
        _bizdays(date.today() - timedelta(days=60), date.today())
    rows = []
    for i, tk in enumerate(TICKERS):
        for j, d in enumerate(days):
            base = 100.0 + i * 50.0
            rows.append({
                "ticker": tk, "date": d,
                "close": base * (1 + 0.0004 * j),   # gentle uptrend
                "volume": 1_000_000 * (i + 1),       # AAPL<MSFT<NVDA liq order
            })
    pl.DataFrame(rows).write_parquet(root / "data/canonical/stocks_daily.parquet")

    # fundamentals + filings calendar so asof() exercises the PIT join:
    # a 2024-02-01 period-end filing that landed 2024-02-10 (inside the
    # +5..+120d window) — visible on 2024-03-31 but not on 2024-02-05.
    pl.DataFrame([{
        "ticker": "AAPL", "date": date(2024, 2, 1),
        "revenue": 1.0e11, "fcf_per_share": 5.0,
    }]).write_parquet(root / "data/canonical/fundamentals_actuals.parquet")

    pl.DataFrame([{
        "ticker": "AAPL", "filing_date": date(2024, 2, 10),
        "form_type": "10-Q", "accession_number": "0000-24-000001",
    }]).write_parquet(root / "data/structural/report_calendar_us.parquet")

    (root / "data/structural/universe_us.txt").write_text(
        "\n".join(TICKERS) + "\n")

    return root
