"""Regression tests: exact period_end->filing resolution in
vector._asof_actuals once report_calendar_us carries period_end."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdk"))

from statevector.vector import _asof_actuals  # noqa: E402


def _cal(rows):
    return pd.DataFrame(rows)


def _act(period_end: date):
    return pd.DataFrame([{
        "ticker": "X", "date": period_end,
        "cf_free_cash_flow": 1.0, "sales_rev_turn": 2.0,
    }])


def test_late_filer_uses_real_filing_date_not_plus45():
    """A filing >95d after period end is the real knowable date —
    the +45d fallback would leak actuals before they existed."""
    pe = date(2020, 3, 31)
    filed_late = date(2020, 8, 10)  # 132d after period end
    cal = _cal([{"ticker": "X", "filing_date": filed_late,
                 "period_end": pe}])
    dates = [pe, date(2020, 5, 20), filed_late]
    out = _asof_actuals(_act(pe), cal, "X", dates)
    assert pd.isna(out["asof_date"].iloc[0])
    assert pd.isna(out["asof_date"].iloc[1])  # +45d would be May 15
    assert pd.Timestamp(out["asof_date"].iloc[2]) == \
        pd.Timestamp(filed_late)


def test_normal_filer_exact_match():
    pe = date(2020, 3, 31)
    filed = date(2020, 5, 8)
    cal = _cal([{"ticker": "X", "filing_date": filed,
                 "period_end": pe}])
    out = _asof_actuals(_act(pe), cal, "X", [pe, filed])
    assert pd.isna(out["asof_date"].iloc[0])
    assert pd.Timestamp(out["asof_date"].iloc[1]) == \
        pd.Timestamp(filed)


def test_uncovered_period_falls_back_to_window_then_45d():
    pe = date(2020, 3, 31)
    cal = _cal([{"ticker": "X", "filing_date": date(2021, 1, 15),
                 "period_end": date(2020, 12, 31)}])
    expected = pe + pd.Timedelta(days=45)
    out = _asof_actuals(_act(pe), cal, "X", [expected])
    assert pd.Timestamp(out["asof_date"].iloc[0]) == \
        pd.Timestamp(expected)


def test_calendar_without_period_end_column_unchanged():
    pe = date(2020, 3, 31)
    filed = date(2020, 5, 8)
    cal = _cal([{"ticker": "X", "filing_date": filed}])
    out = _asof_actuals(_act(pe), cal, "X", [filed])
    assert pd.Timestamp(out["asof_date"].iloc[0]) == \
        pd.Timestamp(filed)


def test_amended_period_uses_first_filing():
    """Original + amendment share a period_end — the first public
    availability is the knowable date."""
    pe = date(2020, 3, 31)
    cal = _cal([
        {"ticker": "X", "filing_date": date(2020, 9, 1),
         "period_end": pe},
        {"ticker": "X", "filing_date": date(2020, 5, 8),
         "period_end": pe},
    ])
    out = _asof_actuals(_act(pe), cal, "X", [date(2020, 5, 8)])
    assert pd.Timestamp(out["asof_date"].iloc[0]) == \
        pd.Timestamp(date(2020, 5, 8))
