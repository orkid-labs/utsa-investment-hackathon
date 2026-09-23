"""Point-in-time helpers — the leakage guards packaged as functions.

Fundamentals are only valid once the SEC filing exists. Prefer these over
raw panel scans so a model only "sees" what it could have seen.
"""

from __future__ import annotations

from datetime import date, timedelta


def _as_date(v):
    if v is None or isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])

import polars as pl


def _cal_lf(ds) -> pl.LazyFrame:
    return ds._scan("report_calendar_us").filter(
        pl.col("filing_date").is_not_null())


def report_calendar(ds, ticker: str | None = None):
    """Filings calendar (ticker, filing_date, form_type, accession_number)."""
    lf = _cal_lf(ds)
    if ticker:
        lf = lf.filter(pl.col("ticker") == ticker)
    return lf.sort("filing_date").collect().to_pandas()


def asof_fundamentals(ds, ticker: str, asof: str | date):
    """Fundamental rows for `ticker` whose filings existed by `asof`.

    Returns pandas DataFrame. Falls back to fiscal-date filtering if the
    report calendar is empty (PIT caveat — see DEVIATIONS.md).
    """
    if isinstance(asof, str):
        asof = date.fromisoformat(asof)

    t = ds.panel("fundamentals_actuals").filter(pl.col("ticker") == ticker)

    cal = (
        _cal_lf(ds)
        .filter((pl.col("ticker") == ticker) & (pl.col("filing_date").cast(pl.Date) <= _as_date(asof)))
        .select("filing_date")
        .collect()
    )
    if cal.is_empty():
        # no filing rows yet — fall back to period-end only
        return t.filter(pl.col("date").cast(pl.Date) <= _as_date(asof)).collect().to_pandas()

    visible = (
        t.join(cal.lazy(), how="cross")
        .filter(
            # filing must land within (period_end+5d, period_end+120d]:
            # a filing years later doesn't retroactively reveal the row
            (pl.col("filing_date") > pl.col("date") + pl.duration(days=5))
            & (pl.col("filing_date") <= pl.col("date") + pl.duration(days=120))
        )
        .select(t.collect_schema().names())
    )
    return (
        visible.filter(pl.col("date").cast(pl.Date) <= _as_date(asof))
        .sort("date")
        .collect()
        .to_pandas()
    )


def holdout_mask(col: str = "date", holdout_days: int = 30, end: date | None = None):
    """Polars expression marking rows inside the sealed trailing holdout.

    Rows where the mask is True belong to the holdout — exclude them from
    training/validation:

        df = ds.get("stocks_daily", ticker="AAPL")
        # pandas version:
        mask = df["date"] >= holdout_cutoff()
    """
    end = end or date.today()
    cutoff = end - timedelta(days=holdout_days)
    return pl.col(col) >= cutoff


def holdout_cutoff(holdout_days: int = 30, end: date | None = None) -> date:
    """The first date inside the sealed holdout — exclude >= this."""
    end = end or date.today()
    return end - timedelta(days=holdout_days)
