"""Hackathon starter app — a working portfolio-management API.

This app ALREADY WORKS end-to-end on the dataset. Your job: make it better —
smarter holdings, better screens, real analytics — without breaking the
judged endpoints' response shapes.

TRADING RULES (enforced by the rubric):
  - long-only: every weight >= 0, weights sum to ~1.0
  - stocks: buy and sell to close; never short
  - options: buy calls/puts and sell to close; never short. To hedge a
    long book, buy a protective put — that's the mandate.
  - option legs use OCC tickers: O:AAPL260923C00245000

Run:
    pip install -r requirements.txt
    export SV_DATA_ROOT=/path/to/state-vector
    uvicorn app:app --port 8000

Judged endpoints (keep these paths + response shapes):
    GET  /health
    GET  /portfolio/holdings
    POST /backtest
    GET  /screen
    GET  /asof
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from statevector import (
    Dataset,
    parse_occ,  # noqa: F401
    run_backtest,
)
from statevector.backtest import REBALANCE_CHOICES, apply_split_factors

ds = Dataset()
app = FastAPI(title="hackathon portfolio api", version="0.1.0")

TRADING_DAYS = 252


def clean(records: list[dict]) -> list[dict]:
    """NaN/inf -> None so sparse fundamentals never 500 on json.dumps."""
    import math
    return [
        {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
         for k, v in r.items()}
        for r in records
    ]


def is_option(t: str) -> bool:
    """OCC contract tickers start with 'O:'. Option legs are supported by
    the mandate but NOT implemented here — that's on you."""
    return t.startswith("O:")


# ---------------------------------------------------------------- health -----

@app.get("/health")
def health() -> dict:
    return {"ok": True, "dataset_root": str(ds.root)}


# ------------------------------------------------------------- portfolio -----

@app.get("/portfolio/holdings")
def holdings(n: int = Query(10, ge=1, le=100)) -> dict:
    """Reference implementation: equal-weight, long-only portfolio of the
    top-N most liquid names (20-day average dollar volume). Replace with
    your own selection logic — the rubric only requires the response shape.

    Option legs are allowed in the same array: {"ticker": "O:SPY...P..."}.
    """
    # ADV20 needs only a trailing window — bound the scan so the remote
    # thin client fetches ~30 day-files instead of all of them.
    cutoff = str(date.today() - timedelta(days=45))
    px = ds._scan("stocks_daily", start=cutoff)
    top = (
        px.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_vol"))
        .group_by("ticker")
        .agg(pl.col("dollar_vol").tail(20).mean().alias("adv20"))
        .sort("adv20", descending=True)
        .limit(n)
        .collect()
    )
    w = round(1.0 / len(top), 8) if len(top) else 0.0
    return {
        "as_of": None,
        "method": "equal_weight_top_liquidity",
        "holdings": [
            {"ticker": t, "weight": w} for t in top["ticker"].to_list()
        ],
    }


# -------------------------------------------------------------- backtest -----

class BacktestRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)   # stocks and/or O: option legs
    weights: list[float] | None = None          # default: equal weight, all >= 0
    start: date
    end: date
    rebalance: str = "none"                     # none|daily|weekly|monthly
    cost_bps: float | None = None               # flat per-leg override; None = schedule


@app.post("/backtest")
def backtest(req: BacktestRequest) -> dict:
    """Daily close-to-close long-only backtest.

    Stock legs price from stocks_daily. Option legs (O: tickers) are part
    of the mandate — pricing them from options_daily is your problem.
    A leg with no print on a given day contributes 0 return that day.
    """
    if req.end <= req.start:
        raise HTTPException(400, "end must be after start")

    weights = req.weights or [1.0 / len(req.tickers)] * len(req.tickers)
    if len(weights) != len(req.tickers):
        raise HTTPException(400, "weights length must match tickers")
    if any(w < 0 for w in weights):
        raise HTTPException(400, "long-only: all weights must be >= 0")
    if abs(sum(weights) - 1.0) > 0.01:
        raise HTTPException(400, "weights must sum to ~1.0")
    if req.rebalance not in REBALANCE_CHOICES:
        raise HTTPException(400, f"rebalance must be one of {REBALANCE_CHOICES}")
    if req.cost_bps is not None and req.cost_bps < 0:
        raise HTTPException(400, "cost_bps must be >= 0")

    wide = (
        ds._scan("stocks_daily", start=str(req.start), end=str(req.end))
        .filter(
            pl.col("ticker").is_in(req.tickers)
            & pl.col("date").is_between(req.start, req.end)
        )
        .select(["date", "ticker", "close"])
        .collect()
        .pivot(on="ticker", index="date", values="close")
        .sort("date")
    )
    if wide.height < 3:
        raise HTTPException(400, "insufficient data in range")

    cols = [c for c in wide.columns if c != "date"]
    rets = wide.select(
        [pl.col(c) / pl.col(c).shift(1) - 1.0 for c in cols]
    ).fill_null(0.0)
    if rets.height == 0:
        raise HTTPException(400, "no overlapping return days")
    # split-adjust so an ex-date inside the window is priced correctly —
    # stocks_daily close is raw; corporate_actions carries the factors
    rets = apply_split_factors(
        rets, wide["date"].to_list(), ds.splits(req.tickers,
                                               req.start, req.end))

    metrics = run_backtest(
        rets, wide["date"].to_list(), req.tickers, weights,
        rebalance=req.rebalance, cost_bps=req.cost_bps,
    )
    if not metrics:
        raise HTTPException(400, "no overlapping return days")

    return {
        "tickers": req.tickers,
        "weights": weights,
        "start": str(req.start),
        "end": str(req.end),
        "rebalance": req.rebalance,
        **{k: (round(v, 4) if k == "sharpe" else round(v, 6))
           for k, v in metrics.items()},
    }


# ----------------------------------------------------------------- screen ----

@app.get("/screen")
def screen(
    min_adv: float = Query(0, description="min 20d avg dollar volume"),
    sector_contains: str | None = None,
    limit: int = Query(25, le=200),
) -> dict:
    """Reference screen: liquidity filter + optional industry substring."""
    px = ds._scan("stocks_daily", start=str(date.today() - timedelta(days=45)))
    adv = (
        px.with_columns((pl.col("close") * pl.col("volume")).alias("dv"))
        .group_by("ticker")
        .agg(pl.col("dv").tail(20).mean().alias("adv20"))
        .filter(pl.col("adv20") >= min_adv)
    )
    out = adv.collect().to_pandas()
    if sector_contains:
        try:
            sec = ds.sectors()
            match = sec[
                sec["sic_description"].str.contains(sector_contains, case=False, na=False)
            ]["ticker"]
            out = out[out["ticker"].isin(match)]
        except FileNotFoundError:
            pass
    out = out.sort_values("adv20", ascending=False).head(limit)
    return {"count": int(len(out)), "results": out.to_dict("records")}


# ------------------------------------------------------------------- asof ----

@app.get("/asof")
def asof(ticker: str, on: date):
    """PIT-safe fundamentals — what a model could have seen on `on`."""
    try:
        return clean(ds.fundamentals(ticker, asof=str(on)).to_dict("records"))
    except Exception as e:
        raise HTTPException(400, str(e))
