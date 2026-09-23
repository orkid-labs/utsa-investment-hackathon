"""Starter HTTP API contestants can extend into their hackathon app.

    pip install statevector[serve]
    SV_DATA_ROOT=/path/to/state-vector svq serve --port 8000

Endpoints provided out of the box — extend or replace freely:
    GET /health
    GET /universe
    GET /panels
    GET /panel/{name}?ticker=&start=&end=&limit=
    GET /asof?ticker=&date=                 PIT-safe fundamentals
    GET /calendar?ticker=                   report calendar
    GET /panel/{name}/sample?n=             random sample rows
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache

import polars as pl
from fastapi import FastAPI, HTTPException, Query

from .dataset import Dataset
from .pit import asof_fundamentals, report_calendar

app = FastAPI(title="state-vector dataset", version="0.1.0")


def clean(records: list[dict]) -> list[dict]:
    """NaN/inf -> None so json.dumps never 500s on sparse fields."""
    import math
    return [
        {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
         for k, v in r.items()}
        for r in records
    ]


@lru_cache
def ds() -> Dataset:
    return Dataset(os.environ.get("SV_DATA_ROOT", "."))


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/universe")
def universe(us_only: bool = True) -> list[str]:
    return ds().universe(us_only=us_only)


@app.get("/panels")
def panels() -> dict[str, int]:
    return ds().panels()


@app.get("/panel/{name}")
def panel_rows(
    name: str,
    ticker: str | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int = Query(500, le=10_000),
):
    try:
        lf = ds().panel(name)
    except KeyError as e:
        raise HTTPException(404, str(e))
    cols = lf.collect_schema().names()
    tcol = "ticker" if "ticker" in cols else ("ticker_bbg" if "ticker_bbg" in cols else None)
    if ticker and tcol:
        lf = lf.filter(pl.col(tcol) == ticker)
    if start and "date" in cols:
        lf = lf.filter(pl.col("date") >= start)
    if end and "date" in cols:
        lf = lf.filter(pl.col("date") <= end)
    return clean(lf.limit(limit).collect().to_dicts())


@app.get("/panel/{name}/sample")
def panel_sample(name: str, n: int = Query(50, le=1_000)):
    try:
        lf = ds().panel(name)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return clean(lf.collect().sample(n=n, seed=42).to_dicts())


@app.get("/asof")
def asof(ticker: str, on: date):
    try:
        return clean(asof_fundamentals(ds(), ticker, on).to_dict("records"))
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/calendar")
def calendar(ticker: str):
    cal = report_calendar(ds())
    return clean(cal[cal["ticker"] == ticker].sort_values("filing_date").to_dict("records"))
