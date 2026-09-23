"""Production thin-client data server — hosts the dataset read-only.

    SV_DATA_ROOT=/path/to/state-vector SV_TOKEN=<bearer> \
        uvicorn statevector.dataserver:app --host 127.0.0.1 --port 8471

Two surfaces over the same dataset:

  1. Static parquet/text under /data/** (Range-enabled, so remote
     ``pl.scan_parquet`` does lazy row-group reads — this is what
     ``Dataset("https://…", token=…)`` uses).
  2. JSON convenience endpoints (/universe, /panel/{name}, /asof, …)
     identical to the contestant starter app.

Safety: GET/HEAD only; when SV_TOKEN is set every path except /health
requires ``Authorization: Bearer <SV_TOKEN>`` or ``?key=<SV_TOKEN>``.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .dataset import Dataset
from .pit import asof_fundamentals, report_calendar

DATA = Path(os.environ.get("SV_DATA_ROOT", ".")) / "data"
TOKEN = os.environ.get("SV_TOKEN")

app = FastAPI(title="state-vector data server", version="1.0.0")


@app.middleware("http")
async def gate(request: Request, call_next):
    # read-only surface — nothing else is ever valid here
    if request.method not in ("GET", "HEAD"):
        return JSONResponse({"detail": "read-only"}, status_code=405)
    if TOKEN and request.url.path != "/health":
        auth = request.headers.get("authorization", "")
        key = request.query_params.get("key", "")
        if auth != f"Bearer {TOKEN}" and key != TOKEN:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


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
    return {"ok": True, "data": DATA.exists(), "auth": bool(TOKEN)}


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


# static parquet/text with HTTP range support — the thin-client surface.
# mounted last so /panel/* routes win.
app.mount("/data", StaticFiles(directory=DATA), name="data")
