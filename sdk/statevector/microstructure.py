"""Minute bars -> daily microstructure stats per (ticker, date).

    realized_vol   sqrt(Σ r²) · √252 (annualized)
    amihud_illiq   mean(|r| / dollar_vol) · 1e6
    kyle_lambda    slope of Δp on tick-rule-signed volume
    fdt_deviation  mean |cumRV_i/RV − i/N| (vol-timing U-shape residual)
    dollar_vol     Σ close·volume
    vwap_day       Σ close·volume / Σ volume
    n_bars         minute bars that day
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pathlib import Path


def stats_for_file(path: Path) -> pl.DataFrame:
    """One stocks_minute parquet (all tickers, one day) -> per-ticker rows."""
    df = pl.read_parquet(path)
    if df.is_empty():
        return pl.DataFrame()
    out = []
    d = df["date"][0]
    for tk, grp in df.group_by("ticker"):
        c = grp.sort("window_start")
        close = c["close"].to_numpy()
        vol = c["volume"].to_numpy().astype(float)
        n = len(close)
        if n < 30 or close[0] <= 0:
            continue
        lr = np.diff(np.log(close))
        lr = lr[np.isfinite(lr)]
        if len(lr) < 10:
            continue
        rv = float(np.sum(lr * lr))
        realized_vol = float(np.sqrt(rv * 252))
        dvol = close[1:] * vol[1:]
        amihud = float(np.mean(np.abs(lr) / np.maximum(dvol, 1e-9)) * 1e6)
        # tick-rule signed volume
        sign = np.sign(np.diff(close))
        sign[sign == 0] = 0
        # carry last nonzero sign forward
        for i in range(1, len(sign)):
            if sign[i] == 0:
                sign[i] = sign[i - 1]
        signed = sign * vol[1:]
        if signed.std() > 0:
            kyle = float(np.cov(lr, signed)[0, 1] / np.var(signed) * 1e9)
        else:
            kyle = float("nan")
        cum_rv = np.cumsum(lr * lr)
        frac = cum_rv / cum_rv[-1] if cum_rv[-1] > 0 else np.linspace(0, 1, len(lr))
        ideal = (np.arange(1, len(lr) + 1)) / len(lr)
        fdt = float(np.mean(np.abs(frac - ideal)))
        tv = float((close * vol).sum())
        out.append({
            "ticker": tk[0],
            "date": d,
            "realized_vol": realized_vol,
            "amihud_illiq": amihud,
            "kyle_lambda": kyle,
            "fdt_deviation": fdt,
            "dollar_vol": tv,
            "vwap_day": float(tv / vol.sum()) if vol.sum() > 0 else float("nan"),
            "n_bars": n,
        })
    return pl.DataFrame(out) if out else pl.DataFrame()
