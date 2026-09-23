"""Options smile panel: per (ticker, date) implied-vol features.

Scans contract-level day-aggs (options_daily, O: OCC tickers), inverts
BS-2002 on the close, and aggregates a compact smile:

    atm_iv_30d / atm_iv_90d   nearest-ATM IV at ~30d / ~90d buckets
    skew_25d                  IV(5% OTM puts) − IV(5% OTM calls), / atm
    rn_kurtosis               smile-curvature excess-kurtosis approx
    term_slope                atm_iv_90d − atm_iv_30d
    n_contracts               contracts used (liquidity proxy)

Approximations (documented in DEVIATIONS): 25Δ wings proxied by 5% OTM
strikes; rn_kurtosis from a quadratic smile fit; q=0 (no div yield).
"""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from .greeks import implied_vol_vec, bs_greeks_vec, parse_occ


OCC_PAT = r"^O:([A-Z]+)(\d{6})([CP])(\d{8})$"


def _contracts(df: pl.DataFrame) -> pl.DataFrame:
    """Parse OCC tickers -> root, expiry, cp, strike; keep tradeable rows."""
    parsed = (
        df.lazy()
        .filter((pl.col("close") > 0) & (pl.col("volume") > 0)
                & pl.col("ticker").str.contains(OCC_PAT))
        .with_columns(
            pl.col("ticker").str.extract_groups(OCC_PAT)
            .alias("_occ"),
        )
        .unnest("_occ")
        .rename({"1": "root", "2": "ymd", "3": "cp", "4": "k"})
        .with_columns([
            pl.col("k").cast(pl.Float64).truediv(1000).alias("strike"),
            (pl.lit(2000) + pl.col("ymd").str.slice(0, 2).cast(pl.Int32))
            .alias("_yy"),
            pl.col("ymd").str.slice(2, 2).cast(pl.Int32).alias("_mm"),
            pl.col("ymd").str.slice(4, 2).cast(pl.Int32).alias("_dd"),
        ])
        .with_columns(
            pl.date("_yy", "_mm", "_dd").alias("expiry"),
        )
        .with_columns(
            ((pl.col("expiry") - pl.col("date")).dt.total_days() / 365.25)
            .alias("T"),
        )
        .filter((pl.col("T") >= 7 / 365.25) & (pl.col("T") <= 370 / 365.25))
        .select([
            "root", pl.col("ticker").alias("contract"), "date",
            pl.col("close").alias("opt_close"), "volume",
            "expiry", "cp", "strike", "T",
        ])
        .collect()
    )
    return parsed


def smile_stats(chain: pl.DataFrame) -> dict | None:
    """One ticker-day chain -> smile aggregates. chain already has
    mny, T, cp, iv columns (mny = log K/S, |mny|<=0.5, iv solved)."""
    if chain.is_empty():
        return None
    mny = chain["mny"].to_numpy()
    T_ = chain["T"].to_numpy()
    cp_ = chain["cp"].to_numpy()
    iv = chain["iv"].to_numpy()
    good = np.isfinite(iv) & (iv > 0.005) & (iv < 5)
    if good.sum() < 3:
        return None
    mny, T_, cp_, iv = mny[good], T_[good], cp_[good], iv[good]

    def atm(bucket):
        sel = (np.abs(T_ - bucket) < bucket * 0.6) & (np.abs(mny) < 0.15)
        if sel.sum() == 0:
            return float("nan")
        i = np.argmin(np.abs(mny[sel]) * 10 + np.abs(T_[sel] - bucket))
        return float(iv[sel][i])

    atm30, atm90 = atm(30 / 365.25), atm(90 / 365.25)

    wing_put = (cp_ == "P") & (mny > -0.12) & (mny < -0.03) & (np.abs(T_ - 30 / 365.25) < 20 / 365.25)
    wing_call = (cp_ == "C") & (mny > 0.03) & (mny < 0.12) & (np.abs(T_ - 30 / 365.25) < 20 / 365.25)
    skew = float("nan")
    if wing_put.any() and wing_call.any() and np.isfinite(atm30) and atm30 > 0:
        skew = float((iv[wing_put].mean() - iv[wing_call].mean()) / atm30)

    # quadratic smile fit iv ≈ a + b·m + c·m² -> excess kurtosis ≈ 6c·T
    rn_kurt = float("nan")
    sel = np.abs(T_ - 30 / 365.25) < 20 / 365.25
    if sel.sum() >= 6 and np.ptp(mny[sel]) > 0.05:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                a_, b_, c_ = np.polyfit(mny[sel], iv[sel], 2)
                if np.isfinite(c_) and c_ > 0:
                    rn_kurt = float(6.0 * c_ * (30 / 365.25))
            except (np.linalg.LinAlgError, ValueError):
                pass

    return {
        "atm_iv_30d": atm30,
        "atm_iv_90d": atm90,
        "skew_25d": skew,
        "rn_kurtosis": rn_kurt,
        "term_slope": (atm90 - atm30)
        if np.isfinite(atm90) and np.isfinite(atm30) else float("nan"),
        "n_contracts": int(good.sum()),
    }


def build_day(opt_path: Path, spot_map: dict[str, float],
              rate_map: dict[date, float]) -> pl.DataFrame:
    """Process one options_daily parquet file -> smile rows for the day."""
    df = pl.read_parquet(opt_path)
    if df.is_empty():
        return pl.DataFrame()
    ch = _contracts(df)
    if ch.is_empty():
        return pl.DataFrame()
    d = df["date"][0]
    r = rate_map.get(d, 0.04)
    spots = pl.DataFrame({"root": list(spot_map.keys()),
                          "_spot": list(spot_map.values())})
    ch = ch.join(spots, on="root")
    ch = ch.filter(pl.col("_spot") > 0)
    if ch.is_empty():
        return pl.DataFrame()
    ch = ch.with_columns(
        (pl.col("strike") / pl.col("_spot")).log().alias("mny"))
    ch = ch.filter(pl.col("mny").abs() <= 0.5)
    if ch.is_empty():
        return pl.DataFrame()
    # one Newton solve across the whole day's contracts
    iv = implied_vol_vec(
        ch["opt_close"].to_numpy(), ch["_spot"].to_numpy(),
        ch["strike"].to_numpy(), ch["T"].to_numpy(), r, 0.0,
        ch["cp"].to_numpy())
    ch = ch.with_columns(pl.Series("iv", iv))
    out = []
    for root, grp in ch.group_by("root"):
        res = smile_stats(grp)
        if res:
            out.append({"ticker": root[0], "date": d,
                        "spot": grp["_spot"][0], **res})
    return pl.DataFrame(out) if out else pl.DataFrame()


def greeks_for_day(opt_path: Path, spot_map: dict[str, float],
                   rate_map: dict[date, float]) -> pl.DataFrame:
    """BS2002 Greeks for every tradeable contract in one options_daily
    file -> (contract, root, date, spot, strike, expiry, cp, T, close,
    volume, iv, delta, gamma, vega, theta, rho)."""
    df = pl.read_parquet(opt_path)
    if df.is_empty():
        return pl.DataFrame()
    ch = _contracts(df)
    if ch.is_empty():
        return pl.DataFrame()
    d = df["date"][0]
    r = rate_map.get(d, 0.04)
    spots = pl.DataFrame({"root": list(spot_map.keys()),
                          "_spot": list(spot_map.values())})
    ch = ch.join(spots, on="root").filter(pl.col("_spot") > 0)
    if ch.is_empty():
        return pl.DataFrame()
    iv = implied_vol_vec(
        ch["opt_close"].to_numpy(), ch["_spot"].to_numpy(),
        ch["strike"].to_numpy(), ch["T"].to_numpy(), r, 0.0,
        ch["cp"].to_numpy())
    ch = ch.with_columns(pl.Series("iv", iv)).filter(
        pl.col("iv").is_finite())
    if ch.is_empty():
        return pl.DataFrame()
    g = bs_greeks_vec(ch["_spot"].to_numpy(), ch["strike"].to_numpy(),
                      ch["T"].to_numpy(), r, 0.0,
                      ch["iv"].to_numpy(), ch["cp"].to_numpy())
    return ch.rename({"_spot": "spot"}).with_columns([
        pl.Series("delta", g["delta"]), pl.Series("gamma", g["gamma"]),
        pl.Series("vega", g["vega"]), pl.Series("theta", g["theta"]),
        pl.Series("rho", g["rho"]),
    ]).select([
        "contract", "root", "date", "spot", "strike", "expiry", "cp",
        "T", pl.col("opt_close").alias("close"), "volume",
        "iv", "delta", "gamma", "vega", "theta", "rho",
    ])
