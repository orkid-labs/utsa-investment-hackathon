"""State vector assembly — 27 numeric features + 2 clocks + 7 flags
per (ticker, date) per features.yaml v1.1.

Inputs (canonical/ + raw/massive/):
    stocks_daily, microstructure_daily, options_smile_daily,
    options_agg_daily, valuation_weekly, fundamentals_actuals,
    fundamentals_estimates_recent, guidance_all, macro_* (5 panels),
    report_calendar_us, index_daily

Methodology notes (documented simplifications):
    - z-scores use trailing 252-trading-day windows (PIT-safe)
    - guidance track preferred; snapshot track fills residual coverage
      (flag snapshot_track_used=1). guidance_absent=1 only where the
      ticker has no guidance history at all (~14% of universe).
    - 25Δ wings proxied by ~5% OTM strikes (options_smile.py)
    - rn_kurtosis from quadratic smile fit ≈ 6·c·T
    - half-life/Kalman fall back to imputed constants when the event
      series is too short (flag half_life_imputed)
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "fwd_fcf_fair_value", "fundamental_surprise", "kl_surprise_bits",
    "measured_half_life", "growth_kalman_update", "fundamental_confidence",
    "log_fv_gap", "guidance_range_velocity",
    "composite_valuation_gap", "valuation_kurtosis", "cornish_fisher_gap",
    "mean_reversion_speed",
    "minute_realized_diffusion", "amihud_illiq", "kyle_lambda",
    "fdt_deviation", "liquidity_roc",
    "atm_iv", "variance_risk_premium", "skew_25d", "rn_kurtosis",
    "oi_divergence", "term_slope",
    "treasury_funding_interact", "inflation_expectation", "funding_stress",
    "mktcap_duration_interact",
]
CLOCK = ["days_to_next_report", "days_to_opex"]
FLAGS = [
    "snapshot_track_used", "guidance_absent", "options_thin_chain",
    "spread_filtered", "half_life_imputed", "staleness_quarterly",
    "regime_calib_error_quantile",
]


def _scan(root: Path, name: str) -> pd.DataFrame:
    """canonical/{name}/ dir, canonical/{name}.parquet, or
    structural/{name}.parquet -> pandas."""
    import polars as pl
    for p in (root / "data" / "canonical" / name,
              root / "data" / "structural" / name):
        if p.is_dir():
            return pl.concat(
                [pl.read_parquet(f) for f in sorted(p.glob("*.parquet"))],
                how="diagonal_relaxed",
            ).to_pandas()
        f = p.with_suffix(".parquet")
        if f.exists():
            return pl.read_parquet(f).to_pandas()
    raise FileNotFoundError(f"{name} not found under canonical/ or structural/")


def _raw(root: Path, name: str) -> pd.DataFrame:
    import polars as pl
    base = root / "data" / "raw" / "massive" / name
    if base.is_dir():
        return pl.concat(
            [pl.read_parquet(f) for f in sorted(base.glob("*.parquet"))],
            how="diagonal_relaxed",
        ).to_pandas()
    return pl.read_parquet(base.with_suffix(".parquet")).to_pandas()


def _roll_z(s: pd.Series, win: int = 252, minp: int = 60) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - mu) / sd.replace(0, np.nan)


def _third_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7 + 14)


def build(root: str | Path, tickers: list[str] | None = None,
          start: str = "2014-01-01", end: str | None = None,
          audit: bool = False) -> pd.DataFrame:
    """Assemble the full state vector. Returns a pandas DataFrame keyed
    by (ticker, date) with 27 numeric + 2 clock + 7 flag columns.
    audit=True appends actuals_date (PIT provenance) for the leakage
    sidecar — strip it before writing the canonical panel."""
    root = Path(root)
    start_d = pd.Timestamp(start).date()

    # ---------- base frame: trading days × tickers ----------
    daily = _raw(root, "stocks_daily")
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily[daily["date"] >= start_d]
    if tickers:
        daily = daily[daily["ticker"].isin(tickers)]
    if end:
        daily = daily[daily["date"] <= pd.Timestamp(end).date()]
    daily = daily[["ticker", "date", "close", "volume"]].rename(
        columns={"close": "spot", "volume": "day_volume"})

    # ---------- macro (broadcast to all tickers by date) ----------
    def macro(name, col):
        m = _scan(root, name)[["date", col]].copy()
        m["date"] = pd.to_datetime(m["date"]).dt.date
        return m.set_index("date")[col]

    y10 = macro("macro_treasury_yields", "yield_10_year")
    infl_fwd = macro("macro_inflation_expectations", "forward_years_5_to_10")
    effr = macro("macro_funding_conditions", "effective_fed_funds_rate")
    cp90 = macro("macro_funding_conditions",
                 "financial_commercial_paper_90d_rate")
    funding = (cp90 - effr).dropna()

    # ---------- valuation (weekly -> ffill to daily) ----------
    val = _scan(root, "valuation_weekly")
    val["date"] = pd.to_datetime(val["date"]).dt.date
    val = val[["ticker", "date", "cur_mkt_cap", "pe_ratio", "ev_to_ebitda",
               "ev_to_sales", "px_to_book_ratio"]].dropna(subset=["cur_mkt_cap"])

    # ---------- per-ticker assembly ----------
    rep = _scan(root, "report_calendar_us")
    rep["filing_date"] = pd.to_datetime(rep["filing_date"]).dt.date
    rep_map = {t: sorted(g["filing_date"].tolist())
               for t, g in rep.groupby("ticker")}
    # hoisted for the per-ticker loop (avoid re-reading parquets 1258x)
    act_all = _scan(root, "fundamentals_actuals")
    act_all["date"] = pd.to_datetime(act_all["date"]).dt.date
    est_all = pd.concat(
        [_scan(root, "fundamentals_estimates_hist"),
         _scan(root, "fundamentals_estimates_recent")])
    est_all["date"] = pd.to_datetime(est_all["date"]).dt.date
    cal_all = rep

    micro = _scan(root, "microstructure_daily")
    micro["date"] = pd.to_datetime(micro["date"]).dt.date
    smile = _scan(root, "options_smile_daily")
    smile["date"] = pd.to_datetime(smile["date"]).dt.date
    optagg = _scan(root, "options_agg_daily")
    optagg["date"] = pd.to_datetime(optagg["date"]).dt.date
    optagg = optagg[["ticker", "date", "implied_volatility",
                     "open_int_total_call", "open_int_total_put"]]

    # guidance events per ticker
    guid = _raw(root, "guidance_all")
    guid["date"] = pd.to_datetime(guid["date"]).dt.date
    gmap = {t: g.sort_values("date") for t, g in guid.groupby("ticker")}

    frames = []
    for tk, ddf in daily.groupby("ticker"):
        ddf = ddf.sort_values("date").set_index("date")

        # --- valuation block (ffill weekly to daily) ---
        v = val[val["ticker"] == tk].set_index("date").sort_index()
        v = v.reindex(v.index.union(ddf.index)).ffill().reindex(ddf.index)
        gap = pd.concat(
            [_roll_z(np.log(v[c].astype(float))) for c in
             ["pe_ratio", "ev_to_ebitda", "ev_to_sales", "px_to_book_ratio"]],
            axis=1).mean(axis=1)
        ddf["composite_valuation_gap"] = gap
        ddf["valuation_kurtosis"] = gap.rolling(504, min_periods=120).kurt()
        z = gap
        sk = z.rolling(504, min_periods=120).skew()
        ku = z.rolling(504, min_periods=120).kurt()
        ddf["cornish_fisher_gap"] = z + (sk / 6) * (z**2 - 1) + (ku / 24) * (z**3 - 3 * z)
        # mean-reversion speed: AR(1) on gap -> annualized -ln(rho)*252/21
        # rolling corr with lag-1 self = lag-1 autocorrelation, vectorized
        rho = gap.rolling(504, min_periods=120).corr(gap.shift(1))
        ddf["mean_reversion_speed"] = -np.log(rho.clip(1e-4, 0.9999)) * (252 / 21)
        ddf["mcap"] = v["cur_mkt_cap"]

        # --- fundamentals / info dynamics ---
        a = _asof_actuals(act_all, cal_all, tk, ddf.index)
        ddf["cf_free_cash_flow"] = a["fcf"]
        ddf["sales"] = a["sales"]
        ddf["actuals_date"] = a["asof_date"]
        sh = v["cur_mkt_cap"] / ddf["spot"].replace(0, np.nan)
        # BBG fundamentals are $millions; mcap is raw dollars — align units
        fcf_ps = ddf["cf_free_cash_flow"] * 1e6 / sh.replace(0, np.nan)
        ddf["fwd_fcf_fair_value"] = fcf_ps / 0.05   # 5% FCF-yield anchor
        ddf["log_fv_gap"] = np.log(
            ddf["spot"] / ddf["fwd_fcf_fair_value"].replace(0, np.nan))
        ddf["growth_kalman_update"] = _kalman_innovation(ddf["sales"])
        ddf["staleness_quarterly"] = (
            (pd.Series(ddf.index, index=ddf.index)
             - pd.Series(a["asof_date"].values, index=ddf.index))
            .map(lambda x: x.days if pd.notna(x) else np.nan) > 100).astype(int)

        g = gmap.get(tk)
        est = est_all[est_all["ticker"] == tk]
        ddf = _info_dynamics(ddf, g, a["events"], est)

        # --- microstructure (daily stats already aggregated) ---
        m = micro[micro["ticker"] == tk].set_index("date").sort_index()
        for src, dst in [("realized_vol", "minute_realized_diffusion"),
                         ("amihud_illiq", "amihud_illiq"),
                         ("kyle_lambda", "kyle_lambda"),
                         ("fdt_deviation", "fdt_deviation")]:
            ddf[dst] = m[src]
        ddf["liquidity_roc"] = (
            m["dollar_vol"] / m["dollar_vol"].rolling(20).mean() - 1)

        # --- options ---
        s = smile[smile["ticker"] == tk].set_index("date").sort_index()
        ddf["atm_iv"] = s["atm_iv_30d"]
        ddf["skew_25d"] = s["skew_25d"]
        ddf["rn_kurtosis"] = s["rn_kurtosis"]
        ddf["term_slope"] = s["term_slope"]
        ddf["variance_risk_premium"] = s["atm_iv_30d"] ** 2 - m["realized_vol"] ** 2
        ddf["options_thin_chain"] = (s["n_contracts"].fillna(0) < 10).astype(int)
        oa = optagg[optagg["ticker"] == tk].set_index("date").sort_index()
        d_oi = (oa["open_int_total_put"] - oa["open_int_total_call"]).diff()
        d_px = ddf["spot"].pct_change()
        ddf["oi_divergence"] = _roll_z(d_oi / (d_oi.rolling(20).std() + 1e-9)
                                     - np.sign(d_px), 252)
        # calibration error vs BBG aggregate IV -> quantile flag
        calib = (s["atm_iv_30d"] - oa["implied_volatility"]).abs()
        ddf["regime_calib_error_quantile"] = calib.rank(pct=True)

        # --- macro ---
        ddf["funding_stress"] = funding.reindex(ddf.index).ffill()
        ddf["inflation_expectation"] = infl_fwd.reindex(ddf.index).ffill()
        y10d = y10.reindex(ddf.index).ffill()
        ddf["treasury_funding_interact"] = y10d * ddf["funding_stress"]
        mcap_z = _roll_z(np.log(ddf["mcap"].astype(float)))
        ddf["mktcap_duration_interact"] = mcap_z * y10d.diff()

        # --- clocks ---
        # days_to_next_report is PIT: expected next filing projected from
        # filings OBSERVED so far (last filing + expanding median gap) —
        # using the realized future calendar would leak whether/when a
        # filing actually happened.
        rdates = rep_map.get(tk, [])
        ddf["days_to_next_report"] = _expected_next_report(
            list(ddf.index), rdates)
        opex = [_third_friday(y, m)
                for y in range(start_d.year, start_d.year + 30)
                for m in range(1, 13)]
        ddf["days_to_opex"] = [_days_to(d, opex) for d in ddf.index]

        frames.append(ddf)

    out = pd.concat(frames)
    # DEVIATION: spread-z uses the high-low range proxy for all rows —
    # NBBO quotes are not available at this scale
    out["spread_filtered"] = 1
    out.index.name = "date"
    out = out.reset_index().rename(columns={"index": "date"})
    cols = ["ticker", "date", "spot"] + NUMERIC + CLOCK + FLAGS
    if audit:
        # leakage-gate sidecar columns — asof dates prove each row's
        # fundamentals were knowable on the row date (G-50)
        return out[cols + ["actuals_date"]]
    return out[cols]


def _days_to(d: date, sched: list[date]) -> int | float:
    import bisect
    i = bisect.bisect_left(sched, d)
    return (sched[i] - d).days if i < len(sched) else np.nan


def _expected_next_report(dates: list[date], filings: list[date]):
    """PIT-safe next-report clock: for each day d, expected next filing
    = last filing <= d + expanding median inter-filing gap (default
    91d). Never touches filings after d."""
    import bisect
    out = np.full(len(dates), np.nan)
    gaps: list[int] = []
    for i, d in enumerate(dates):
        k = bisect.bisect_right(filings, d)
        if k == 0:
            continue
        if k >= 3:
            gaps = [ (filings[j] - filings[j-1]).days
                     for j in range(1, k) ]
        gap = float(np.median(gaps)) if gaps else 91.0
        out[i] = (filings[k - 1] + timedelta(days=int(round(gap))) - d).days
    return pd.Series(out, index=dates)


def _asof_actuals(act_all: pd.DataFrame, cal_all: pd.DataFrame,
                  ticker: str, dates) -> dict:
    """Latest actuals row knowable at each date — PIT via filing dates.

    actuals.date is period-END (BBG convention). A quarter's numbers are
    knowable only once filed: map each period_end to the first filing in
    (end+5d, end+95d] from report_calendar_us; fallback end+45d.
    """
    act = act_all[act_all["ticker"] == ticker].copy()
    if act.empty:
        return {"fcf": pd.Series(np.nan, index=dates),
                "sales": pd.Series(np.nan, index=dates),
                "asof_date": pd.Series(pd.NaT, index=dates),
                "events": pd.DataFrame(columns=["asof", "sales_rev_turn"])}
    cal = cal_all[cal_all["ticker"] == ticker]
    fd = sorted(cal["filing_date"].tolist())
    import bisect

    def knowable(period_end: date) -> date:
        # first filing strictly after period_end + 4d, within 95d
        i = bisect.bisect_right(fd, period_end + timedelta(days=4))
        if i < len(fd) and (fd[i] - period_end).days <= 95:
            return fd[i]
        return period_end + timedelta(days=45)

    act["asof"] = [knowable(d) for d in act["date"]]
    act = act.sort_values("asof")
    idx = np.searchsorted(
        np.array([d.toordinal() for d in act["asof"]]),
        np.array([d.toordinal() for d in dates]), side="right") - 1
    idx = np.clip(idx, -1, len(act) - 1)

    def col(name):
        vals = act[name].to_numpy()
        return pd.Series(
            np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan),
            index=dates)
    return {
        "fcf": col("cf_free_cash_flow"),
        "sales": col("sales_rev_turn"),
        "asof_date": pd.Series(
            np.where(idx >= 0,
                     act["asof"].to_numpy()[np.clip(idx, 0, None)], pd.NaT),
            index=dates),
        # event table for PIT-safe TTM aggregation in _info_dynamics
        "events": act[["date", "asof", "sales_rev_turn"]]
        .reset_index(drop=True),
    }


def _kalman_innovation(sales: pd.Series) -> pd.Series:
    """1-D Kalman on log-sales: output = standardized innovation."""
    x = np.log(sales.replace(0, np.nan)).to_numpy()
    out = np.full(len(x), np.nan)
    level, var, q, r = 0.0, 1.0, 1e-4, 1e-2
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        if np.isnan(prev):
            level, prev = v, v
            continue
        var += q
        gain = var / (var + r)
        innov = v - level
        out[i] = innov / math.sqrt(var + r)
        level += gain * innov
        var *= 1 - gain
        prev = v
    return pd.Series(out, index=sales.index)


def _snapshot_events(events: pd.DataFrame, est: pd.DataFrame | None):
    """Snapshot-track surprise events: (actual − estimate)/|estimate|
    per actuals period, revealed at the period's filing asof date.
    Returns sorted (dates_ns, surprise, kl_bits) arrays."""
    empty = (np.array([], dtype="datetime64[ns]"),
             np.array([], dtype=float), np.array([], dtype=float))
    if events.empty or est is None or est.empty or "date" not in events:
        return empty
    e = est[["date", "sales_rev_turn"]].dropna().rename(
        columns={"sales_rev_turn": "est_sales"})
    m = events.merge(e, on="date", how="inner").dropna(
        subset=["est_sales", "sales_rev_turn", "asof"])
    if m.empty:
        return empty
    m = m.sort_values("asof")
    act = m["sales_rev_turn"].to_numpy(float)
    est_v = m["est_sales"].to_numpy(float)
    ok = np.abs(est_v) > 0
    surp = np.where(ok, (act - est_v) / np.abs(est_v), np.nan)
    # dispersion unknown for a single snapshot — assume sig = 10% of |est|
    sig = np.maximum(np.abs(est_v) * 0.10, 1e-9)
    kl = np.where(ok, 0.5 * (act - est_v) ** 2 / (sig * sig) / math.log(2),
                  np.nan)
    return (pd.to_datetime(m["asof"]).to_numpy(), surp, kl)


def _info_dynamics(ddf: pd.DataFrame, g: pd.DataFrame | None,
                   events: pd.DataFrame,
                   est: pd.DataFrame | None = None) -> pd.DataFrame:
    """Guidance-track surprise/KL/half-life/confidence/velocity with
    snapshot fallback. `events` = actuals rows with period_end + PIT
    asof dates (TTM aggregation); `est` = snapshot estimates table
    (surprise fallback before the first guidance event — DEVIATION 1
    vintage caveat, flagged by snapshot_track_used)."""
    idx = ddf.index
    ddf["guidance_absent"] = 1   # cleared below when FY guidance exists
    ddf["snapshot_track_used"] = 1
    ddf["half_life_imputed"] = 1
    for c in ("fundamental_surprise", "kl_surprise_bits",
              "measured_half_life", "fundamental_confidence",
              "guidance_range_velocity"):
        ddf[c] = np.nan

    # snapshot-track surprise events: (actual − estimate)/|estimate|
    # per period, revealed at that period's filing asof date
    snap_dates, snap_surp, snap_kl = _snapshot_events(events, est)
    snap_pos = np.searchsorted(snap_dates, pd.to_datetime(
        pd.Series(idx)).to_numpy(), side="right") - 1
    snap_has = snap_pos >= 0

    if g is None or g.empty:
        ddf["measured_half_life"] = 21.0
        ddf["fundamental_confidence"] = 0.25
        if len(snap_surp):
            ddf["fundamental_surprise"] = pd.Series(
                np.where(snap_has, snap_surp[np.clip(snap_pos, 0, None)],
                         np.nan), index=idx)
            ddf["kl_surprise_bits"] = pd.Series(
                np.where(snap_has, snap_kl[np.clip(snap_pos, 0, None)],
                         np.nan), index=idx)
        return ddf

    g = g.copy()
    # period-aware scale: FY guides compare to TTM actuals, quarterly
    # guides to the latest knowable quarter. Mixing periods in mid is
    # fine for confidence/velocity (scale-free ratios), and surprise
    # picks the matching actual per event.
    if not g.empty:
        ddf["guidance_absent"] = 0
    # Benzinga revenue guidance is raw dollars -> $millions to match BBG
    # actuals. EPS guidance (per-share $) is a different scale — excluded
    # from mid/width rather than mixed units.
    lo = g["min_revenue_guidance"] / 1e6
    hi = g["max_revenue_guidance"] / 1e6
    mid = (lo + hi) / 2
    width = (hi - lo).abs()
    g = g.assign(mid=mid, width=width)
    g = g[g["mid"].notna() & (g["mid"] != 0)].sort_values("date")
    if g.empty:
        ddf["measured_half_life"] = 21.0
        ddf["fundamental_confidence"] = 0.25
        if len(snap_surp):
            ddf["fundamental_surprise"] = pd.Series(
                np.where(snap_has, snap_surp[np.clip(snap_pos, 0, None)],
                         np.nan), index=idx)
            ddf["kl_surprise_bits"] = pd.Series(
                np.where(snap_has, snap_kl[np.clip(snap_pos, 0, None)],
                         np.nan), index=idx)
        return ddf

    ev_dates = pd.to_datetime(g["date"]).to_numpy()
    pos = np.searchsorted(ev_dates, pd.to_datetime(pd.Series(idx)).to_numpy(),
                          side="right") - 1
    has = pos >= 0
    ddf["snapshot_track_used"] = (~has).astype(int)

    mid_v = g["mid"].to_numpy()
    wid_v = g["width"].to_numpy()
    per_v = g["fiscal_period"].astype(str).str.upper().to_numpy()
    ddf.loc[has, "fundamental_confidence"] = [
        1.0 / (1.0 + w / (abs(m) + 1e-9))
        for m, w in zip(mid_v[pos[has]], wid_v[pos[has]])]
    # range velocity: Δmid per day vs the most recent SAME-period
    # event (period transitions would inject phantom ~4x moves)
    vel_full = np.full(len(g), np.nan)
    last_mid: dict[str, float] = {}
    last_day: dict[str, np.datetime64] = {}
    for j in range(len(g)):
        pj = per_v[j]
        if pj in last_mid:
            dd = max(float((ev_dates[j] - last_day[pj])
                           / np.timedelta64(1, "D")), 1.0)
            vel_full[j] = (mid_v[j] - last_mid[pj]) / dd
        last_mid[pj] = mid_v[j]
        last_day[pj] = ev_dates[j]
    ddf.loc[has, "guidance_range_velocity"] = vel_full[pos[has]]

    # surprise: guided mid vs actuals knowable at the event date —
    # FY events compare to TTM sales, quarterly events to the latest
    # reported quarter (both express guide-vs-run-rate gap).
    ev_asof = pd.to_datetime(events["asof"]).to_numpy() \
        if not events.empty else np.array([], dtype="datetime64[ns]")
    ev_sales = events["sales_rev_turn"].to_numpy() \
        if not events.empty else np.array([], dtype=float)
    # compute once per guidance EVENT (the info set at issuance), then
    # broadcast to every day that event is the latest
    val_surp = np.full(len(g), np.nan)
    val_kl = np.full(len(g), np.nan)
    for j in range(len(g)):
        m_, w_ = mid_v[j], wid_v[j]
        if not np.isfinite(m_) or m_ == 0:
            continue
        k = np.searchsorted(ev_asof, ev_dates[j], side="right")
        if k == 0:
            continue
        actual = (np.nansum(ev_sales[max(0, k - 4):k])
                  if per_v[j] == "FY" else ev_sales[k - 1])
        if not np.isfinite(actual) or actual <= 0:
            continue
        val_surp[j] = (actual - m_) / abs(m_)
        sig = max(w_ / 4.0, abs(m_) * 0.01, 1e-9)
        val_kl[j] = 0.5 * (actual - m_) ** 2 / (sig * sig) / math.log(2)
    surp = np.where(has, val_surp[np.clip(pos, 0, None)], np.nan)
    kl = np.where(has, val_kl[np.clip(pos, 0, None)], np.nan)
    # snapshot fallback fills the rows with no guidance event yet
    no_guid = ~has
    fill_pos = snap_pos[no_guid]
    fill_ok = fill_pos >= 0
    if len(snap_surp) and fill_ok.any():
        surp_vals = np.full(int(no_guid.sum()), np.nan)
        surp_vals[fill_ok] = snap_surp[fill_pos[fill_ok]]
        surp[no_guid] = surp_vals
        kl_vals = np.full(int(no_guid.sum()), np.nan)
        kl_vals[fill_ok] = snap_kl[fill_pos[fill_ok]]
        kl[no_guid] = kl_vals
    ddf["fundamental_surprise"] = pd.Series(surp, index=idx).ffill()
    ddf["kl_surprise_bits"] = pd.Series(kl, index=idx).ffill()

    # half-life: AR(1) on |surprise| at event granularity
    es = np.abs(surp[~np.isnan(surp)])
    if len(es) >= 8:
        x, y = es[:-1], es[1:]
        if x.std() > 0:
            rho = np.corrcoef(x, y)[0, 1]
            if 0 < rho < 1:
                ddf["measured_half_life"] = math.log(2) / -math.log(rho)
                ddf["half_life_imputed"] = 0
            else:
                ddf["measured_half_life"] = 21.0
        else:
            ddf["measured_half_life"] = 21.0
    else:
        ddf["measured_half_life"] = 21.0
    ddf["fundamental_confidence"] = ddf["fundamental_confidence"].fillna(0.25)
    return ddf
