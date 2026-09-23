"""Validation: rank-IC significance + embargoed walk-forward folds,
plus additive vs route-energy formulation selection.

Additive:     score = Σ w_i · z_i
Route-energy: score = sqrt(Σ (w_i · z_i)²)  signed by Σ w_i·z_i —
              pays on joint deviation rather than linear mix.
Selected by which produces higher mean rank-IC over the folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .vector import NUMERIC


def forward_returns(daily: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """(ticker, date, fwd) — forward simple return over `horizon`
    trading days, merged downstream on keys (never positional)."""
    d = daily.sort_values(["ticker", "date"]).copy()
    d["fwd"] = d.groupby("ticker")["spot"].shift(-horizon) / d["spot"] - 1
    return d[["ticker", "date", "fwd"]]


def rank_ic_by_day(df: pd.DataFrame,
                   feat_cols: list[str],
                   score_col: str = "score") -> pd.Series:
    """Per-date cross-sectional Spearman IC of a score column vs `fwd`.
    df must have ticker,date,fwd + feat_cols (+ score_col)."""
    z = df[feat_cols].apply(
        lambda c: (c - c.mean()) / (c.std() or np.nan))
    df = df.assign(**{score_col: z.mean(axis=1)})
    ics = {}
    for d, g in df.groupby("date"):
        g = g.dropna(subset=[score_col, "fwd"])
        if len(g) < 30:
            continue
        ic = spearmanr(g[score_col], g["fwd"]).statistic
        if np.isfinite(ic):
            ics[d] = ic
    return pd.Series(ics).sort_index()


def embargoed_folds(dates: list, n_folds: int = 5,
                    embargo_days: int = 21) -> list[tuple[list, list]]:
    """Walk-forward (train < test) folds with an embargo gap."""
    ds = sorted(pd.to_datetime(dates).unique())
    fold_size = max(len(ds) // (n_folds + 2), 21)
    folds = []
    for k in range(n_folds):
        test_end = len(ds) - k * fold_size
        test_start = test_end - fold_size
        if test_start < fold_size:
            break
        embargo = pd.Timedelta(days=embargo_days)
        train = [d for d in ds[:test_start]
                 if d < ds[test_start] - embargo]
        test = list(ds[test_start:test_end])
        folds.append((train, test))
    return folds


def dual_formulation_scores(z: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(additive, route_energy) per row given standardized features."""
    lin = z @ w
    route = np.sign(lin) * np.sqrt(((z * w) ** 2).sum(axis=1))
    return lin, route


def ic_significance(ics: pd.Series) -> dict:
    """t-stat + two-sided p for a per-day IC series."""
    from scipy.stats import t as tdist
    n = len(ics)
    if n < 3:
        return {"n": n, "mean": float(ics.mean()) if n else np.nan,
                "t": np.nan, "p": np.nan}
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": float(ics.mean()),
            "ic_ir": float(ics.mean() / ics.std(ddof=1)),
            "t": float(t), "p": float(2 * tdist.sf(abs(t), n - 1))}


def select_formulation(vec: pd.DataFrame, daily: pd.DataFrame,
                       feat_cols: list[str] | None = None,
                       horizon: int = 21, n_folds: int = 5,
                       ridge: float = 1e-3) -> dict:
    """Compare additive vs route-energy over embargoed walk-forward
    folds. Per fold: standardize on TRAIN stats, ridge-fit weights on
    TRAIN (z -> fwd), score TEST with both formulations. Pass a vec
    that already excludes the sealed holdout."""
    feat_cols = feat_cols or [c for c in vec.columns
                              if c.startswith("pc")] or NUMERIC
    df = vec[["ticker", "date"] + feat_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    fwd = forward_returns(daily, horizon)
    fwd["date"] = pd.to_datetime(fwd["date"])
    df = df.merge(fwd, on=["ticker", "date"], how="inner")
    df = df.dropna(subset=["fwd"])
    df = df.dropna(subset=feat_cols, how="all")

    dates = np.sort(df["date"].unique())
    folds = embargoed_folds(list(dates), n_folds)
    ics = {"additive": [], "route_energy": []}
    ic_dates = {"additive": [], "route_energy": []}
    for train, test in folds:
        tr = df[df["date"].isin(train)]
        te = df[df["date"].isin(test)]
        if len(tr) < 500 or te.empty:
            continue
        mu = tr[feat_cols].mean()
        sd = tr[feat_cols].std().replace(0, np.nan)
        ztr = ((tr[feat_cols] - mu) / sd).fillna(0.0).to_numpy(float)
        ytr = tr["fwd"].to_numpy(float)
        w = np.linalg.solve(
            ztr.T @ ztr + ridge * np.eye(len(feat_cols)),
            ztr.T @ ytr)
        nrm = np.linalg.norm(w)
        w = w / nrm if nrm > 0 else np.ones(len(feat_cols)) / len(feat_cols)
        for d, g in te.groupby("date"):
            if len(g) < 30:
                continue
            z = ((g[feat_cols] - mu) / sd).fillna(0.0).to_numpy(float)
            a, r = dual_formulation_scores(z, w)
            for key, sc in (("additive", a), ("route_energy", r)):
                ic = spearmanr(sc, g["fwd"]).statistic
                if np.isfinite(ic):
                    ics[key].append(ic)
                    ic_dates[key].append(d)
    a_ser = pd.Series(ics["additive"], index=ic_dates["additive"])
    r_ser = pd.Series(ics["route_energy"], index=ic_dates["route_energy"])
    a_sig, r_sig = ic_significance(a_ser), ic_significance(r_ser)
    out = {
        "folds": len(folds),
        "horizon_days": horizon,
        "features": feat_cols,
        "additive_ic_mean": a_sig["mean"],
        "additive_ic_ir": a_sig.get("ic_ir", np.nan),
        "additive_ic_t": a_sig["t"],
        "additive_ic_p": a_sig["p"],
        "route_ic_mean": r_sig["mean"],
        "route_ic_ir": r_sig.get("ic_ir", np.nan),
        "route_ic_t": r_sig["t"],
        "route_ic_p": r_sig["p"],
    }
    out["selected"] = (
        "route_energy"
        if (np.isfinite(out["route_ic_ir"]) and np.isfinite(out["additive_ic_ir"])
            and out["route_ic_ir"] > out["additive_ic_ir"])
        else "additive")
    return out
