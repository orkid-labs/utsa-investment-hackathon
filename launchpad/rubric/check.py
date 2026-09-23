#!/usr/bin/env python3
"""Deterministic rubric checker for hackathon submissions.

Usage:
    SV_DATA_ROOT=/path/to/state-vector \
    python check.py --base-url http://localhost:8000 [--rubric rubric.yaml]

Hits a running submission, validates response shapes, recomputes the
reference backtest from the dataset, and emits a score JSON on stdout.
Deterministic: same app + same data => same score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

RUBRIC_PATH = Path(__file__).parent / "rubric.yaml"
TRADING_DAYS = 252


def http(base: str, method: str, path: str, body=None, timeout: int = 30):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def is_option(t: str) -> bool:
    return t.startswith("O:")


def leg_returns(ds, tickers: list[str], start: date, end: date,
                dividends: bool = False):
    """Wide return frame for stock + option legs (0.0 for missing prints).

    dividends=True: stock legs get total-return treatment —
    r_t = (close_t + cash_div_t) / close_{t-1} - 1 on ex-dates.
    """
    import polars as pl

    opts = [t for t in tickers if is_option(t)]
    stocks = [t for t in tickers if not is_option(t)]
    scans = []
    if stocks:
        scans.append(
            ds._scan("stocks_daily").filter(pl.col("ticker").is_in(stocks))
        )
    if opts:
        scans.append(
            ds._scan("options_daily").filter(pl.col("ticker").is_in(opts))
        )
    wide = (
        pl.concat(scans, how="diagonal_relaxed")
        .filter(pl.col("date").is_between(start, end))
        .select(["date", "ticker", "close"])
        .collect()
        .pivot(on="ticker", index="date", values="close")
        .sort("date")
    )
    cols = [c for c in wide.columns if c != "date"]
    exprs = [pl.col(c) / pl.col(c).shift(1) - 1.0 for c in cols]
    rets = wide.select(["date", *exprs]).fill_null(0.0)

    if dividends:
        divs = _dividends(ds, stocks, start, end)
        if divs is not None:
            long_px = wide.select(["date", *cols]).unpivot(
                index="date", variable_name="ticker", value_name="px")
            long_px = long_px.with_columns(
                pl.col("px").shift(1).over("ticker").alias("prev_px")
            ).join(divs, on=["ticker", "date"], how="left").with_columns(
                (pl.col("div").fill_null(0.0) / pl.col("prev_px")).alias("dy")
            )
            dy_wide = long_px.select(["date", "ticker", "dy"]).pivot(
                on="ticker", index="date", values="dy")
            rets = rets.join(dy_wide, on="date", how="left", suffix="_dy")
            for c in cols:
                if f"{c}_dy" in rets.columns:
                    rets = rets.with_columns(
                        (pl.col(c) + pl.col(f"{c}_dy").fill_null(0.0)).alias(c)
                    ).drop(f"{c}_dy")
    return (rets.drop("date") if "date" in rets.columns else rets), cols


def _dividends(ds, stocks, start, end):
    """ticker/date/cash_amount dividend table, or None if panel absent."""
    import polars as pl

    p = next(
        (x for x in (
            ds.root / "data/raw/massive/dividends.parquet",
            ds.root / "data/raw/massive/dividends_all.parquet",
        ) if x.exists()),
        None,
    )
    if p is None:
        return None
    df = pl.read_parquet(p)
    dcol = next((c for c in df.columns if "ex_dividend" in c or c == "ex_date"), None)
    acol = next((c for c in df.columns if c in ("cash_amount", "amount")), None)
    if not dcol or not acol:
        return None
    return (
        df.filter(pl.col("ticker").is_in(stocks))
        .with_columns(pl.col(dcol).cast(pl.Date).alias("date"))
        .filter(pl.col("date").is_between(start, end))
        .group_by(["ticker", "date"])
        .agg(pl.col(acol).sum().alias("div"))
    )


def portfolio_metrics(rets, cols, wmap) -> dict:
    port = sum(rets[c] * wmap.get(c, 0.0) for c in cols)
    n = port.len()
    if n == 0:
        return {}
    total = float((1 + port).product() - 1)
    mean = float(port.mean())
    std = float(port.std() or 0.0)
    ann_ret = (1 + total) ** (TRADING_DAYS / n) - 1
    ann_vol = std * TRADING_DAYS**0.5
    sharpe = (mean / std * TRADING_DAYS**0.5) if std else 0.0
    curve = (1 + port).cum_prod()
    max_dd = abs(float((curve / curve.cum_max() - 1).min()))
    return {
        "n_days": int(n),
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def reference_backtest(ds, tickers, weights, start, end,
                       dividends: bool = False) -> dict:
    rets, cols = leg_returns(ds, tickers, start, end, dividends=dividends)
    wmap = dict(zip(tickers, weights))
    return portfolio_metrics(rets, cols, wmap)


def get_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def fail(check, note, earned=0.0):
    return {"id": check["id"], "earned": earned, "max": check["points"],
            "note": note}


def score_check(check, base: str, ds, universe: set, helpers: dict) -> dict:
    pts = check["points"]
    req = check.get("request")
    exp = check.get("expect", {})

    if req is None:  # umbrella checks (e.g., responsiveness)
        return {"id": check["id"], "earned": pts, "max": pts, "note": "n/a"}

    # latency wrapper — measures median response time when requested
    lat_ms = exp.get("latency_ms")
    reps = 5 if lat_ms else 1
    status, body, worst = None, None, 0.0
    try:
        for _ in range(reps):
            t0 = time.monotonic()
            status, body = http(base, req["method"], req["path"],
                                req.get("body"), timeout=30)
            worst = max(worst, (time.monotonic() - t0) * 1000)
    except Exception as e:
        return fail(check, f"request failed: {e}")

    if lat_ms:
        helpers["latency_ms"].append(worst)

    if exp.get("status") and status != exp["status"]:
        return fail(check, f"status {status} != {exp['status']}")

    if "json_contains" in exp:
        for k, v in exp["json_contains"].items():
            if get_path(body, k) != v:
                return fail(check, f"{k}={get_path(body, k)!r} expected {v!r}")

    target = body
    if "json_path" in exp:
        target = get_path(body, exp["json_path"])

    if exp.get("is_list") and not isinstance(target, list):
        return fail(check, "not a list")

    if "item_has_keys" in exp and isinstance(target, list) and target:
        missing = [k for k in exp["item_has_keys"] if k not in target[0]]
        if missing:
            return fail(check, f"items missing keys {missing}")

    if "has_keys" in exp:
        missing = [k for k in exp["has_keys"] if k not in body]
        if missing:
            return fail(check, f"missing keys {missing}")

    if isinstance(target, list) and any(
        k in exp for k in ("weights_sum_to", "long_only", "tickers_in_universe")
    ):
        ws = [h.get("weight") for h in target if isinstance(h, dict)]
        if not ws or any(not isinstance(w, (int, float)) for w in ws):
            return fail(check, "weights missing/non-numeric")
        if "weights_sum_to" in exp:
            total = sum(ws)
            if abs(total - exp["weights_sum_to"]) > exp.get("weights_tol", 0.01):
                return fail(check, f"weights sum {total:.4f}")
        if exp.get("long_only") and any(w < 0 for w in ws):
            return fail(check, "short positions present (long-only mandate)")
        if exp.get("tickers_in_universe"):
            bad = []
            for h in target:
                t = h.get("ticker", "")
                if is_option(t):
                    from statevector import parse_occ
                    try:
                        t = parse_occ(t)["underlying"]
                    except ValueError:
                        bad.append(t)
                        continue
                if t not in universe:
                    bad.append(t)
            if bad:
                return fail(check, f"tickers not in universe: {bad[:5]}")

    if exp.get("reference_backtest") or exp.get("reference_total_return"):
        b = req["body"]
        weights = b.get("weights") or [1 / len(b["tickers"])] * len(b["tickers"])
        ref = reference_backtest(
            ds, b["tickers"], weights,
            date.fromisoformat(b["start"]), date.fromisoformat(b["end"]),
            dividends=bool(b.get("adjust_dividends")),
        )
        if not ref:
            return fail(check, "reference produced no data — check request dates")
        # vacuous-pass guard: if the request has O: legs, the reference must
        # have actually priced them (else both sides trivially agree)
        o_legs = [t for t in b["tickers"] if is_option(t)]
        if o_legs:
            rets_check, cols_check = leg_returns(
                ds, b["tickers"],
                date.fromisoformat(b["start"]), date.fromisoformat(b["end"]))
            missing = [t for t in o_legs
                       if t not in cols_check or rets_check[t].sum() == 0]
            if missing:
                return fail(check,
                            f"reference contract(s) absent in data: {missing}")
        # vacuous-pass guard for dividends: if requested, the dividends
        # panel must exist AND cover the window
        if exp.get("reference_total_return") and b.get("adjust_dividends"):
            stocks = [t for t in b["tickers"] if not is_option(t)]
            divs = _dividends(
                ds, stocks,
                date.fromisoformat(b["start"]), date.fromisoformat(b["end"]))
            if divs is None or divs.height == 0:
                return fail(check, "reference dividends panel missing/empty "
                                   "— cannot verify total-return math")
        tol = exp.get("tolerance", 0.02)
        diffs = {}
        for k, rv in ref.items():
            cv = body.get(k)
            if not isinstance(cv, (int, float)):
                diffs[k] = f"missing/non-numeric (ref {rv:.4f})"
            elif abs(cv - rv) / max(abs(rv), 1e-9) > tol:
                diffs[k] = f"got {cv:.4f} vs ref {rv:.4f}"
        if diffs:
            frac = max(0.0, 1 - len(diffs) / len(ref))
            return fail(check, f"value mismatches: {diffs}",
                        earned=round(pts * frac, 1))

    if exp.get("sharpe_floor"):
        # Their reported metrics must be self-consistent AND beat equal-weight
        b = req["body"]
        weights = b.get("weights") or [1 / len(b["tickers"])] * len(b["tickers"])
        rets, cols = leg_returns(
            ds, b["tickers"],
            date.fromisoformat(b["start"]), date.fromisoformat(b["end"]),
        )
        theirs = portfolio_metrics(rets, cols,
                                   dict(zip(b["tickers"], weights)))
        ew = portfolio_metrics(
            rets, cols, {c: 1 / len(cols) for c in cols})
        s_rep = body.get("sharpe")
        if not isinstance(s_rep, (int, float)):
            return fail(check, "sharpe missing/non-numeric")
        if theirs and abs(s_rep - theirs["sharpe"]) > 0.05:
            return fail(check,
                        f"reported sharpe {s_rep} inconsistent with "
                        f"weights (recomputed {theirs['sharpe']:.4f})")
        if ew["sharpe"] and s_rep < ew["sharpe"] * exp["sharpe_floor"]:
            return fail(check,
                        f"sharpe {s_rep} < {exp['sharpe_floor']}x equal-weight "
                        f"({ew['sharpe']:.4f})")

    if "all_items_field_lte" in exp and isinstance(target, list):
        spec = exp["all_items_field_lte"]
        bad = [r.get(spec["field"]) for r in target
               if isinstance(r, dict) and str(r.get(spec["field"], "")) > spec["value"]]
        if bad:
            return fail(check, f"lookahead rows after {spec['value']}: {bad[:3]}")

    return {"id": check["id"], "earned": pts, "max": pts, "note": "ok"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--rubric", default=str(RUBRIC_PATH))
    args = ap.parse_args()

    import yaml
    rubric = yaml.safe_load(Path(args.rubric).read_text())

    from statevector import Dataset
    ds = Dataset()
    helpers = {
        "latency_ms": [],
    }
    universe = set(ds.universe())

    results = [score_check(c, args.base_url, ds, universe, helpers)
               for c in rubric["checks"]]

    # latency gate post-processing: replace placeholder notes
    for r, c in zip(results, rubric["checks"]):
        lat = c.get("expect", {}).get("latency_ms")
        if lat and helpers["latency_ms"]:
            med = sorted(helpers["latency_ms"])[len(helpers["latency_ms"]) // 2]
            if med > lat:
                r["earned"] = 0
                r["note"] = f"median latency {med:.0f}ms > {lat}ms"
            else:
                r["note"] = f"median latency {med:.0f}ms"

    out = {
        "base_url": args.base_url,
        "rubric": Path(args.rubric).name,
        "score": round(sum(r["earned"] for r in results), 1),
        "max": sum(r["max"] for r in results),
        "checks": results,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
