"""Shared reference backtest engine.

Used by the launchpad template app AND the rubric checker, so reported
metrics and recomputed metrics come from the same code path — the
reference is authoritative by construction.

Fee model — proportional transaction cost on turnover:
  * every dollar traded pays cost_bps/1e4. Buys and sells both pay.
    Turnover on a trade day is sum_i |w_target_i - w_held_i|.
  * stocks pay STOCK_COST_BPS, option legs (O:) pay OPTION_COST_BPS.
  * entering the book at t=0 is a trade: cost on sum_i w_i.
  * liquidating at the end is a trade: cost on the final weights.
  * a rebalance resets weights to targets at that day's close; the
    day's return is earned on the pre-rebalance (drifted) weights and
    the fee is deducted from the same day's return.

rebalance semantics:
  "none"    — buy-and-hold; weights drift with returns; entry+exit only
  "daily"   — reset to targets every day (classic constant-mix)
  "weekly"  — reset on the first trading day of each ISO week
  "monthly" — reset on the first trading day of each month
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252
STOCK_COST_BPS = 10.0
OPTION_COST_BPS = 50.0
REBALANCE_CHOICES = ("none", "daily", "weekly", "monthly")

# Pseudo-ticker for the cash sleeve: zero return, zero transaction cost.
# Weights still sum to ~1.0 — CASHHOLDING fills the residual inside the book.
CASH_TICKER = "CASHHOLDING"


def apply_split_factors(rets, dates, splits):
    """Correct split-day returns inside a wide return frame.

    rets:   polars frame — one column per priced leg.
    dates:  trading-day labels aligned to rets rows.
    splits: polars frame (ticker, date, factor); factor = to/from ratio
            (10-for-1 -> 10.0). On an ex-date the raw close ratio embeds
            a phantom ~1/factor move; the true return is
            (1 + r_raw) * factor - 1.

    The stocks_daily `close` is raw/unadjusted — judged paths apply the
    documented corporate_actions events so an in-window split cannot
    corrupt a scored backtest.
    """
    import polars as pl
    if splits is None or getattr(splits, "height", 0) == 0 \
            or rets.height == 0:
        return rets
    dmap = {d: i for i, d in enumerate(dates)}
    cols = list(rets.columns)
    R = rets.to_numpy().copy()
    for sp in splits.iter_rows(named=True):
        i = dmap.get(sp["date"])
        if i is None or sp["ticker"] not in cols:
            continue
        j = cols.index(sp["ticker"])
        if np.isfinite(R[i, j]):
            R[i, j] = (R[i, j] + 1.0) * sp["factor"] - 1.0
    return pl.DataFrame(R, schema=cols)


def leg_bps(ticker: str) -> float:
    if ticker == CASH_TICKER:
        return 0.0
    return OPTION_COST_BPS if ticker.startswith("O:") else STOCK_COST_BPS


def _rebalance_mask(dates: list, freq: str) -> np.ndarray:
    """Boolean mask over trading days marking rebalance days (t=0 excluded —
    the entry trade is charged separately)."""
    n = len(dates)
    mask = np.zeros(n, dtype=bool)
    if freq == "daily":
        mask[1:] = True
    elif freq in ("weekly", "monthly"):
        prev = (dates[0].isocalendar()[0], dates[0].isocalendar()[1]) \
            if freq == "weekly" else (dates[0].year, dates[0].month)
        for t in range(1, n):
            key = (dates[t].isocalendar()[0], dates[t].isocalendar()[1]) \
                if freq == "weekly" else (dates[t].year, dates[t].month)
            if key != prev:
                mask[t] = True
            prev = key
    return mask


def run_backtest(rets, dates: list, tickers: list[str], weights: list[float],
                 *, rebalance: str = "none",
                 cost_bps: float | None = None,
                 exit_cost: bool = True) -> dict:
    """Run the reference backtest on a polars return frame.

    rets:   polars DataFrame, one row per trading day, one col per priced
            leg (already shifted to returns, nulls filled 0).
    dates:  trading-day labels aligned to rets rows (polars Date/datetime).
    tickers/weights: target weights (may include unpriced legs — they
            simply never trade and never earn).

    Returns the judged metric keys plus cost disclosure fields.
    """
    if CASH_TICKER in tickers and CASH_TICKER not in rets.columns:
        import polars as pl
        rets = rets.with_columns(pl.lit(0.0).alias(CASH_TICKER))
    cols = [t for t in tickers if t in rets.columns]
    if rets.height == 0 or not cols:
        return {}

    R = np.nan_to_num(rets.select(cols).to_numpy(), nan=0.0)
    n = R.shape[0]
    w_target = np.array([weights[tickers.index(c)] for c in cols])
    bps = np.array([
        (cost_bps if cost_bps is not None else leg_bps(c)) / 1e4
        for c in cols
    ])
    rebal = _rebalance_mask(dates, rebalance)

    w = w_target.copy()
    port = np.empty(n)
    turnover = np.zeros(n)
    for t in range(n):
        r_t = float(w @ R[t])
        cost = 0.0
        if t == 0:
            # entry trade: buy the whole book
            cost += float((w_target * bps).sum())
            turnover[t] += float(w_target.sum())
        w_post = w * (1.0 + R[t])
        s = float(w_post.sum())
        if s > 0:
            w_post /= s
        if rebal[t]:
            turnover[t] += float(np.abs(w_target - w_post).sum())
            cost += float((np.abs(w_target - w_post) * bps).sum())
            w = w_target.copy()
        else:
            w = w_post
        if exit_cost and t == n - 1:
            # liquidation at the final close
            cost += float((w_post * bps).sum())
            turnover[t] += float(w_post.sum())
        port[t] = r_t - cost

    gross = np.empty(n)  # recompute gross curve for cost_drag
    wg = w_target.copy()
    for t in range(n):
        gross[t] = float(wg @ R[t])
        wp = wg * (1.0 + R[t])
        sg = float(wp.sum())
        wg = wp / sg if sg > 0 else wp
        if rebal[t]:
            wg = w_target.copy()

    total = float((1.0 + port).prod() - 1.0)
    gross_total = float((1.0 + gross).prod() - 1.0)
    mean = float(port.mean())
    std = float(port.std()) if n > 1 else 0.0
    ann_ret = (1.0 + total) ** (TRADING_DAYS / n) - 1.0 if n else 0.0
    ann_vol = std * TRADING_DAYS ** 0.5
    sharpe = (mean / std * TRADING_DAYS ** 0.5) if std else 0.0
    curve = np.cumprod(1.0 + port)
    max_dd = abs(float((curve / np.maximum.accumulate(curve) - 1.0).min()))

    return {
        "n_days": int(n),
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover": float(turnover.sum()),
        "cost_total": gross_total - total,
    }


# ---------------------------------------------------------------------------
# Decision-series backtests (launchpad/rubric/decision-series.md)
#
# A submission may be a chronological series of target-portfolio records
# instead of one static book. Each record carries information_cutoff /
# decision_time / execution_time; the engine transitions the book to the
# record's target at its execution day and drifts between decisions.
# ---------------------------------------------------------------------------

SERIES_MAX_WEIGHT = 0.20          # per-name cap from the format spec
SERIES_ACTIONS = ("rebalance", "hold")


def _parse_ts(v) -> float | None:
    """ISO8601 string -> epoch float (naive treated as UTC)."""
    from datetime import datetime, timezone
    if v is None:
        return None
    try:
        s = str(v).replace("Z", "+00:00")
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except (ValueError, TypeError):
        return None


def validate_decision_series(records) -> tuple[list[dict], list[str]]:
    """Validate a decision-series submission.

    Returns (decisions, errors): decisions normalized to
    [{"date": <date>, "weights": {ticker: w}}] sorted by execution date —
    "weights" is None for action="hold". Any error aborts the run: a
    malformed series scores zero, not partial.
    """
    import datetime as _dt

    errors: list[str] = []
    if not isinstance(records, list) or not records:
        return [], ["series missing or empty"]

    normed = []
    prev_id = 0
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            errors.append(f"record {i}: not an object")
            continue
        if str(r.get("schema_version", "")) != "1.0":
            errors.append(f"record {i}: schema_version must be '1.0'")
        did = r.get("decision_id")
        if not isinstance(did, int) or did <= prev_id:
            errors.append(f"record {i}: decision_id must be an int > prior "
                          f"(got {did!r})")
        else:
            prev_id = did

        ic = _parse_ts(r.get("information_cutoff"))
        dt = _parse_ts(r.get("decision_time"))
        et = _parse_ts(r.get("execution_time"))
        if ic is None or dt is None or et is None:
            errors.append(f"record {i}: unparseable timestamps")
        elif not (ic <= dt < et):
            errors.append(
                f"record {i}: require information_cutoff <= decision_time "
                f"< execution_time")

        if r.get("action", "rebalance") not in SERIES_ACTIONS:
            errors.append(f"record {i}: action must be one of {SERIES_ACTIONS}")

        holds = r.get("target_holdings")
        if not isinstance(holds, list):
            errors.append(f"record {i}: target_holdings must be a list")
        else:
            tot = 0.0
            for leg in holds:
                t, w = (leg.get("ticker"), leg.get("weight")) \
                    if isinstance(leg, dict) else (None, None)
                if not isinstance(t, str) or not t:
                    errors.append(f"record {i}: leg missing ticker")
                    continue
                if not isinstance(w, (int, float)) or w < 0:
                    errors.append(f"record {i}: {t} bad weight {w!r}")
                    continue
                if w > SERIES_MAX_WEIGHT + 1e-9:
                    errors.append(
                        f"record {i}: {t} weight {w} exceeds "
                        f"{SERIES_MAX_WEIGHT} cap")
                tot += w
            if r.get("action", "rebalance") == "rebalance" and \
                    abs(tot - 1.0) > 0.01:
                errors.append(
                    f"record {i}: weights sum {tot:.4f} != 1.0 +/- 0.01")

        exec_date = None
        if et is not None:
            exec_date = _dt.datetime.fromtimestamp(et, _dt.timezone.utc).date()
        normed.append({
            "date": exec_date,
            "weights": {
                leg["ticker"]: float(leg["weight"]) for leg in (holds or [])
                if isinstance(leg, dict) and isinstance(leg.get("ticker"), str)
                and isinstance(leg.get("weight"), (int, float))
            } if r.get("action", "rebalance") == "rebalance" else None,
        })

    normed.sort(key=lambda d: d["date"] or _dt.date.max)
    ex_dates = [d["date"] for d in normed if d["date"]]
    if len(ex_dates) != len(set(ex_dates)):
        errors.append("duplicate execution dates in series")

    return ([d for d in normed if d["date"]] if not errors else []), errors


def run_backtest_series(rets, dates: list, decisions: list[dict],
                        *, cost_bps: float | None = None,
                        exit_cost: bool = True) -> dict:
    """Event-driven variant of run_backtest for decision-series submissions.

    decisions: [{"date": <date>, "weights": {ticker: w}}] sorted by
    execution date (output of validate_decision_series). At each execution
    day the book transitions to that record's target; between executions the
    book drifts with prices. A decision dated <= the window start is the
    opening book (entry cost on its weights, day-0 return earned on it —
    identical to run_backtest's t=0 trade). A decision landing on a
    non-trading day executes on the next trading day. Execution dates
    beyond the window are ignored.

    Identical semantics to run_backtest: day-t return accrues on pre-
    transition drifted weights; transition cost = |w_new - w_drifted| at
    that leg's bps; liquidation cost at the final close when exit_cost.
    A one-decision series executed at the window start is exactly a
    buy-and-hold run_backtest.
    """
    import bisect
    import polars as pl  # local import keeps module import cheap

    tickers = sorted({t for d in decisions for t in (d["weights"] or {})})
    if CASH_TICKER in tickers and CASH_TICKER not in rets.columns:
        rets = rets.with_columns(pl.lit(0.0).alias(CASH_TICKER))
    cols = [t for t in tickers if t in rets.columns]
    if rets.height == 0 or not cols or not decisions:
        return {}

    R = np.nan_to_num(rets.select(cols).to_numpy(), nan=0.0)
    n = R.shape[0]
    jcol = {t: i for i, t in enumerate(cols)}
    bps = np.array([
        (cost_bps if cost_bps is not None else leg_bps(c)) / 1e4
        for c in cols
    ])

    def target_vec(weights: dict) -> np.ndarray:
        v = np.zeros(len(cols))
        for t, w in weights.items():
            if t in jcol:
                v[jcol[t]] = w
        return v

    # execution-day -> list of target vectors (record order preserved).
    # Dates not in the trading calendar clamp forward to the next day.
    dset = sorted(dates)
    by_day: dict[int, list] = {}
    for d in decisions:
        i = bisect.bisect_left(dset, d["date"])  # next trading day >= exec
        if i < n:
            by_day.setdefault(i, []).append(
                target_vec(d["weights"]) if d["weights"] is not None else None)

    w = np.zeros(len(cols))          # held weights (drift), starts all-cash
    port = np.empty(n)
    turnover = np.zeros(n)
    for t in range(n):
        cost = 0.0
        if t == 0 and 0 in by_day:
            # opening book established before the day's return — v1 parity.
            for tv in by_day[0]:
                if tv is None:       # 'hold' as first record: stay in cash
                    continue
                turnover[0] += float(np.abs(tv - w).sum())
                cost += float((np.abs(tv - w) * bps).sum())
                w = tv.copy()
        r_t = float(w @ R[t])
        w_post = w * (1.0 + R[t])
        s = float(w_post.sum())
        if s > 0:
            w_post /= s
        if t > 0 and t in by_day:
            for tv in by_day[t]:
                if tv is None:       # 'hold': keep drifting
                    continue
                turnover[t] += float(np.abs(tv - w_post).sum())
                cost += float((np.abs(tv - w_post) * bps).sum())
                w_post = tv.copy()
        w = w_post
        if exit_cost and t == n - 1:
            cost += float((w_post * bps).sum())
            turnover[t] += float(w_post.sum())
        port[t] = r_t - cost

    # gross curve for cost_drag: same transitions, no costs
    gross = np.empty(n)
    wg = np.zeros(len(cols))
    for t in range(n):
        if t == 0 and 0 in by_day:
            for tv in by_day[0]:
                if tv is not None:
                    wg = tv.copy()
        gross[t] = float(wg @ R[t])
        wp = wg * (1.0 + R[t])
        sg = float(wp.sum())
        wg = wp / sg if sg > 0 else wp
        if t > 0 and t in by_day:
            for tv in by_day[t]:
                if tv is not None:
                    wg = tv.copy()

    total = float((1.0 + port).prod() - 1.0)
    gross_total = float((1.0 + gross).prod() - 1.0)
    mean = float(port.mean())
    std = float(port.std()) if n > 1 else 0.0
    ann_ret = (1.0 + total) ** (TRADING_DAYS / n) - 1.0 if n else 0.0
    ann_vol = std * TRADING_DAYS ** 0.5
    sharpe = (mean / std * TRADING_DAYS ** 0.5) if std else 0.0
    curve = np.cumprod(1.0 + port)
    max_dd = abs(float((curve / np.maximum.accumulate(curve) - 1.0).min()))

    return {
        "n_days": int(n),
        "n_decisions": int(len(decisions)),
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover": float(turnover.sum()),
        "cost_total": gross_total - total,
    }
