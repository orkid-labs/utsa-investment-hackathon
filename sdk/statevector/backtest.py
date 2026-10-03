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


def leg_bps(ticker: str) -> float:
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
