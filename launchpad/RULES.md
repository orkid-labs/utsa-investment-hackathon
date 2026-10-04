# Hackathon Trading Rules

## Portfolio semantics

Your app holds positions expressed as weights per ticker-day:

- **Stocks are long-only.** Buy a stock, later sell it to close.
  No short stock positions — ever.
- **Options are long-only.** Buy a contract, later sell it to close.
  No writing or shorting options.
- **Hedging = purchased puts** (protective puts). That's the only
  hedge available under long-only rules. Why: writing a put is
  short-volatility exposure — the mandate is a long-only book that may
  *buy* insurance, not sell it.
- Weights are **nonnegative** and sum to **~1.0** (±1%). Hold a cash
  sleeve with the **`CASHHOLDING`** pseudo-ticker: flat 0% return,
  zero transaction cost — the way you de-risk or stage entries inside
  the book. `[AAPL 0.6, CASHHOLDING 0.4]` is 60% invested, 40% cash.

## Tickers

- Stocks: universe roots — `AAPL`, `MSFT`, `NVDA`, … (see
  `GET /universe` or `ds.universe()`; ~1,258 US names with options
  coverage).
- Options: **OCC tickers** — `O:AAPL200221P00315000`
  (`O:` + root + YYMMDD + C/P + strike×1000, zero-padded to 8 digits).
  Option legs live in the same `holdings` array as stocks.
- `ds.options_chain(ticker)` and `options_daily` carry contract-level
  daily bars (`O:` prefix) for the full 2014→ window.
- `ds.options_greeks(ticker)` serves BS2002 delta/gamma/vega/theta/rho +
  IV per contract-day — use it for option selection and hedge sizing.
- `ds.smile(ticker)` / `ds.microstructure(ticker)` serve the daily
  smile and minute-bar liquidity panels.

## Corporate actions

- `corporate_actions` (structural): `ticker, ex_date, kind, value, raw`
  - `kind=split` → `value` = to/from ratio (2-for-1 → 2.0)
  - `kind=dividend:*` → `value` = cash per share, `raw` has pay date
- `splits_all` / `dividends_all` carry the raw Massive rows.
- Total-return calculations should use `ex_dividend_date` (the PIT-correct
  field — `pay_date` arrives later). Why it matters: on the ex-date the
  stock drops by roughly the dividend amount — ignoring it counts a
  real cash payment as a price loss.

## Transaction costs

Every backtest is evaluated **net of fees** — the reference engine is
authoritative and applies the same model to your reported metrics:

- **Stocks: 10 bps** per dollar traded. **Options (O: legs): 50 bps**
  per dollar traded — wider spreads are the honest cost of convexity.
- Buys and sells both pay. Turnover on a trade day is
  `Σ|w_target − w_held|` per leg.
- Entering the book at `start` is a trade; liquidating at `end` is a
  trade. A simple buy-and-hold round trip therefore costs ~20 bps
  on the invested notional.
- `rebalance` decides how often you pay: `"none"` = buy-and-hold
  (weights drift, entry+exit only); `"daily"`, `"weekly"`,
  `"monthly"` reset to target weights at that day's close and pay the
  turnover. Rebalancing more often is a real decision now.
- `CASHHOLDING` legs trade free — moving into or out of the cash
  sleeve costs 0 bps; the stock side of the trade still pays its rate.
- `POST /backtest` accepts an optional `cost_bps` override (flat,
  per-leg) for sensitivity analysis — the reference recomputes with
  whatever you declare, so declaring a lower number moves the
  goalposts with you. The judged default is the schedule above.
- Your response must report `turnover` and `cost_total` alongside the
  performance metrics — cost disclosure is scored.

The shared engine is `statevector.backtest.run_backtest` — import it
rather than re-deriving; it is exactly what the judges run.

## Data windows

- Storage: **2014-01-01 → present**
- Training: **2017-01-01 → present − 30 days**
- Sealed **30-calendar-day trailing holdout** — `ds.holdout_cutoff()`
  returns the boundary; treat everything after it as evaluation-only.
  Why sealed: it's the judges' test set. Fitting on it is lookahead
  bias — your backtest would claim knowledge of the future.

## Point-in-time rules

The single most important concept in this dataset: **you may only use
information that was public on the simulated date.**

- Fundamentals are joined by **filing date** (when the market could
  first see the number), not period end. A quarter ending Dec 31 isn't
  knowable until the 10-Q/10-K files in February. Joining on period-end
  leaks ~5 weeks of future information into every backtest — the
  classic way naive backtests show fake alpha. Use `ds.asof()` /
  `GET /asof?ticker=&on=` — don't hand-join on `period_end`.
- `days_to_next_report` is a **projection** from past filing cadence,
  not the true future calendar. That's intentional — a company filing
  earlier or later than usual is itself information you can't have
  in advance.
- Estimates are a **single snapshot** (2026-08-28 vintage) — a known,
  documented limitation. Don't read them as historical estimate
  history; see `DEVIATIONS.md`.

## Scoring contract

The deterministic scorer (`rubric/check.py`) calls your running app:

- `GET /health`, `GET /universe`
- `POST /holdings {date, weights}` — weights ≥0, Σ≈1
- `POST /backtest {start, end, holdings}` — recomputed against the
  dataset; your reported metrics are *not* trusted
- `GET /asof`, `GET /panel/{name}` — response shapes validated
- Latency budget applies (see `rubric.yaml`)

Fair play: the reference backtester is authoritative. A "better" metric
that disagrees with the recomputed one scores zero, not partial credit.
This is the same standard a skeptical PM applies: if your tear sheet
doesn't match the desk's recompute, the desk wins.
