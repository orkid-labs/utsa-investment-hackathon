# statevector — contestant SDK

Access layer for the Equity Composite State Vector dataset. You build the
portfolio-management app; this package handles the data plumbing. Every
accessor returns a plain **pandas DataFrame** — no Rust, no lazy evaluation,
no parquet wrangling.

## Concepts in five minutes

**What the data is.** A collection of *panels* — think of each as one big
tidy table saved as parquet. The star panel is `state_vector`: one row per
(ticker, trading day) containing 27 features + 2 timing clocks + 7 data
flags describing everything knowable about that company that day. See
**`FEATURES.md`** (repo root) for what every feature means.

**Point-in-time (PIT) safety.** The cardinal sin of backtesting is
*lookahead bias* — using data that wasn't public yet. Example: a company's
Q4 ends Dec 31, but the 10-Q isn't *filed* until Feb. If you join on
period-end you leak a month of information. This dataset joins
fundamentals by **filing date** — the day the market could first see the
number. `ds.fundamentals(t, asof=...)` enforces this; raw panels don't.
Always use the `asof`-style accessors in backtests.

**The sealed holdout.** The trailing 30 calendar days are reserved for
final evaluation — like a test set your professor keeps. `holdout_cutoff()`
returns the boundary; never train or validate past it. Using it is an
automatic integrity fail.

**Universe.** `ds.universe()` ≈ 1,258 US tickers with complete coverage
2014→present. Stay inside it — the rubric checks.

## Install

```bash
pip install -e sdk/            # from the repo root
pip install -e "sdk/[serve]"   # + FastAPI starter API
export SV_DATA_ROOT=/path/to/state-vector
svq doctor                     # sanity-check your dataset
```

## 5-minute quickstart

```python
from statevector import Dataset, holdout_cutoff

ds = Dataset()                             # reads $SV_DATA_ROOT

ds.universe()                              # ~1,258 US tickers (effective)
px = ds.prices("AAPL", start="2020-01-01") # OHLCV + ret_1d
f  = ds.fundamentals("AAPL", asof="2024-03-31")  # PIT-safe
spx = ds.benchmark("SPX")                  # index series
sv = ds.state_vector("AAPL")               # the 27-feature rows

holdout_cutoff()                           # exclude dates >= this
```

Then run `launchpad/examples/quickstart.py` for a guided tour, and
`launchpad/template/app.py` for a working API you extend — it scores
100/100 on the public rubric out of the box.

## Thin client (hosted data — no local dataset needed)

If you were given a data URL + token instead of a dataset download:

```python
ds = Dataset("https://<host>.ts.net", token="<your-token>")
# or: export SV_DATA_ROOT=https://<host>.ts.net SV_DATA_TOKEN=<token>
ds = Dataset()
```

Everything below works identically — panels are scanned lazily over
HTTP range requests, so filters only download the row-groups needed.
Tips: filter by `ticker`/`start`/`end` before collecting; prefer
`ticker`-level pulls over full-panel scans on the partitioned panels
(`stocks_daily`, `options_daily`, `options_greeks_daily`).

## The one-stop accessor

```python
ds.get("stocks_daily", ticker="NVDA", start="2024-01-01")  # pandas
ds.get("options_agg_daily", ticker="TSLA", limit=500)
ds.panel("valuation_weekly")               # polars LazyFrame (power users)
```

## Domain accessors

| call | returns |
|---|---|
| `ds.prices(t, start, end)` | OHLCV + `ret_1d` |
| `ds.state_vector(t, start, end, pca=False)` | the 27-feature vector (`pca=True` → 22 PCs) |
| `ds.options_chain(t, on=)` | contract-level bars (OCC tickers) |
| `ds.options_greeks(t, on=)` | BS delta/gamma/vega/theta/rho + IV per contract-day |
| `ds.smile(t, start, end)` | daily smile summary (ATM IV, skew, term slope) |
| `ds.microstructure(t, start, end)` | minute-bar liquidity stats |
| `ds.fundamentals(t, asof=)` | PIT-safe quarterly fundamentals |
| `ds.benchmark("SPX"/"MID"/"SML")` | index OHLCV |
| `ds.adv(t, days=20)` | avg dollar volume |
| `ds.sectors()` | ticker → name, industry, exchange, mktcap |
| `ds.calendar(t)` | SEC filing dates (the PIT gate) |
| `ds.guidance(t)` | company guidance events (EPS/rev min/max, PIT-stamped) |
| `ds.trading_days(start, end)` | valid trading dates |
| `ds.holidays()` | market holiday calendar |
| `ds.universe()` | effective-universe tickers |
| `ds.report("dq_report")` | data-quality report text |
| `parse_occ("O:AAPL...")` | underlying, expiry, side, strike |

## Worked example — a simple factor screen

Rank the universe on valuation cheapness and check a candidate's
options-market sentiment:

```python
sv = ds.get("state_vector", start="2025-01-01")

# latest row per ticker
latest = sv.sort_values("date").groupby("ticker").tail(1)

# cheap stocks with liquid options chains
cand = latest[(latest["composite_valuation_gap"] < -1) &
              (latest["options_thin_chain"] == 0)]

# what does the options market think of the cheapest one?
t = cand.sort_values("composite_valuation_gap").iloc[0]["ticker"]
print(ds.smile(t).tail())          # skew / term_slope / atm_iv
```

## CLI

```bash
svq doctor                    # verify install + expected row counts
svq panels                    # panel name -> row count
svq head stocks_daily --ticker AAPL
svq tickers | head
svq asof AAPL 2024-03-31      # PIT fundamentals
svq calendar AAPL             # filing dates
svq serve --port 8000         # generic dataset API
```

## Trading rules (the mandate)

Long-only everything — stocks and options alike. Buy to open, sell to
close. All holding weights ≥ 0, sum ≈ 1. To hedge, buy a protective put:
`{"ticker": "O:AAPL200221P00315000", "weight": 0.05}`.
Full rules: `launchpad/RULES.md`.

## Panels

| panel | contents | range |
|---|---|---|
| `state_vector` | **the main event** — 27 features + clocks + flags per ticker-day | 2014→ |
| `state_vector_pca` | 22 principal components of the above | 2014→ |
| `fundamentals_actuals` | quarterly actuals (EPS, margins, EBIT/DA, FCF, capex, net debt) | 2000→ |
| `fundamentals_estimates_hist` | analyst estimates (single vintage — see DEVIATIONS) | 2000→ |
| `fundamentals_estimates_recent` | forward estimates | 2025→2029 |
| `valuation_weekly` | mktcap, shares, EV/EBITDA, EV/S, P/E, P/B (Fridays) | 2011→ |
| `options_agg_daily` | aggregate IV + call/put OI + volume | 2016→ |
| `index_daily` | SPX/MID/SML OHLCV (long format) | 2016→ |
| `stocks_daily/` | per-ticker daily OHLCV (Massive SIP) | 2014→ |
| `stocks_minute/` | per-ticker minute bars (Massive SIP) | 2014→ |
| `options_daily/` | contract-level day bars, OCC tickers (Massive OPRA) | 2014→ |
| `guidance_all` | Benzinga guidance events (EPS/rev min/max/est, methods) | 2014→ |
| `dividends_all` | dividend declarations + ex-dates | 2014→ |
| `splits_all` | split events | 2014→ |
| `filings_10k`/`filings_10q` | SEC filing index (report calendar source) | 2014→ |
| `macro_*` | treasury yields, funding, CPI/PCE, inflation expectations, labor | → |

## Known data traps (read DEVIATIONS.md)

- Estimates are a single 2026-08-28 snapshot — `snapshot_track_used` flag set
- Valuation/options panels are a fixed 1,275-name universe (survivorship)
- Options quotes are not in the dataset — spread proxy via high-low range
- No announcement dates in Bloomberg fields — use `report_calendar_us` (SEC filings)
- `guidance_absent` applies to ~14% of the universe — guidance coverage is 86% (2014→)
