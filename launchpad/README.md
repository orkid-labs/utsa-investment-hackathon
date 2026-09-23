# Hackathon Launchpad — Portfolio Management API

Everything you need to ship a working, judgeable app on the dataset.
You don't need to know Rust or wrangle parquet — the `statevector`
package handles the data layer; you write the finance logic.

## What you're actually building

A small web API that behaves like a portfolio manager's desk:

- `/holdings` — "what do we own today, and how much?" (weights)
- `/backtest` — "how would this book have performed?" (the judges
  recompute your numbers against the dataset — no fudging)
- `/screen` — "find me names matching these criteria"
- `/asof` — "what did we know about this company on this date?"

The starter app in `template/` already implements all of these
correctly but simply. Your job: replace the simple finance logic with
your own — better selection, better weighting, better analytics.

**Read first:** `../FEATURES.md` — plain-English explanation of all 27
features, the timing clocks, and the quality flags. Then `RULES.md`.

## 1. Setup (5 min)

```bash
pip install -e sdk/                      # the data SDK (polars+pandas inside)
pip install -r launchpad/template/requirements.txt
export SV_DATA_ROOT=/path/to/state-vector
svq doctor                               # verifies the dataset is intact
```

Hosted thin client instead? `export SV_DATA_ROOT=https://<host>.ts.net`
and `SV_DATA_TOKEN=<your-token>` — same commands.

## 2. Run the starter app — it already works

```bash
cd launchpad/template
uvicorn app:app --port 8000
```

Open http://localhost:8000/docs — an interactive Swagger UI where you
can call every endpoint from your browser, no code needed.

Try it:

```bash
curl localhost:8000/portfolio/holdings
curl -X POST localhost:8000/backtest -H 'Content-Type: application/json' \
  -d '{"tickers":["AAPL","MSFT"],"weights":[0.5,0.5],"start":"2020-01-02","end":"2020-12-31"}'
curl "localhost:8000/screen?min_adv=50000000&limit=10"
curl "localhost:8000/asof?ticker=AAPL&on=2024-03-31"
```

## 3. Trading rules (the mandate)

- **Long-only everything.** Stocks: buy, then sell to close — never short.
  Options: buy calls/puts, sell to close — never write. Every holding
  weight must be ≥ 0 and weights sum to ~1.0.
- **Hedging = protective puts.** Long book + long put legs. An option leg
  is a holding with an OCC ticker, e.g. `{"ticker": "O:AAPL250117P00220000",
  "weight": 0.05}`.
- Universe = `ds.universe()` (US names). Stay inside it.
- The trailing 30 calendar days are a **sealed holdout** — never train or
  validate on them (`holdout_cutoff()`).

## 4. Make it yours

The reference implementations are deliberately simple (equal-weight by
liquidity, close-to-close backtest). Differentiation ideas:

- **Selection**: momentum, sector tilts (`ds.sectors()`), low-vol screens,
  options-implied signals (`ds.smile()` — rising skew = demand for
  protection; `ds.options_greeks()` for contract-level Greeks)
- **Weighting**: risk parity, vol-targeting, mean-variance on the dataset
- **Analytics**: factor attribution vs `ds.benchmark("SPX")`, drawdown
  decomposition, earnings-event overlays (`ds.calendar()` +
  `days_to_next_report`)
- **Endpoints**: `/rebalance`, `/risk`, `/explain` — add any you like

Suggested first afternoon: pull `ds.state_vector()`, pick 3-4 features
you understand from FEATURES.md, rank the universe, and compare the
top decile's forward returns to the bottom decile's. That's the entire
workflow — everything else is refinement.

## 5. How judging works

Judging is **deterministic** — `rubric/check.py` hits your running app and
scores it against `rubric/rubric.yaml`. Self-score anytime:

```bash
SV_DATA_ROOT=/path/to/state-vector \
python launchpad/rubric/check.py --base-url http://localhost:8000
```

| check | pts | what it verifies |
|---|---|---|
| health | 5 | `/health` responds ok |
| holdings_shape | 10 | holdings list w/ ticker+weight |
| holdings_weights_sum | 10 | weights≈1.0, tickers in universe |
| backtest_shape | 15 | required metrics present |
| backtest_values | 25 | matches reference within 2% |
| screen_shape | 10 | filtered results list |
| asof_pit | 10 | no lookahead (dates ≤ `on`) |
| responsiveness | 15 | all calls < 30s |

**The reference app scores 100.** Keep the judged paths and response
shapes intact and you're guaranteed a complete submission.

Note `backtest_values`: the scorer recomputes performance from your
weights against the dataset itself. A metric that disagrees with the
recompute scores zero — build on the real data, not assumed returns.

## 6. Data traps that will bite you (read `DEVIATIONS.md`)

- **Point-in-time**: use `ds.fundamentals(t, asof=...)` — raw fundamentals
  leak (fiscal period-ends, not filing dates; estimates are one snapshot)
- **Holdout**: the trailing 30 calendar days are sealed —
  `holdout_cutoff()` gives you the cutoff
- **Universe**: `ds.universe()` = ~1,258 US names with complete coverage;
  the 1,275-name valuation/options panel is a fixed universe (survivorship)
- **Sparse estimates**: FCF/capex estimates are mostly absent — real
  markets, real missingness
- **Options spreads**: no quote-level NBBO — use high-low range as proxy
- **Nulls are information**: a null `skew_25d` usually means "no liquid
  options chain that day," not a pipeline bug — check the flags first

## 7. Cheat sheet

```python
ds.prices("AAPL", start="2020-01-01")        # OHLCV + ret_1d (pandas)
ds.state_vector("AAPL")                      # 27 features + clocks + flags
ds.state_vector("AAPL", pca=True)            # 22 principal components
ds.options_chain("AAPL", on="2024-06-21")    # contract bars
ds.options_greeks("AAPL", on="2024-06-21")   # BS Greeks + IV per contract
ds.smile("AAPL")                             # daily smile summary
ds.fundamentals("AAPL", asof="2024-03-31")   # PIT-safe
ds.benchmark("SPX")                          # index series
ds.adv("AAPL", days=20)                      # avg dollar volume
ds.sectors()                                 # industry map
ds.trading_days("2020-01-01","2020-12-31")   # valid dates
ds.report("dq_report")                       # data-quality report
```
