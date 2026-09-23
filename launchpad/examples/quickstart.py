"""Guided tour of the state-vector dataset — run top to bottom in Jupyter
(or `python quickstart.py`). Each section prints real output so you can
sanity-check your install.

Prereqs: pip install -e ../../sdk ; export SV_DATA_ROOT=/path/to/state-vector
  (or SV_DATA_ROOT=https://<host>.ts.net + SV_DATA_TOKEN=<token>)

The mental model: every accessor returns a pandas DataFrame. One row =
one (ticker, day) or one filing/contract event. Everything is
point-in-time safe — a row dated 2020-03-15 only contains what was
public that morning.
"""

from statevector import Dataset, holdout_cutoff, parse_occ

ds = Dataset()  # reads SV_DATA_ROOT

# -- 1. What's here -----------------------------------------------------------
print(ds.panels())           # panel name -> row count
print(len(ds.universe()), "US tickers")

# -- 2. Prices ----------------------------------------------------------------
# Daily OHLCV + ret_1d. Nothing exotic — this is your benchmark data.
aapl = ds.prices("AAPL", start="2024-01-01")
print(aapl[["date", "close", "ret_1d"]].tail())

# -- 3. The state vector ------------------------------------------------------
# The core object: 27 features + 2 clocks + 7 flags per (ticker, day).
# See FEATURES.md for what each column means.
sv = ds.state_vector("AAPL", start="2024-01-01")
print(sv[["date", "spot", "atm_iv", "composite_valuation_gap",
          "fundamental_surprise", "days_to_next_report"]].tail())

# -- 4. Fundamentals, point-in-time -------------------------------------------
# What was knowable on 2024-03-31 — actuals filed on/before that date
# only. The single most important guard against lookahead bias.
print(ds.fundamentals("AAPL", asof="2024-03-31"))

# -- 5. Options ---------------------------------------------------------------
# Contract-level daily bars (OCC tickers) + solved Greeks.
try:
    chain = ds.options_chain("AAPL", on="2024-06-21").head()
    chain["parsed"] = chain["ticker"].map(parse_occ)
    print(chain)
    # Daily smile summary: what vol/skew the market was pricing:
    print(ds.smile("AAPL", start="2024-06-01").tail())
except KeyError:
    print("options panels not available")

# -- 6. A tiny factor screen ---------------------------------------------------
# Classic quant 101: rank by momentum, compare to valuation.
px = ds.get("stocks_daily", start="2024-01-01", limit=500_000)
mom = (
    px.sort_values("date")
      .groupby("ticker")
      .apply(lambda g: g["close"].iloc[-1] / g["close"].iloc[0] - 1)
      .sort_values(ascending=False)
)
print("top 5 by 2024 momentum:\n", mom.head())

# Same idea on a state-vector feature: who looks cheapest right now?
# Pull a recent window across the whole universe, keep each ticker's
# last row, rank.
recent = ds.get("state_vector", start="2026-08-01")
latest = recent.sort_values("date").groupby("ticker").tail(1)
print("cheapest 5 by composite_valuation_gap:\n",
      latest.nsmallest(5, "composite_valuation_gap")[["ticker",
          "composite_valuation_gap"]])

# -- 7. Holdout — the sealed window -------------------------------------------
print("exclude rows on/after:", holdout_cutoff())
