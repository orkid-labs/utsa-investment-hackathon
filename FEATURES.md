# FEATURES.md — the state vector, explained for finance people

The dataset's core object is the **state vector**: for every (ticker,
trading day) pair, a row of 27 numbers + 2 clocks + 7 flags that
summarizes everything knowable about that company *on that day* —
fundamentals, valuation, market microstructure, options-market
information, and the macro backdrop.

Think of each row as a Bloomberg terminal screenshot compressed into a
feature vector — except it's point-in-time safe (nothing in a row uses
information that wasn't public that morning) and machine-readable.

```python
sv = ds.state_vector("AAPL", start="2024-01-01")   # one row per day
```

Use it three ways:

- **As screening signals** — rank the universe on `atm_iv`,
  `composite_valuation_gap`, etc.
- **As model inputs** — feed rows into regression/ML; the `pca` panel
  gives you a 22-component compressed version.
- **As context for portfolio rules** — e.g., scale position size by
  `amihud_illiq`, time entries around `days_to_next_report`.

Below: every field, what it measures, and how to read it.

---

## Fundamentals & information dynamics

How the business is doing, and how fast that information is changing.

| feature | what it is | how to read it |
|---|---|---|
| `fwd_fcf_fair_value` | Model fair value per share: latest knowable free cash flow per share ÷ a 5% FCF-yield anchor (perpetuity-style anchor) | Spot far above = rich on cash flow; far below = cheap. Compare with `log_fv_gap`, which is the same idea in logs |
| `fundamental_surprise` | (Actual − guided midpoint) / \|mid\| — how the latest reported results landed vs. what management guided. FY guidance is compared to trailing-twelve-month actuals; quarterly guidance to the latest quarter. Before a company's first guidance event, falls back to the analyst-estimate snapshot | >0 = beat, <0 = miss. Persistent positives = management under-promising |
| `kl_surprise_bits` | The same surprise expressed in *bits of information* — how far the event should move a Bayesian's belief (Gaussian KL divergence between prior and posterior, using guidance width as uncertainty) | Big bits = genuinely informative event; ~0 = the print told you nothing new |
| `measured_half_life` | Estimated decay of surprise relevance: AR(1) fitted to \|surprise\| across events, converted to a half-life (≈21 days when too few events — see `half_life_imputed` flag) | Short = market digests this name's news fast; long = slow-moving information |
| `growth_kalman_update` | Kalman-filter innovation on the sales series — the portion of each revenue print the trend model did *not* expect | Persistent positive = accelerating growth state; negative = decelerating |
| `fundamental_confidence` | 1 / (1 + guidance width / \|mid\|) — how precise management's guidance range is | Near 1 = tight, confident guidance; near 0 = wide "we don't know either" range |
| `log_fv_gap` | log(spot / `fwd_fcf_fair_value`) | Positive = trading above the FCF anchor. Distribution center is ~0 only if price ≈ fair value |
| `guidance_range_velocity` | Change in guidance midpoint per day, measured between *same-period* events only | Positive = management raising over time; a fast negative is a guide-down cycle |

## Valuation

Where price sits relative to the company's own history and fundamentals.

| feature | what it is | how to read it |
|---|---|---|
| `composite_valuation_gap` | Mean z-score of log P/E, EV/EBITDA, EV/Sales, P/B — each z-scored against the stock's own trailing year | >0 = expensive vs. its own history (all four multiples elevated); <0 = cheap |
| `valuation_kurtosis` | Rolling 2-year kurtosis of that gap | High = valuation swings come in bursts (regime-y name); low = stable re-rating |
| `cornish_fisher_gap` | The valuation z-score corrected for skewness and kurtosis (Cornish–Fisher expansion) | Same signal as `composite_valuation_gap` but adjusted for asymmetric tails — better behaved for ranking |
| `mean_reversion_speed` | AR(1) on the valuation gap → annualized −ln(ρ): how fast the gap has historically closed | High = mispricings correct quickly (harvest fast or not at all); low = gaps persist |

## Market microstructure

Built from **minute bars**, not daily candles — these describe the
*trading process*, not just where price ended up.

| feature | what it is | how to read it |
|---|---|---|
| `minute_realized_diffusion` | Annualized realized vol from minute returns: √(Σr²)·√252 | The honest volatility measure — captures intraday swings daily closes hide |
| `amihud_illiq` | Mean(\|return\| / dollar volume)·10⁶ — Amihud's classic price-impact measure | High = small trades move price (expensive to size into); low = deep liquidity |
| `kyle_lambda` | Regression slope of price change on tick-rule-signed volume — Kyle's λ: how much a unit of order flow moves price | High = informed-flow sensitivity; rising λ ahead of events = information leaking into price |
| `fdt_deviation` | Mean \|cumRVᵢ/RV − i/N\| over the day — how far intraday volume/vol timing deviates from even arrival (the U-shape residual) | High = volatility arrives in bursts (auctions, news drops); low = smooth trading day |
| `liquidity_roc` | Today's dollar volume ÷ 20-day mean − 1 | Participation shock: >0 = unusually heavy trading (attention, news, rebalancing) |

## Options market

From **contract-level OPRA** day bars — the implied-vol surface is
solved in-house (Black–Scholes inversion with Newton + bisection
fallback), not vendor fields.

| feature | what it is | how to read it |
|---|---|---|
| `atm_iv` | 30-day at-the-money implied volatility | The market's priced forecast of vol. Compare to `minute_realized_diffusion` |
| `variance_risk_premium` | IV² − RV² — implied minus realized variance | Positive = options charge a premium over delivered vol (normal); negative = market underpricing risk |
| `skew_25d` | (IV of 5%-OTM puts − IV of 5%-OTM calls) / ATM IV | Crash-insurance premium. Steep/rising skew = demand for downside protection. ~52% null — needs a liquid options chain |
| `rn_kurtosis` | Risk-neutral tail kurtosis from a quadratic smile fit | High = market pricing fat tails / jump risk |
| `oi_divergence` | z-scored Δ(put OI − call OI) vs. the signed price move | Put OI building while price rises (or the reverse) = positioning diverging from price |
| `term_slope` | ATM IV(90d) − ATM IV(30d) | Positive = normal upward term structure; negative (inverted) = near-term stress priced over long-run risk |

## Macro overlay

Same value for every ticker on a given day — the regime backdrop.
Most useful *interacted with* stock-level features (two are already).

| feature | what it is | how to read it |
|---|---|---|
| `funding_stress` | 90-day commercial paper rate − effective fed funds | Wholesale funding premium — rises in credit stress (2008/2020-style widening) |
| `inflation_expectation` | 5y5y forward inflation breakeven | The market's long-run inflation anchor |
| `treasury_funding_interact` | 10-year yield × `funding_stress` | High rates *and* stressed funding = the discount-rate + credit double-hit |
| `mktcap_duration_interact` | z(log market cap) × daily change in 10-year yield | Captures that large-cap (long-duration equity) names react harder to rate moves |

## Clock fields

Not features — *when* things happen. Timing context for the row.

| field | what it is |
|---|---|
| `days_to_next_report` | Days until the next *projected* SEC filing: last observed filing + expanding median cadence. **Deliberately a projection** — the true future filing calendar would leak information (a company filing early/late is itself a signal). This is the PIT-correct clock |
| `days_to_opex` | Calendar days to the next monthly options expiry (third Friday) — for timing option legs |

## Control flags

Binary columns marking data quality/edge cases. **Don't drop them** —
they tell you when a feature means something different than usual.

| flag | set when | implication |
|---|---|---|
| `snapshot_track_used` | Row predates the ticker's first guidance event | `fundamental_surprise`/`kl_surprise_bits` came from the analyst snapshot, not guidance |
| `guidance_absent` | Ticker has no usable guidance history (~14% of universe) | Guidance-derived features are NaN — normal, not a bug |
| `options_thin_chain` | Fewer than 10 usable contracts that day | Options features are unreliable/absent for this name |
| `spread_filtered` | Spread proxy (high-low range) filtered the observation | Options data quality caveat — see DEVIATIONS.md |
| `half_life_imputed` | Too few surprise events to fit decay | `measured_half_life` = 21-day prior, not measured |
| `staleness_quarterly` | Latest knowable actuals are >100 days old | Fundamentals are stale — usual for pre-first-filing windows |
| `regime_calib_error_quantile` | Rank (0–1) of our solved ATM IV vs. Bloomberg's aggregate IV | High = our contract-level IV disagrees with the aggregate — thin or unusual chain |

## The PCA panel

`ds.get("state_vector_pca", ...)` — the 27 features compressed to **22
principal components** (97.6% of variance), fit on the training window
only (2017-01-01 → holdout cutoff). Components are orthogonalized, so
they're cleaner model inputs than correlated raw features — but they
lose the finance interpretation. A common workflow: explore/select on
raw features, compress to PCs for the final model.

## Nulls are information

Missingness in this dataset is real, not sloppy: `skew_25d`/`rn_kurtosis`
are ~52% null because roughly half of ticker-days lack a liquid options
chain; guidance features are null where companies don't guide; options
panels start 2014-06. The flags tell you *which* kind of null you're
looking at — check them before imputing.
