# DEVIATIONS.md — spec vs. achievable from available data

Frozen config v1.1. Sources: Bloomberg drop 2026-09-20 + Massive (flat files + REST).

## Material deviations

1. **Estimates vintage leakage (accepted).** `bdh_*_estimates.parquet` are a single
   snapshot dated 2026-08-28. Historical-consensus-vs-actual spread (~17-22%/yr)
   suggests they behave like period consensus, but they are NOT point-in-time
   vintages. `snapshot_track_used=1` flag set on all estimate-derived features.

2. **No announcement/report dates in the Bloomberg drop.** `date` fields are
   fiscal period-ends. US report calendar is rebuilt from Massive
   `/v3/reference/financials` (`filing_date`, `acceptance_datetime`) ->
   `structural/report_calendar_us.parquet`. Non-US names have no PIT report gate
   -> excluded from the effective universe.

3. **Bloomberg options file is aggregate per ticker-day** (IV + call/put OI +
   volume, 1275 names, 2016-09+). Contract-level smile comes from Massive OPRA
   `day_aggs_v1` (2014+, ~1.6MB/day). Bloomberg aggregate retained as
   fallback/cross-check (`options_agg_daily.parquet`).

4. **Options quotes (NBBO spreads) infeasible.** OPRA quotes_v1 exists only
   2022+ and is ~100+GB/day. Spread-z uses per-contract high-low range proxy;
   `spread_filtered` flag semantics become "range-proxy based".

5. **Greeks are computed, not sourced.** Spec's Bjerksund-Stensland pricer at
   Build 12 (Massive MCP also exposes bs_* helpers for spot checks).

6. **Analyst dispersion absent** from the drop unless separately pulled.

7. **Universe is survivorship-marked.** Valuation/options panels are balanced
   (785 and 2512 rows x 1275 tickers exactly) — fixed-universe bias flagged in
   census; treated as known limitation for v1.1.

8. ~~Guidance events unavailable~~ **RESOLVED 2026-09-22.** Massive
   `/benzinga/v1/guidance` provides structured EPS/revenue guidance with
   `last_updated` (ns, PIT-clean), fiscal period/year, min/max/estimated,
   method (gaap/adj), release_type, positioning. Coverage: 124,013 rows,
   2014->present, 86% of the effective universe (1086/1258) — clears the
   Build 0.6 >=80% threshold for running negentropy features primary.
   `guidance_absent` now applies only to the residual 14%.

9. **Index history starts 2016-09** (SPX/MID/SML xlsx) though storage window is
   2014-01-01 — index features are NaN-gated pre-2016-09.

10. **Sentinel row.** `data/structural/sentinel.parquet` contains a synthetic
    ticker (`ORKD`). It is intentional; do not "clean" it.

## Achievable per spec

- ~600-700-name US universe after coverage filters (options+valuation panel)
- Fundamentals/valuation/microstructure/options-agg/macro layers
- Dual additive + route-energy formulations (downstream of this crate)
- PIT gates: estimates-vintage caveat flagged; report calendar real for US
- Massive: stocks day_aggs 2003+, options day_aggs 2014+, fed/v1 economy
  (treasury 1962+, funding 1954+, CPI, inflation expectations, labor),
  dividends, splits, financials filing dates

## Assembly methodology notes (added 2026-09-22)

11. **`days_to_next_report` is a PIT projection, not the realized calendar.**
    Each row uses the last filing observed by that date plus the expanding
    median inter-filing gap (default 91d). Using the realized future calendar
    would leak whether/when a filing actually occurred.

12. **`fundamental_surprise` uses TTM actuals.** Benzinga guidance is annual
    revenue; BBG `sales_rev_turn` is quarterly. Surprise = (TTM sales knowable
    at the guidance event − guided mid) / |mid|. KL surprise uses
    N(mid, (width/4)²) vs the same TTM figure.

13. **Smile internals.** IVs solved once per day over the whole contract set
    (vectorized Newton + bisection fallback). Smile stats use |log K/S| ≤ 0.5
    contracts; 25Δ wings proxied by ~5% OTM strikes; `rn_kurtosis` is the
    quadratic-curvature approximation 6·c·T, not a full SVI fit. q=0 (no
    discrete dividend yield in the inversion).

14. **G-24 parity gate** uses a constant r=3% discount — the 10%-of-spot
    violation band absorbs the rate error for T<2y.
