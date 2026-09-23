# SPEC_COMPLIANCE.md — v2.0 spec cross-reference (updated 2026-09-22 late)

## Global gates

- 0 Timestamp units: N/A-by-design — all sources normalized to
  Date/Datetime at parse; no mixed ms/µs/ns feeds.
- 1 Cross-feed equality (stock vs option minute bars): runnable —
  stocks_minute complete; gate not yet implemented.
- 2 PIT joins: EDGAR filing dates real; orphan gate pending.
- 3 Options overlap: DEVIATION — single premium source (day-agg close),
  no NBBO (quotes ~100GB/day infeasible).
- 4 Mkt-cap tripwire (G-18): **DONE** — 241/763,984 ticker-weeks
  anomalous (0.03% < 0.5% limit), split-aware via corporate_actions.
- 5 IC significance / 6 holdout: holdout seal in SDK; IC downstream.

## Phases

- Phase 0: manifest done; membership stub; **corporate_actions
  MATERIALIZED** (1.86M rows: 1.84M dividends + 15.7k splits, schema
  ticker/ex_date/kind/value/raw); GUIDANCE DONE — /benzinga/v1/guidance
  124,013 rows, 2014+, 86% of universe (1,086/1,258) > 80% → negentropy
  PRIMARY; guidance_absent only for residual 14%.
- Phase 1: daily bars done; **MINUTE BARS COMPLETE** (stocks_minute
  3,198 files, 12GB, 2014→, universe-filtered); trades/quotes deviation;
  benchmarks done (2016-09+).
- Phase 2: fundamentals done; valuation raw done; report calendar done
  (EDGAR); guidance done (PIT last_updated ns).
- Phase 3: OCC parse done; premium loader = day-agg close (deviation;
  options_daily COMPLETE 3,098 days); Greeks DONE — BS2002 vectorized
  (Newton+bisection IV solve), options_greeks_daily panel building;
  rates/macro done (5 panels).
- Phase 4/16/17: vector.py assembles all 27 numeric + 2 clocks + 7 flags
  (PIT actuals via filing dates, TTM-surprise, same-period guidance
  velocity, projected report clock); pca.py fits 22 components on
  2017-01-01..data_end-30d (holdout sealed, stats persisted);
  validate.py runs embargoed walk-forward ridge weights + additive vs
  route-energy rank-IC selection.

## DQ

**24 gates, 0 hard failures.** G-02 OHLC (0 violations / 3,198 files),
G-18 tripwire (0.03%), G-24 put-call parity (0.1% violation over 8.56M
matched C/P pairs on the built greeks panel). Leakage gates G-50/G-51
implemented in tools/dq_leakage.py (audit-sidecar asof check, report-clock
projection check, PCA fit-window recompute). Remaining: G-05/G-06 minute
census + minute↔daily reconciliation.

## Pulls

Done: stocks daily, **stocks minute (12GB)**, options daily, dividends
(1.84M), splits, filings 10-K/10-Q, reference tickers, holidays,
guidance, macro. Running: crypto backfill (daily→hourly→minute majors,
sv-crypto-backfill unit).

## New since last update

`crypto/` subtree: Massive crypto aggregates → Orkid reference service
(tailnet :8378), separate product track — see crypto/README.md +
INTEGRATION.md.
