# Reference Implementation — UTSA Investment Hackathon

**Private until judging closes. Do not push to the public repo mid-contest.**

This directory contains a rules-following reference app built entirely on
the canonical dataset path — the same contract contestants implement,
same `run_backtest` engine the rubric recomputes with. It exists to:

- Prove the documented data path can score 100/100 on the rubric.
- Give the internal team a correctly-built book to compare contestants
  and research models against (same judge, same terms).
- Serve as the post-contest "intended solution" reference.

## What it does differently from a naive approach

Every signal comes from documented panels rather than hand-rolled math
off the raw tape:

- `state_vector` for all features — `cornish_fisher_gap`, `log_fv_gap`,
  `fundamental_surprise/confidence`, `amihud_illiq`,
  `minute_realized_diffusion`, `variance_risk_premium`, `skew_25d`,
  `mean_reversion_speed` — cross-sectional z-composite.
- `corporate_actions` defensively: names with split events in
  `[cutoff-60d, end]` are excluded, protecting both judged P&L and
  signal integrity (`stocks_daily.close` is raw/unadjusted by design).
- `microstructure_daily.dollar_vol` for the ADV floor — a pruned fetch
  instead of rescanning daily bars.
- `days_to_next_report` blackout, `staleness_quarterly` gate,
  spot-momentum knife guard, water-filling name/sector caps.

## Known limitations (documented, not bugs)

- `days_to_next_report` is a cadence projection; early filings inside
  the projection gap can gap the book — that is the platform's second
  intentional trap.
- `ds.sectors()` fails on the current build (`reference_tickers`
  carries no `sic_description`); sector caps degrade to per-ticker
  pseudo-sectors.
- `ds.holdout_cutoff()` scans every day-file remotely (~85s); the app
  derives the cutoff from the file manifest instead.
- The dataset is a fixed snapshot; `date.today()`-anchored scans land
  past the data end.

## Endpoints

`GET /health`, `GET /asof`, `GET /screen`, `GET /portfolio/holdings`,
`POST /backtest` — per `launchpad/RULES.md`.

Score: 100/100 on the live rubric (verified post-refactor).
