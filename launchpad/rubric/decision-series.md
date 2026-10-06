# Decision-series submission format

Status: implemented — validate_decision_series / run_backtest_series in
sdk/statevector/backtest.py; opt-in scoring via launchpad/rubric/decisions.yaml.
Not part of the v1 rubric contract.

## Motivation

`POST /backtest` accepts one fixed `ticker`/`weight` vector, optionally
rebalanced on a calendar grid (`none | daily | weekly | monthly`). It cannot
express a model that changes its holdings *when information changes* —
adaptive screens, regime policies, RL agents, anything whose book is a
function of what it has seen so far.

The proposed format replaces the single static book with a **chronological
series of target portfolios**, each carrying three timestamps:

- `information_cutoff` — the newest data the model was allowed to see when it
  made this call. This is the PIT boundary.
- `decision_time` — when the model emitted the decision.
- `execution_time` — when the target book is assumed to trade.

Judges generate this series by **replaying the frozen model through the
holdout**, revealing information one step at a time. The team ships a model;
the judge walks it forward. Nothing after `information_cutoff` may influence
`target_holdings` — that is the entire anti-lookahead property, made explicit
per decision instead of once per backtest.

## The record

One complete target portfolio per decision. Example — seven names, every
weight ≤ 20%:

```json
{
  "schema_version": "1.0",
  "team_id": "team-alpha",
  "model_id": "ppo-v1",
  "decision_id": 1,
  "information_cutoff": "2026-09-01T16:00:00-04:00",
  "decision_time": "2026-09-01T16:15:00-04:00",
  "execution_time": "2026-09-02T09:30:00-04:00",
  "action": "rebalance",
  "target_holdings": [
    {"ticker": "AAPL", "weight": 0.20},
    {"ticker": "MSFT", "weight": 0.18},
    {"ticker": "JPM",  "weight": 0.15},
    {"ticker": "XOM",  "weight": 0.14},
    {"ticker": "JNJ",  "weight": 0.13},
    {"ticker": "PG",   "weight": 0.10},
    {"ticker": "CAT",  "weight": 0.10}
  ]
}
```

## Schema

| field | rule |
|---|---|
| `schema_version` | `"1.0"` |
| `team_id` / `model_id` | attribution; one model per submission |
| `decision_id` | strictly increasing integers, starting at 1 |
| `information_cutoff` | PIT boundary — model must not observe data after this |
| `decision_time` | ≥ `information_cutoff` |
| `execution_time` | > `decision_time` (no instant fills) |
| `action` | `rebalance` — target replaces book. `hold` allowed: empty `target_holdings` = no change |
| `target_holdings` | complete desired book; Σ|w| = 1 (long-only), each |w| ≤ 20%, O: option legs allowed per mandate |

`CASHHOLDING` pseudo-ticker is valid inside `target_holdings` — an RL policy
must be able to de-risk.

## Replay semantics (judge side)

1. Feed the model observations up to `information_cutoff` only.
2. Collect the emitted `target_holdings`; validate the record.
3. At `execution_time`, transition the book: turnover = Σ|w_new − w_old|
   charged at mandate costs (10 bps stocks / 50 bps O: legs), then hold
   mark-to-market until the next decision's `execution_time`.
4. Between decisions the book **drifts with prices** — a target is a
   transition order at `execution_time`, not a frozen weight curve.
5. Corporate actions apply to open legs exactly as in the v1 engine
   (`apply_split_factors` semantics).
6. The final score is the series' net equity curve — computed entirely by the
   judge. Reported metrics in the submission are advisory only, same as v1.

Backward-compatible: a v1 static book is a one-decision series, and
`rebalance_frequency = daily|weekly|monthly` is a series with a fixed
calendar — the format is a strict superset.

## Endpoint

```
POST /decisions
{ "series": [ <record>, <record>, ... ] }
```

The judge replays the series through the holdout window and returns the
recomputed curve + metrics. Whether the series is submitted as a static
artifact (post-hoc replay log) or generated live by `check.py` driving the
frozen model is an open implementation choice — the record format is
identical either way.
