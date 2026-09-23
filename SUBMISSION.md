# Submitting your project

Two artifacts, one score. Judges need **a running endpoint** and
**a readable repo**.

## What to hand in

| artifact | form | what it's for |
|---|---|---|
| Endpoint URL | `https://…` serving your API | the deterministic rubric (`check.py`) runs against it live |
| Repo URL | public GitHub repo | the repo audit — judges read your code |

Hand both to the judging table before the deadline announced at the
event. The endpoint only has to stay up through judging — a free-tier
deploy or a tunnel from your laptop (`ngrok`, `cloudflared`, Tailscale
Funnel) is fine.

## What the rubric runs

```bash
python launchpad/rubric/check.py --base-url <your-endpoint-url>
```

`check.py` is deterministic — same app, same data, same score — and
it's public, in this repo, so you can self-score all weekend. The 100
points break down as:

- `GET /health` → `{ok: true}` (5)
- `GET /portfolio/holdings` shape (10) + long-only weights summing to
  1 ± 0.01 with tickers in the universe (10)
- `POST /backtest` shape (15) + metrics matching the reference
  computation within tolerance (25)
- `GET /screen` shape (10)
- `GET /asof` returns only rows dated ≤ the `on` parameter — the
  point-in-time check (10)
- Responsiveness — every check inside the timeout (15)

The starter app (`launchpad/template`) already passes all of it.
Points come from what you build on top.

## What the repo audit looks for

Judges read the code behind the endpoint. They're checking:

- **It runs.** Clear setup path, honest README, no dead code paths.
- **It's yours.** Original finance logic — screen criteria, signal
  construction, portfolio rules. Copied boilerplate is fine; copied
  *thinking* is not.
- **It respects the data.** Point-in-time joins by filing/asof date,
  the sealed holdout untouched, no lookahead anywhere.
- **It uses the dataset.** The state vector, options, fundamentals —
  the more of the data's depth your logic actually touches, the
  better it reads.

## Rules recap

- Long-only; weights ≥ 0, sum ≈ 1
- Options allowed; **protective puts are the only hedge** — no short
  positions, no written options
- Options use OCC tickers: `O:AAPL250117P00220000`
- Fundamentals join by filing date, not period end
- The trailing 30 calendar days are a sealed holdout — training and
  backtests stop at the cutoff (`ds.holdout_cutoff()`)

## Secrets

Your `SV_DATA_TOKEN` is a credential. Never commit it — read it from
the environment (`os.environ["SV_DATA_TOKEN"]`) the way the template
does. A token in a public repo gets revoked and looks bad in front of
judges.

## Timeline

- Event: **October 2026**, UTSA Carlos Alvarez College of Business
- Endpoint + repo due: deadline announced on-site
- Judging: rubric run live on your endpoint + repo audit, then awards
