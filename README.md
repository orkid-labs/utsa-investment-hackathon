# UTSA Investment Club Hackathon — October 2026

Official repository for the **University of Texas at San Antonio,
Carlos Alvarez College of Business — Investment Club October 2026
Hackathon**.

Build a portfolio-management API on a point-in-time-safe equity
dataset covering ~1,250 US tickers, 2014→present: daily and minute
prices, contract-level options with Greeks, fundamentals, analyst
estimates, corporate guidance, SEC filings, corporate actions, and a
27-feature composite **state vector** per ticker-day.

## Credits

- **Host** — UTSA College of Business Investment Club
- **Faculty sponsor** — Dr. R. Sweet, Alvarez College of Business,
  Finance Department
- **Club president** — Joshua Davidson
- **Data** — paid for by Joshua Davidson in his capacity as president
  of the Investment Club
- **Platform built by** — Jacob Cavazos, Chairman & Chief Architect,
  Orkid Labs Inc.
- **Sponsored by** — [Orkid Labs Inc.](https://www.orkidlabs.com)
  ([orkidlabs.xyz](https://orkidlabs.xyz)) and additional sponsors to
  be named

## Getting started

**Fastest path — GitHub Codespaces / Dev Containers:** open this
repository in a Codespace (or VS Code → "Reopen in Container"). The
`.devcontainer` builds Python 3.12, installs the locked dependencies
and the SDK — you just add your team token:

```bash
export SV_DATA_ROOT=https://pop-os.tail01ad.ts.net
export SV_DATA_TOKEN=<your-team-token>
```

**Local setup** works the same way:

```bash
pip install -e sdk/                          # the data SDK
pip install -r launchpad/template/requirements.txt

# hosted data — credentials are issued to your team
export SV_DATA_ROOT=https://pop-os.tail01ad.ts.net
export SV_DATA_TOKEN=<your-team-token>

svq doctor                                   # sanity-check access
cd launchpad/template && uvicorn app:app --port 8000
```

Open http://localhost:8000/docs — the starter app already scores
100/100 on the public rubric. Extend the finance logic; keep the
endpoint shapes.

## Read these first

| doc | what |
|---|---|
| `launchpad/README.md` | contestant onboarding — what you're building, setup, judging |
| `SUBMISSION.md` | what to hand in — endpoint URL + repo, and what judges run |
| `FEATURES.md` | every state-vector feature explained in finance terms |
| `launchpad/RULES.md` | the mandate — long-only, PIT rules, sealed holdout |
| `sdk/README.md` | full SDK reference |
| `DEVIATIONS.md` | documented data limitations — read before trusting a number |
| `launchpad/examples/quickstart.py` | runnable guided tour |

## The dataset

Served remotely — no download. The SDK scans parquet lazily over HTTP:

```python
from statevector import Dataset
ds = Dataset()                                # reads env vars above
ds.state_vector("AAPL")                       # 27 features + clocks + flags
ds.prices("NVDA", start="2020-01-01")         # OHLCV
ds.options_greeks("TSLA", on="2024-06-21")    # contract-level Greeks
```

## Self-score

```bash
python launchpad/rubric/check.py --base-url http://localhost:8000
```
