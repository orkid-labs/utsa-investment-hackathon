"""Contestant-facing accessors — everything returns pandas DataFrames.

Polars does the heavy lifting internally (lazy parquet scans), but you
never have to touch it. If you want lazy queries, pass ``lazy=True``.

    from statevector import Dataset
    ds = Dataset("/path/to/state-vector")      # or set SV_DATA_ROOT

    ds.get("stocks_daily", ticker="AAPL")       # pandas DataFrame
    ds.prices("NVDA", start="2024-01-01")
    ds.universe()                                # list of tickers
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import date as _date
from pathlib import Path

import polars as pl


def _as_date(v):
    """Accept 'YYYY-MM-DD' or date; return a polars-comparable date."""
    if v is None or isinstance(v, _date):
        return v
    return _date.fromisoformat(str(v)[:10])

# canonical single-file panels
CANONICAL_PANELS = {
    "fundamentals_actuals",
    "fundamentals_estimates_hist",
    "fundamentals_estimates_recent",
    "valuation_weekly",
    "options_agg_daily",
    "index_daily",
    "macro_treasury_yields",
    "macro_labor_market",
    "macro_inflation",
    "macro_inflation_expectations",
    "macro_funding_conditions",
}

# partitioned directories (one parquet per trading day — large!)
PARTITIONED_PANELS = {
    "stocks_daily", "options_daily", "stocks_minute", "crypto_daily",
}

# per-ticker nested dirs: {panel}/{X_BTCUSD}/START_END.parquet
NESTED_PANELS = {"crypto_hourly", "crypto_minute"}

# Massive aggs ship single-letter fields; normalize for crypto panels
CRYPTO_RENAME = {
    "T": "ticker", "t": "ts", "o": "open", "h": "high", "l": "low",
    "c": "close", "v": "volume", "vw": "vwap", "n": "trades",
}

# single-file raw panels under data/raw/massive/
RAW_PANELS = {
    "guidance_all",
    "market_holidays",
    "reference_tickers",
    "splits_all",
    "dividends_all",
    "filings_10k",
    "filings_10q",
}

STRUCTURAL_TABLES = {
    "ticker_map",
    "universe_membership",
    "corporate_actions",
    "report_calendar_us",
}

# canonical derived panels built by tools/build_*.py (dir-partitioned)
DERIVED_PANELS = {
    "microstructure_daily", "options_smile_daily", "options_greeks_daily",
    "state_vector", "state_vector_pca",
}

# rows we will silently cap at when no filter is given
DEFAULT_LIMIT = 10_000


class Dataset:
    """Root accessor for the state-vector dataset."""

    def __init__(self, root: str | Path | None = None,
                 token: str | None = None):
        root = root or os.environ.get("SV_DATA_ROOT", ".")
        self.token = token or os.environ.get("SV_DATA_TOKEN")
        self.base: str | None = None
        self._index: dict = {}
        self.root: Path | None = None
        if str(root).startswith(("http://", "https://")):
            # thin client: panels resolve via the server's index.json and
            # are scanned lazily over HTTP range requests
            self.base = str(root).rstrip("/")
            self._index = self._fetch_json("/data/index.json")
        else:
            self.root = Path(root)
            if not (self.root / "data").exists():
                raise FileNotFoundError(
                    f"{self.root}/data not found — point Dataset() at the "
                    "repo root or set SV_DATA_ROOT"
                )

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _url(self, rel: str) -> str:
        """Panel file URL — token rides the query string (object_store
        preserves ?key= where it drops custom auth headers)."""
        u = f"{self.base}/data/{rel}"
        return f"{u}?key={self.token}" if self.token else u

    def _fetch(self, path: str) -> bytes:
        req = urllib.request.Request(
            f"{self.base}{path}", headers=self._headers)
        return urllib.request.urlopen(req, timeout=60).read()

    def _fetch_json(self, path: str):
        return json.loads(self._fetch(path))

    # -- core ------------------------------------------------------------------

    def _scan_remote(self, name: str) -> pl.LazyFrame:
        """Lazy scan over HTTP using the server's index.json manifest."""
        try:
            entry = self._index["panels"][name]
        except KeyError:
            raise KeyError(
                f"unknown panel {name!r}; available: "
                f"{sorted(self._index['panels'])}"
            )
        scans = []
        first_schema: dict | None = None
        for rel in entry["files"]:
            lf = pl.scan_parquet(self._url(rel))
            if first_schema is None:
                # one footer fetch — assume homogeneous schema across
                # the panel's files (true for all shipped panels)
                first_schema = dict(lf.collect_schema())
            scans.append((rel, lf))

        def fix(lf: pl.LazyFrame, rel: str) -> pl.LazyFrame:
            if "volume" in first_schema and \
                    first_schema["volume"] != pl.Float64:
                lf = lf.with_columns(pl.col("volume").cast(pl.Float64))
            if entry["kind"] == "raw_nested":
                tk = Path(rel).parent.name.replace("_", ":", 1)
                lf = lf.with_columns(pl.lit(tk).alias("ticker"))
                ren = {k: v for k, v in CRYPTO_RENAME.items()
                       if k in first_schema}
                if ren:
                    lf = lf.rename(ren)
            elif name == "crypto_daily":
                ren = {k: v for k, v in CRYPTO_RENAME.items()
                       if k in first_schema}
                if ren:
                    lf = lf.rename(ren)
            return lf

        out = [fix(lf, rel) for rel, lf in scans]
        if len(out) == 1:
            return out[0]
        return pl.concat(out, how="diagonal_relaxed")

    def _scan(self, name: str) -> pl.LazyFrame:
        if self.base is not None:
            return self._scan_remote(name)
        can = self.root / "data" / "canonical" / f"{name}.parquet"
        if can.exists():
            return pl.scan_parquet(can)
        can_dir = self.root / "data" / "canonical" / name
        if can_dir.is_dir():
            files = sorted(can_dir.glob("*.parquet"))
            if not files:
                raise KeyError(f"panel {name!r} exists but is empty")
            return pl.concat(
                [pl.scan_parquet(f) for f in files],
                how="diagonal_relaxed",
            )
        struct = self.root / "data" / "structural" / f"{name}.parquet"
        if struct.exists():
            return pl.scan_parquet(struct)
        part_file = self.root / "data" / "raw" / "massive" / f"{name}.parquet"
        if part_file.exists():
            return pl.scan_parquet(part_file)
        part = self.root / "data" / "raw" / "massive" / name
        if part.is_dir():
            if name in NESTED_PANELS:
                files = sorted(part.glob("*/*.parquet"))
            else:
                files = sorted(part.glob("*.parquet"))
            if not files:
                raise KeyError(f"panel {name!r} exists but is empty")
            scans = []
            for f in files:
                lf = pl.scan_parquet(f)
                for c, dt in lf.collect_schema().items():
                    if c == "volume" and dt != pl.Float64:
                        lf = lf.with_columns(pl.col("volume").cast(pl.Float64))
                if name in NESTED_PANELS:
                    # range-agg bars carry no ticker field — take it from
                    # the directory name (X_BTCUSD -> X:BTCUSD)
                    tk = f.parent.name.replace("_", ":", 1)
                    lf = lf.with_columns(pl.lit(tk).alias("ticker"))
                    ren = {k: v for k, v in CRYPTO_RENAME.items()
                           if k in lf.collect_schema().names()}
                    lf = lf.rename(ren)
                elif name == "crypto_daily":
                    ren = {k: v for k, v in CRYPTO_RENAME.items()
                           if k in lf.collect_schema().names()}
                    lf = lf.rename(ren)
                scans.append(lf)
            return pl.concat(scans, how="diagonal_relaxed")
        raise KeyError(
            f"unknown panel {name!r}; available: "
            f"{sorted(CANONICAL_PANELS | PARTITIONED_PANELS | RAW_PANELS | NESTED_PANELS | DERIVED_PANELS | STRUCTURAL_TABLES)}"
        )

    def get(
        self,
        panel: str,
        ticker: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        lazy: bool = False,
    ):
        """Get a panel as a pandas DataFrame.

        Args:
            panel:  panel name (see ``ds.panels()``)
            ticker: filter to one ticker (matches `ticker` or `ticker_bbg`)
            start/end: inclusive date range, "YYYY-MM-DD"
            limit:  row cap (default {DEFAULT_LIMIT} on huge panels)
            lazy:   return a polars LazyFrame instead of pandas
        """
        lf = self._scan(panel)
        cols = lf.collect_schema().names()
        tcol = "ticker" if "ticker" in cols else ("ticker_bbg" if "ticker_bbg" in cols else None)
        if ticker:
            if tcol is None:
                raise KeyError(f"panel {panel} has no ticker column")
            lf = lf.filter(pl.col(tcol) == ticker)
        if start and "date" in cols:
            lf = lf.filter(pl.col("date") >= _as_date(start))
        if end and "date" in cols:
            lf = lf.filter(pl.col("date") <= _as_date(end))

        cap = limit
        if cap is None and (panel in PARTITIONED_PANELS and not ticker):
            cap = DEFAULT_LIMIT
        if cap is not None:
            lf = lf.limit(cap)

        if lazy:
            return lf
        return lf.collect().to_pandas()

    # public alias — LazyFrame access for power users / internal modules
    def panel(self, name: str) -> pl.LazyFrame:
        return self._scan(name)

    def crypto(
        self,
        pair: str,
        timespan: str = "day",
        start: str | int | None = None,
        end: str | int | None = None,
        lazy: bool = False,
    ):
        """Crypto bars for a pair. timespan: day|hour|minute.

        ``pair`` accepts "BTCUSD" or "X:BTCUSD". ``start``/``end`` may be
        "YYYY-MM-DD" dates or ms-epoch ints (bars are keyed by ``ts`` ms).
        Reference aggregates only — coverage is ~400 pairs/day, majors
        reliable; long-tail pairs may be absent.
        """
        panel = {"day": "crypto_daily", "hour": "crypto_hourly",
                 "minute": "crypto_minute"}.get(timespan)
        if panel is None:
            raise ValueError("timespan must be day|hour|minute")
        p = pair.upper()
        if not p.startswith("X:"):
            p = f"X:{p}"
        lf = self._scan(panel).filter(pl.col("ticker") == p)

        def to_ms(x):
            if isinstance(x, str):
                dt = __import__("datetime").datetime.fromisoformat(x)
                return int(dt.timestamp() * 1000)
            return int(x)

        if start is not None:
            lf = lf.filter(pl.col("ts") >= to_ms(start))
        if end is not None:
            lf = lf.filter(pl.col("ts") <= to_ms(end))
        if lazy:
            return lf
        return lf.sort("ts").collect().to_pandas()

    def panels(self) -> dict[str, int]:
        """Panel name -> row count."""
        out: dict[str, int] = {}
        for name in sorted(
            CANONICAL_PANELS | PARTITIONED_PANELS | RAW_PANELS | NESTED_PANELS | DERIVED_PANELS | STRUCTURAL_TABLES
        ):
            try:
                out[name] = self._scan(name).select(pl.len()).collect().item()
            except Exception:
                continue
        return out

    # -- universe ----------------------------------------------------------------

    def universe(self, us_only: bool = True) -> list[str]:
        """Effective universe roots.

        Default: the ~1,258 US names with options+valuation coverage
        (structural/universe_us.txt). Pass ``us_only=False`` for every
        mapped ticker, or ``us_only="fundamentals"`` for all 2,553 US names.
        """
        if us_only is True:
            if self.base is not None:
                txt = self._fetch("/data/structural/universe_us.txt")
                return [t.strip() for t in txt.decode().splitlines() if t.strip()]
            uni = self.root / "data/structural/universe_us.txt"
            if uni.exists():
                return [
                    t.strip() for t in uni.read_text().splitlines() if t.strip()
                ]
        lf = self._scan("ticker_map")
        if us_only:
            lf = lf.filter(pl.col("is_us"))
        return lf.select("ticker").collect().to_series().drop_nulls().to_list()

    def structural(self, name: str):
        """Structural table as pandas DataFrame."""
        if self.base is not None:
            return self._scan(name).collect().to_pandas()
        p = self.root / "data/structural" / f"{name}.parquet"
        if not p.exists():
            raise KeyError(f"unknown structural table {name!r}")
        return pl.read_parquet(p).to_pandas()

    # -- friendly domain accessors -------------------------------------------------

    def prices(self, ticker: str, start=None, end=None):
        """Daily OHLCV + 1-day return for a ticker."""
        lf = self._scan("stocks_daily").filter(pl.col("ticker") == ticker)
        if start:
            lf = lf.filter(pl.col("date") >= _as_date(start))
        if end:
            lf = lf.filter(pl.col("date") <= _as_date(end))
        lf = lf.sort("date").with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_1d")
        )
        return lf.collect().to_pandas()

    def options_chain(self, ticker: str, on=None, limit: int = 5_000):
        """Contract-level day bars for one underlying.

        Rows include the OCC contract ticker (``O:AAPL260923C00245000``),
        OHLC, volume, transactions, date. Parse the contract with
        ``parse_occ()``.
        """
        lf = self._scan("options_daily").filter(
            pl.col("ticker").str.extract(r"^O:([A-Z]+)", 1) == ticker
        )
        if on:
            lf = lf.filter(pl.col("date") == _as_date(on))
        return lf.limit(limit).collect().to_pandas()

    def fundamentals(self, ticker: str, asof: str | None = None):
        """Quarterly fundamentals for a ticker.

        Pass ``asof="YYYY-MM-DD"`` to restrict to filings that existed by
        that date (point-in-time safe; needs report_calendar_us).
        """
        from .pit import asof_fundamentals

        if asof:
            return asof_fundamentals(self, ticker, asof)
        return self.get("fundamentals_actuals", ticker=ticker)

    def benchmark(self, index: str = "SPX"):
        """Index OHLCV history — "SPX", "MID", or "SML"."""
        return (
            self._scan("index_daily")
            .filter(pl.col("index") == index)
            .sort("date")
            .collect()
            .to_pandas()
        )

    def adv(self, ticker: str, days: int = 20):
        """Prices + rolling average dollar volume (close*volume)."""
        lf = self._scan("stocks_daily").filter(pl.col("ticker") == ticker)
        lf = lf.sort("date").with_columns(
            (pl.col("close") * pl.col("volume")).rolling_mean(days).alias(f"adv_{days}")
        )
        return lf.collect().to_pandas()

    def trading_days(self, start=None, end=None) -> list:
        """All trading dates present in stocks_daily."""
        lf = self._scan("stocks_daily").select("date").unique().sort("date")
        if start:
            lf = lf.filter(pl.col("date") >= _as_date(start))
        if end:
            lf = lf.filter(pl.col("date") <= _as_date(end))
        return lf.collect().to_series().to_list()

    def sectors(self):
        """ticker -> company name, industry, exchange, market cap."""
        lf = self._scan("reference_tickers").select(
            ["ticker", "name", "sic_description", "primary_exchange", "market_cap"]
        )
        return lf.collect().to_pandas()

    def guidance(self, ticker: str | None = None):
        """Structured company guidance events (Benzinga).

        Columns include date, time, fiscal_year/period, min/max/estimated
        EPS + revenue guidance, eps_method/revenue_method, release_type,
        positioning, importance, and last_updated (ns — PIT-clean).
        Coverage: 2014→, ~86% of the effective universe.
        """
        df = self.get("guidance_all", ticker=ticker, lazy=True)
        return df.collect().to_pandas()

    def calendar(self, ticker: str | None = None):
        """SEC filing calendar — the PIT report dates."""
        from .pit import report_calendar

        return report_calendar(self, ticker)

    def options_greeks(self, ticker: str, on=None, limit: int = 5_000):
        """BS2002 Greeks per contract-day for one underlying.

        Columns: contract (OCC), spot, strike, expiry, cp, T, close,
        volume, iv, delta, gamma, vega, theta, rho."""
        lf = self._scan("options_greeks_daily").filter(
            pl.col("root") == ticker)
        if on:
            lf = lf.filter(pl.col("date") == _as_date(on))
        return lf.limit(limit).collect().to_pandas()

    def smile(self, ticker: str | None = None, start=None, end=None):
        """Daily smile features per ticker: atm_iv_30d/90d, skew_25d,
        rn_kurtosis, term_slope, n_contracts."""
        return self.get("options_smile_daily", ticker=ticker,
                        start=start, end=end, limit=DEFAULT_LIMIT)

    def microstructure(self, ticker: str | None = None,
                       start=None, end=None):
        """Daily microstructure stats from minute bars: realized_vol,
        amihud_illiq, kyle_lambda, fdt_deviation, dollar_vol, vwap."""
        return self.get("microstructure_daily", ticker=ticker,
                        start=start, end=end, limit=DEFAULT_LIMIT)

    def state_vector(self, ticker: str | None = None, start=None,
                     end=None, pca: bool = False, lazy: bool = False):
        """The assembled 27-feature state vector (+clocks+flags).
        pca=True returns the PCA-scored variant (pc01..pc22 appended,
        produced by tools/run_validation.py)."""
        panel = "state_vector_pca" if pca else "state_vector"
        return self.get(panel, ticker=ticker, start=start,
                        end=end, lazy=lazy)

    def asof(self, ticker: str, on: str):
        """PIT-safe fundamentals visible at `on` — alias for
        fundamentals(ticker, asof=on)."""
        return self.fundamentals(ticker, asof=on)

    def holdout_cutoff(self, holdout_days: int = 30):
        """First date inside the sealed trailing holdout, anchored to
        the dataset's last trading day."""
        from .pit import holdout_cutoff as _hc
        end = max(self.trading_days())
        return _hc(holdout_days=holdout_days, end=end)

    def holidays(self):
        """US market holiday calendar (rows: date, name, status)."""
        return self._scan("market_holidays").collect().to_pandas()

    # -- reports -----------------------------------------------------------------

    def report(self, name: str) -> str:
        """Generated report text — "census" or "dq_report"."""
        if self.base is not None:
            return self._fetch(f"/data/reports/{name}.md").decode()
        return (self.root / "data/reports" / f"{name}.md").read_text()


def parse_occ(symbol: str) -> dict:
    """Parse an OCC contract ticker like O:AAPL260923C00245000.

    Returns dict(underlying, expiry, side, strike).
    """
    if not symbol.startswith("O:"):
        raise ValueError(f"not an OCC ticker: {symbol!r}")
    body = symbol[2:]
    i = 0
    while i < len(body) and body[i].isalpha():
        i += 1
    underlying, rest = body[:i], body[i:]
    expiry = f"20{rest[:2]}-{rest[2:4]}-{rest[4:6]}"
    return {
        "underlying": underlying,
        "expiry": expiry,
        "side": "call" if rest[6] == "C" else "put",
        "strike": int(rest[7:]) / 1000.0,
    }
