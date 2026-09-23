"""svq — contestant CLI for the state-vector dataset.

    svq panels                 list panels + row counts
    svq head <panel> [-n N]    preview rows
    svq tickers [--all]        universe roots (US-only default)
    svq asof <TICKER> <DATE>   PIT-safe fundamentals at a date
    svq calendar <TICKER>      report calendar for a ticker
    svq serve [--port 8000]    starter HTTP API (FastAPI)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date


def _ds(args):
    from .dataset import Dataset

    root = args.root or os.environ.get("SV_DATA_ROOT") or "."
    return Dataset(root)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="svq", description=__doc__)
    p.add_argument("--root", help="dataset root (default: $SV_DATA_ROOT or .)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("panels", help="list panels + row counts")

    h = sub.add_parser("head", help="preview a panel")
    h.add_argument("panel")
    h.add_argument("-n", type=int, default=8)
    h.add_argument("--ticker")

    t = sub.add_parser("tickers", help="universe roots")
    t.add_argument("--all", action="store_true", help="include non-US")

    a = sub.add_parser("asof", help="PIT-safe fundamentals at a date")
    a.add_argument("ticker")
    a.add_argument("date", help="YYYY-MM-DD")

    c = sub.add_parser("calendar", help="report calendar for a ticker")
    c.add_argument("ticker")

    s = sub.add_parser("serve", help="starter HTTP API")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")

    sub.add_parser("doctor", help="sanity-check the dataset install")

    args = p.parse_args(argv)

    if args.cmd == "panels":
        ds = _ds(args)
        for name, n in ds.panels().items():
            print(f"{name:38s} {n:>12,}")
        return 0

    if args.cmd == "head":
        ds = _ds(args)
        lf = ds.panel(args.panel)
        if args.ticker:
            col = "ticker" if "ticker" in lf.collect_schema().names() else "ticker_bbg"
            lf = lf.filter(__import__("polars").col(col) == args.ticker)
        print(lf.head(args.n).collect())
        return 0

    if args.cmd == "tickers":
        for t in _ds(args).universe(us_only=not args.all):
            print(t)
        return 0

    if args.cmd == "asof":
        from .pit import asof_fundamentals

        d = date.fromisoformat(args.date)
        print(asof_fundamentals(_ds(args), args.ticker, d))
        return 0

    if args.cmd == "calendar":
        from .pit import report_calendar

        cal = report_calendar(_ds(args))
        print(cal[cal["ticker"] == args.ticker].sort_values("filing_date"))
        return 0

    if args.cmd == "serve":
        try:
            import uvicorn
        except ImportError:
            print("serve needs the 'serve' extra: pip install statevector[serve]", file=sys.stderr)
            return 1
        os.environ.setdefault("SV_DATA_ROOT", args.root or os.environ.get("SV_DATA_ROOT", "."))
        uvicorn.run("statevector.serve:app", host=args.host, port=args.port)
        return 0

    if args.cmd == "doctor":
        ok = True
        try:
            ds = _ds(args)
            print(f"data root : {ds.root}")
        except Exception as e:
            print(f"FAIL root: {e}")
            return 1
        counts = ds.panels()
        expected = {
            "fundamentals_actuals": 473_956,
            "fundamentals_estimates_hist": 496_657,
            "fundamentals_estimates_recent": 77_322,
            "valuation_weekly": 1_000_875,
            "options_agg_daily": 3_202_800,
        }
        for name, want in expected.items():
            got = counts.get(name)
            mark = "ok " if got == want else "FAIL"
            if got != want:
                ok = False
            print(f"{mark} {name:38s} {got or 0:>12,} (expected {want:,})")
        for name in sorted(set(counts) - set(expected)):
            print(f"ok  {name:38s} {counts[name]:>12,}")
        n = len(ds.universe())
        print(f"{'ok ' if n > 0 else 'FAIL'} universe: {n:,} US roots")
        return 0 if ok else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
