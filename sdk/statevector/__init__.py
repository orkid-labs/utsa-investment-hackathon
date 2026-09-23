"""Contestant SDK for the Equity Composite State Vector dataset.

Usage:
    from statevector import Dataset

    ds = Dataset()                          # reads $SV_DATA_ROOT
    ds.get("stocks_daily", ticker="AAPL")   # pandas DataFrame
    ds.prices("AAPL", start="2024-01-01")   # OHLCV + returns
    ds.fundamentals("AAPL", asof="2024-03-31")  # point-in-time safe
    ds.universe()                           # US ticker roots

CLI:  svq panels | svq head <panel> | svq doctor | svq serve
"""

from .dataset import Dataset, parse_occ
from .pit import asof_fundamentals, holdout_cutoff, holdout_mask, report_calendar

__all__ = [
    "Dataset",
    "parse_occ",
    "asof_fundamentals",
    "report_calendar",
    "holdout_mask",
    "holdout_cutoff",
]
__version__ = "0.2.0"
