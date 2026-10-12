"""futures_macro accessor tests (synthetic root — public repo)."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset  # noqa: E402


def _mkroot(tmp_path: Path) -> Path:
    (tmp_path / "data" / "structural").mkdir(parents=True)
    return tmp_path


def _mkmacro(root: Path) -> pl.DataFrame:
    df = pl.DataFrame([
        {"session_end_date": "2026-10-09",
         "conditions__knowable": "2026-09-23",
         "conditions__period": "2026-09-22",
         "conditions__effr": 3.88,
         "yields__ust10y": 5.01,
         "yields__ust2y": 4.76,
         "curve_2s10": 0.25,
         "inflation__cpi_yoy": 2.31,
         "inflation_age_days": 34,
         "market__unemp": 4.1},
        {"session_end_date": "2026-10-12",
         "conditions__knowable": "2026-09-23",
         "conditions__period": "2026-09-22",
         "conditions__effr": 3.88,
         "yields__ust10y": 5.00,
         "yields__ust2y": 4.75,
         "curve_2s10": 0.25,
         "inflation__cpi_yoy": 2.31,
         "inflation_age_days": 37,
         "market__unemp": 4.1},
    ])
    df.write_parquet(
        root / "data" / "structural" / "futures_macro.parquet")
    return df


def test_macro_empty_when_absent(tmp_path):
    ds = Dataset(str(_mkroot(tmp_path)))
    m = ds.futures_macro()
    assert m.height == 0
    assert "session_end_date" in m.columns


def test_macro_reads_panel(tmp_path):
    root = _mkroot(tmp_path)
    _mkmacro(root)
    m = Dataset(str(root)).futures_macro()
    assert m.height == 2
    assert "conditions__effr" in m.columns
    assert "curve_2s10" in m.columns


def test_macro_date_filters(tmp_path):
    root = _mkroot(tmp_path)
    _mkmacro(root)
    ds = Dataset(str(root))
    assert ds.futures_macro(start="2026-10-10").height == 1
    assert ds.futures_macro(end="2026-10-09").height == 1
    assert ds.futures_macro(start="2026-10-09",
                            end="2026-10-09").height == 1
