"""futures_news accessor tests (synthetic root — public repo)."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from statevector import Dataset  # noqa: E402


def _mkroot(tmp_path: Path) -> Path:
    (tmp_path / "data" / "structural").mkdir(parents=True)
    return tmp_path


def _mknews(root: Path):
    pl.DataFrame([
        {"product_code": "ES", "session_end_date": "2026-10-09",
         "news_ticker": "SPY", "news_n_1d": 1, "news_n_3d": 2,
         "news_n_7d": 4, "news_sent_mean_1d": 1.0,
         "news_sent_mean_7d": 0.5},
        {"product_code": "6E", "session_end_date": "2026-10-09",
         "news_ticker": "FXE", "news_n_1d": 0, "news_n_3d": 0,
         "news_n_7d": 0, "news_sent_mean_1d": None,
         "news_sent_mean_7d": None},
    ]).write_parquet(
        root / "data" / "structural" / "futures_news.parquet")


def test_news_empty_when_absent(tmp_path):
    ds = Dataset(str(_mkroot(tmp_path)))
    m = ds.futures_news()
    assert m.height == 0
    assert "session_end_date" in m.columns


def test_news_reads_panel(tmp_path):
    root = _mkroot(tmp_path)
    _mknews(root)
    m = Dataset(str(root)).futures_news()
    assert m.height == 2
    assert "news_n_1d" in m.columns


def test_news_filters(tmp_path):
    root = _mkroot(tmp_path)
    _mknews(root)
    ds = Dataset(str(root))
    assert ds.futures_news(product_code="ES").height == 1
    assert ds.futures_news(start="2026-10-10").height == 0
    # thin-coverage proxy keeps null sentiment (never zero-fill)
    fxe = ds.futures_news(product_code="6E")
    assert fxe["news_sent_mean_7d"].null_count() == 1
