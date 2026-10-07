"""news + short_volume accessors — isolated-root tests (no shared
fixture writes; the session-scoped fixture root is off-limits)."""

from pathlib import Path

import polars as pl
import pytest

from statevector import Dataset


def _ds(tmp_path, tables):
    data = Path(tmp_path) / "data" / "structural"
    data.mkdir(parents=True)
    for name, df in tables.items():
        df.write_parquet(data / f"{name}.parquet")
    return Dataset(tmp_path)


NEWS = pl.DataFrame({
    "id": ["a1", "a1", "a2", "a3"],
    "published_utc": pl.Series(["2025-01-10T14:30:00Z",
                                "2025-01-10T14:30:00Z",
                                "2025-03-05T09:00:00Z",
                                "2025-06-01T20:00:00Z"]).str.to_datetime(
                                    "%Y-%m-%dT%H:%M:%SZ"),
    "ticker": ["AAPL", "MSFT", "AAPL", "AAPL"],
    "sentiment": ["positive", "negative", "neutral", None],
    "sentiment_reasoning": ["beat", "miss", "note", None],
    "title": ["t1", "t1", "t2", "t3"],
    "description": ["d1", "d1", "d2", "d3"],
    "publisher": ["pub", "pub", "pub", "pub"],
    "article_url": ["u1", "u1", "u2", "u3"],
})

SHORTVOL = pl.DataFrame({
    "date": ["2024-02-06", "2024-02-06", "2024-02-07"],
    "ticker": ["AAPL", "MSFT", "AAPL"],
    "total_volume": [100.0, 50.0, 120.0],
    "short_volume": [35.0, 10.0, 48.0],
    "exempt_volume": [1.0, 0.5, 1.2],
    "non_exempt_volume": [34.0, 9.5, 46.8],
    "short_volume_ratio": [35.0, 20.0, 40.0],
})


def test_news_accessor(tmp_path):
    ds = _ds(tmp_path, {"news": NEWS})
    out = ds.news()
    assert out.shape == (4, 9)
    # per-(article, ticker) explosion is the row unit
    assert out.filter(pl.col("id") == "a1")["ticker"].to_list() \
        == ["AAPL", "MSFT"]


def test_news_filters(tmp_path):
    ds = _ds(tmp_path, {"news": NEWS})
    out = ds.news("AAPL", start="2025-01-01", end="2025-01-31")
    assert out.shape[0] == 1
    assert out["published_utc"][0].month == 1
    # ticker str + list equivalence
    assert ds.news(["AAPL"]).shape[0] == ds.news("AAPL").shape[0]


def test_news_absent_panel(tmp_path):
    ds = _ds(tmp_path, {})
    out = ds.news()
    assert out.shape == (0, 9)
    assert out.schema["published_utc"] == pl.Datetime


def test_short_volume_accessor(tmp_path):
    ds = _ds(tmp_path, {"short_volume": SHORTVOL})
    out = ds.short_volume("AAPL")
    assert out.shape[0] == 2
    out2 = ds.short_volume(start="2024-02-07", end="2024-02-07")
    assert out2.shape[0] == 1
    assert out2["date"][0] == "2024-02-07"


def test_short_volume_absent_panel(tmp_path):
    ds = _ds(tmp_path, {})
    out = ds.short_volume()
    assert out.shape == (0, 7)
