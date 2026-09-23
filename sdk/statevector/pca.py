"""PCA reduction of the state vector: 27 numeric -> ~22 components.

Fit on the TRAINING window only (2017-01-01 .. holdout cutoff) to keep
the holdout sealed. Standardize with training mean/std, fit PCA on
training rows, transform the full panel.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .vector import NUMERIC


def holdout_cutoff(data_end: pd.Timestamp | None = None,
                   days: int = 30) -> pd.Timestamp:
    """Trailing 30-calendar-day sealed holdout boundary — anchored to
    the DATA's last date, not wall-clock, so the holdout is stable."""
    a = data_end or pd.Timestamp.today().normalize()
    return a - pd.Timedelta(days=days)


def fit_pca(vec: pd.DataFrame, n_components: int = 22,
            train_start: str = "2017-01-01",
            holdout_days: int = 30) -> tuple[pd.DataFrame, dict]:
    """Returns (transformed df, fit meta). vec must have
    ticker,date + NUMERIC columns."""
    vec = vec.copy()
    vec["date"] = pd.to_datetime(vec["date"]).dt.date
    cutoff = holdout_cutoff(
        pd.Timestamp(max(vec["date"])), days=holdout_days).date()
    train = vec[(vec["date"] >= pd.Timestamp(train_start).date())
                & (vec["date"] < cutoff)]
    X = train[NUMERIC].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    if len(X) < 100:
        raise ValueError(f"too few complete training rows: {len(X)}")

    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    # economy SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    W = Vt[:n_components].T            # 27 x 22 loadings
    explained = (S**2 / max(len(Z) - 1, 1))
    ratio = explained / explained.sum()

    scores = transform_pca(vec, mu, sd, W)
    out = vec[["ticker", "date"]].copy()
    for i in range(n_components):
        out[f"pc{i + 1:02d}"] = scores[:, i]
    meta = {
        "n_components": n_components,
        "features": list(NUMERIC),
        "train_start": train_start,
        "holdout_days": holdout_days,
        "train_rows": int(len(X)),
        "data_end": str(max(vec["date"])),
        "cutoff": str(cutoff),
        "imputation": "training column mean (post-standardize 0)",
        "explained_ratio": ratio[:n_components].tolist(),
        "cum_explained": float(ratio[:n_components].sum()),
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "loadings": W.tolist(),
    }
    return out, meta


def transform_pca(vec: pd.DataFrame, mu: np.ndarray, sd: np.ndarray,
                  W: np.ndarray) -> np.ndarray:
    """Apply a frozen fit to arbitrary rows — impute missing with the
    training mean (0 after standardization), project onto loadings."""
    Xa = vec[NUMERIC].to_numpy(float)
    Za = (Xa - mu) / sd
    Za[~np.isfinite(Za)] = 0.0
    return Za @ W


def save_fit(meta: dict, path) -> None:
    import json
    Path(path).write_text(json.dumps(meta, indent=2))


def load_fit(path) -> dict:
    import json
    return json.loads(Path(path).read_text())
