"""OCC parsing + Black-Scholes-2002 (Merton, continuous dividend yield).

Options day-aggs give us premiums, not Greeks — we invert BS on the
close to get IV, then derive Greeks at the observed contract terms.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.stats import norm

OCC_RE = re.compile(r"^O:([A-Z]+)(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True)
class OccContract:
    root: str
    expiry: date
    cp: str          # "C" or "P"
    strike: float


def parse_occ(ticker: str) -> OccContract | None:
    """'O:AAPL200221P00315000' -> OccContract. None if unparseable."""
    m = OCC_RE.match(ticker)
    if not m:
        return None
    root, ymd, cp, k = m.groups()
    yy = int(ymd[:2])
    year = 2000 + yy if yy < 80 else 1900 + yy
    return OccContract(root, date(year, int(ymd[2:4]), int(ymd[4:6])),
                       cp, int(k) / 1000.0)


def occ_ticker(root: str, expiry: date, cp: str, strike: float) -> str:
    return f"O:{root}{expiry:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


# -- Black-Scholes-2002 (Merton) ----------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, q: float,
             vol: float, cp: str) -> float:
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if cp == "C" else (K - S))
    sq = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + vol * vol / 2) * T) / sq
    d2 = d1 - sq
    if cp == "C":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, q: float,
              vol: float, cp: str) -> dict[str, float]:
    """delta, gamma, vega(per 1 vol-pt), theta(per day), rho(per 1%-pt)."""
    if T <= 0 or vol <= 0:
        return {g: float("nan") for g in ("delta", "gamma", "vega", "theta", "rho")}
    sq = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + vol * vol / 2) * T) / sq
    d2 = d1 - sq
    pdf = norm.pdf(d1)
    eqt = math.exp(-q * T)
    ert = math.exp(-r * T)
    sign = 1.0 if cp == "C" else -1.0
    delta = sign * eqt * norm.cdf(sign * d1)
    gamma = eqt * pdf / (S * sq)
    vega = S * eqt * pdf * math.sqrt(T) / 100.0
    theta = (
        -(S * eqt * pdf * vol) / (2 * T)
        - sign * r * K * ert * norm.cdf(sign * d2)
        + sign * q * S * eqt * norm.cdf(sign * d1)
    ) / 365.0
    rho = sign * K * T * ert * norm.cdf(sign * d2) / 100.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                q: float, cp: str, lo: float = 1e-4, hi: float = 10.0,
                tol: float = 1e-6, iters: int = 100) -> float:
    """Bisection IV solver; NaN when no solution in [lo, hi]."""
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    intrinsic = max(0.0, (S - K * math.exp(-r * T)) if cp == "C"
                    else (K * math.exp(-r * T) - S))
    if price < intrinsic * 0.999:
        return float("nan")  # below intrinsic — untradeable quote
    flo = bs_price(S, K, T, r, q, lo, cp) - price
    fhi = bs_price(S, K, T, r, q, hi, cp) - price
    if flo > 0 or fhi < 0:
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = bs_price(S, K, T, r, q, mid, cp) - price
        if abs(fm) < tol:
            return mid
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# vectorized IV for whole chains (per-ticker-day use)
def implied_vol_vec(px: np.ndarray, S, K: np.ndarray, T: np.ndarray,
                    r: float, q: float, cp: np.ndarray) -> np.ndarray:
    """Newton-Raphson IV over arrays, NaN on no-solve.

    S may be a scalar or an array broadcastable to px (whole-day solves
    span many underlyings). r, q are scalars. Rows that don't converge
    or whose quotes are below intrinsic get NaN.
    """
    px = np.asarray(px, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    S_arr = np.broadcast_to(np.asarray(S, float), px.shape)
    is_call = np.asarray(cp) == "C"
    out = np.full(len(px), np.nan)
    ok = (px > 0) & (S_arr > 0) & (K > 0) & (T > 0)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    S_, K_, T_, p_ = S_arr[idx], K[idx], T[idx], px[idx]
    call = is_call[idx]
    # below-intrinsic quotes can't be inverted
    fwd = S_ * np.exp(-q * T_)
    disc = K_ * np.exp(-r * T_)
    intrinsic = np.where(call, np.maximum(0, fwd - disc), np.maximum(0, disc - fwd))
    valid = p_ >= intrinsic * 0.999
    idx, S_, K_, T_, p_, call = (idx[valid], S_[valid], K_[valid],
                               T_[valid], p_[valid], call[valid])
    if len(idx) == 0:
        return out

    vol = np.full(len(idx), 0.25)
    conv = np.zeros(len(idx), bool)
    for _ in range(60):
        price = _bs_price_arr(S_, K_, T_, r, q, vol, call)
        vega = S_ * np.exp(-q * T_) * _pdf_arr(
            (np.log(S_ / K_) + (r - q + 0.5 * vol * vol) * T_)
            / np.maximum(vol * np.sqrt(T_), 1e-9)) * np.sqrt(T_)
        diff = price - p_
        conv = np.abs(diff) < 1e-8
        step = np.where(vega > 1e-10, diff / np.maximum(vega, 1e-10), 0.0)
        vol = np.where(conv, vol, np.clip(vol - step, 1e-4, 10.0))
        if conv.all():
            break
    out[idx] = np.where(conv, vol, np.nan)
    # bisection fallback for non-converged rows (tiny-vega wings)
    bad = np.where(~conv)[0]
    if len(bad):
        lo_v = np.full(len(bad), 1e-4)
        hi_v = np.full(len(bad), 10.0)
        Sb, Kb, Tb, pb, cb = (S_[bad], K_[bad], T_[bad],
                              p_[bad], call[bad])
        conv_b = np.zeros(len(bad), bool)
        mid = lo_v
        for _ in range(200):
            mid = 0.5 * (lo_v + hi_v)
            fm = _bs_price_arr(Sb, Kb, Tb, r, q, mid, cb) - pb
            conv_b = np.abs(fm) < 1e-10
            hi_v = np.where(fm > 0, np.minimum(hi_v, mid), hi_v)
            lo_v = np.where(fm <= 0, np.maximum(lo_v, mid), lo_v)
            if conv_b.all() or (hi_v - lo_v).max() < 1e-12:
                break
        # keep only rows whose residual is actually tiny
        resid = np.abs(_bs_price_arr(Sb, Kb, Tb, r, q, mid, cb) - pb)
        out[idx[bad]] = np.where(resid < 1e-6, mid, np.nan)
    return out


def _bs_price_arr(S: np.ndarray, K: np.ndarray, T: np.ndarray,
                  r: float, q: float, vol: np.ndarray,
                  call: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.maximum(vol * np.sqrt(T), 1e-9)
        d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / sq
        d2 = d1 - sq
        nd1 = _cdf_arr(sign=call, x=d1)
        nd2 = _cdf_arr(sign=call, x=d2)
        return np.where(
            call,
            S * np.exp(-q * T) * nd1 - K * np.exp(-r * T) * nd2,
            K * np.exp(-r * T) * nd2 - S * np.exp(-q * T) * nd1)


def _cdf_arr(sign: np.ndarray, x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm as _n
    return np.where(sign, _n.cdf(x), _n.cdf(-x))


def _pdf_arr(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_greeks_vec(S, K: np.ndarray, T: np.ndarray, r, q,
                  vol: np.ndarray, cp: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized Greeks. S may be scalar or array; r, q scalars.
    Returns arrays aligned with inputs; NaN where T<=0 or vol<=0."""
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    vol = np.asarray(vol, float)
    S = np.broadcast_to(np.asarray(S, float), K.shape)
    is_call = np.asarray(cp) == "C"
    sign = np.where(is_call, 1.0, -1.0)
    ok = (T > 0) & (vol > 0) & (S > 0) & (K > 0)
    nan = np.full(K.shape, np.nan)
    if not ok.any():
        return {g: nan.copy() for g in
                ("delta", "gamma", "vega", "theta", "rho")}
    from scipy.stats import norm as _n
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = vol * np.sqrt(np.maximum(T, 1e-12))
        d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / sq
        d2 = d1 - sq
    pdf = np.where(ok, _n.pdf(d1), np.nan)
    eqt = np.exp(-q * T)
    ert = np.exp(-r * T)
    nd1 = np.where(ok, _n.cdf(sign * d1), np.nan)
    nd2 = np.where(ok, _n.cdf(sign * d2), np.nan)
    delta = np.where(ok, sign * eqt * nd1, np.nan)
    gamma = np.where(ok, eqt * pdf / (S * sq), np.nan)
    vega = np.where(ok, S * eqt * pdf * np.sqrt(T) / 100.0, np.nan)
    theta = np.where(
        ok,
        (-(S * eqt * pdf * vol) / (2 * T)
         - sign * r * K * ert * nd2
         + sign * q * S * eqt * nd1) / 365.0,
        np.nan)
    rho = np.where(ok, sign * K * T * ert * nd2 / 100.0, np.nan)
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}
