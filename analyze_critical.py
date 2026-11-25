"""
analyze_critical.py
====================
Ingest the CSV produced by run_shared.x and compute the derived
thermodynamic observables needed for finite-size scaling near Tc:

  - chi       : magnetic susceptibility  χ = N·(⟨m²⟩ − ⟨|m|⟩²) / T
  - cv        : specific heat            Cv = N·(⟨e²⟩ − ⟨e⟩²) / T²
  - binder    : Binder cumulant          U₄ = 1 − ⟨m⁴⟩ / (3·⟨m²⟩²)

All quantities are *per spin* (intensive): m = M/N, e = E/N.

The CSV must have the columns written by run_shared.cu:
  L, T, run_tag, is_thermalized,
  M_abs, err_M_abs, tau_M_abs,
  M2,    err_M2,    tau_M2,
  M4,    err_M4,    tau_M4,
  E,     err_E,     tau_E,
  E2,    err_E2,    tau_E2

Usage (CLI):
    python analyze_critical.py master_results.csv --plot --save-figs

Usage (library):
    from analyze_critical import load, compute_observables, plot_all, extrapolate_tc
    df_raw = load("master_results.csv")
    df     = compute_observables(df_raw)
    result = extrapolate_tc(df)
    plot_all(df, extrap=result)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional imports (plotting)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")          # headless-safe; change to "TkAgg" if needed
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Optional imports (fitting)
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))   # ≈ 2.26919  (Onsager)


# ===========================================================================
# I/O
# ===========================================================================

def detect_outliers(df: pd.DataFrame) -> pd.Series:
    """Flag isolated trapped / non-equilibrated low-T runs.

    Below Tc, large lattices (L >= 512) initialized randomly can remain
    trapped in metastable domains during part of a measurement run, causing a
    sharp isolated spike in susceptibility and a corresponding dip in M_abs/U4
    relative to neighboring temperatures at the same L.

    Returns a boolean Series aligned with ``df.index`` (True = suspect outlier).
    """
    if df.empty:
        return pd.Series(False, index=df.index)

    N = (df["L"] ** 2).astype(float).values
    T = df["T"].values
    M_abs = df["M_abs"].values
    M2 = df["M2"].values
    M4 = df["M4"].values

    chi = _chi(M2, M_abs, N, T)
    u4 = _binder(M4, M2)

    flagged = pd.Series(False, index=df.index)
    work = pd.DataFrame({
        "T": T,
        "m": M_abs / N,
        "chi": chi,
        "u4": u4,
    }, index=df.index)

    for _, grp in work.groupby(df["L"]):
        grp_s = grp.sort_values("T")
        if len(grp_s) < 5:
            continue
        chi_med = grp_s["chi"].rolling(window=5, center=True, min_periods=3).median()
        u4_med = grp_s["u4"].rolling(window=5, center=True, min_periods=3).median()
        m_med = grp_s["m"].rolling(window=5, center=True, min_periods=3).median()

        # Catch severe negative Binder spikes or >5x isolated susceptibility spikes
        # accompanied by a drop in order parameter / Binder cumulant.
        is_neg_u4 = grp_s["u4"] < -0.05
        is_chi_spike = (
            (grp_s["chi"] > 5.0 * np.maximum(chi_med, 1.0))
            & ((grp_s["u4"] < u4_med - 1e-4) | (grp_s["m"] < m_med - 0.01))
        )
        flagged.loc[grp_s.index] = (is_neg_u4 | is_chi_spike).values

    return flagged


def load(
    path: str | Path,
    drop_unthermalized: bool = False,
    drop_outliers: bool = False,
) -> pd.DataFrame:
    """Load and lightly validate the master CSV produced by run_shared.x.

    Parameters
    ----------
    path:
        Path to the CSV file.
    drop_unthermalized:
        If True, rows with ``is_thermalized == 0`` are discarded.
        Defaults to False because near Tc critical slowing down causes
        the conservative thermalization test to produce false negatives.
    drop_outliers:
        If True, isolated metastable / trapped low-T runs flagged by
        :func:`detect_outliers` are discarded. Defaults to False (retains
        all rows and prints a notice when anomalies are detected).

    Returns
    -------
    pd.DataFrame with the raw columns plus ``N = L**2``.
    """
    df = pd.read_csv(path)

    required = {
        "L", "T", "run_tag", "is_thermalized",
        "M_abs", "err_M_abs",
        "M2",    "err_M2",
        "M4",    "err_M4",
        "E",     "err_E",
        "E2",    "err_E2",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    n_total = len(df)
    bad = df["is_thermalized"] == 0
    if drop_unthermalized:
        if bad.any():
            warnings.warn(
                f"Dropping {bad.sum()} / {n_total} rows where is_thermalized == 0.",
                stacklevel=2,
            )
            df = df[~bad].copy()
    else:
        if bad.any():
            print(f"  Note: {bad.sum()} / {n_total} rows have is_thermalized == 0 (retained by default).")

    df["N"] = df["L"] ** 2          # number of spins
    df["T"] = df["T"].round(6)      # normalise floating-point T labels

    outliers = detect_outliers(df)
    if outliers.any():
        desc = ", ".join(
            f"(L={int(r['L'])}, T={r['T']:.4f})"
            for _, r in df.loc[outliers, ["L", "T"]].head(6).iterrows()
        )
        suffix = ", ..." if outliers.sum() > 6 else ""
        if drop_outliers:
            warnings.warn(
                f"Dropping {outliers.sum()} / {len(df)} suspect trapped-run outlier(s): {desc}{suffix}",
                stacklevel=2,
            )
            df = df[~outliers].copy()
        else:
            print(
                f"  Note: {outliers.sum()} suspect trapped-run outlier(s) detected ({desc}{suffix}); "
                "retained (pass drop_outliers=True / --drop-outliers to filter)."
            )

    return df.reset_index(drop=True)


# ===========================================================================
# Derived observables
# ===========================================================================

def _chi(M2: np.ndarray, M_abs: np.ndarray, N: np.ndarray, T: np.ndarray) -> np.ndarray:
    """χ = N (⟨m²⟩ − ⟨|m|⟩²) / T   (per-spin, intensive)."""
    m2   = M2   / N**2   # ⟨m²⟩  (m = M/N)
    mabs = M_abs / N     # ⟨|m|⟩
    return N * (m2 - mabs**2) / T


def _chi_err(
    M2: np.ndarray, err_M2: np.ndarray,
    M_abs: np.ndarray, err_M_abs: np.ndarray,
    N: np.ndarray, T: np.ndarray,
) -> np.ndarray:
    """Statistical error on χ accounting for the covariance between ⟨m²⟩ and ⟨|m|⟩²."""
    var_M = np.maximum(M2 - M_abs**2, 1e-12)
    chi = var_M / (N * T)
    # N_eff = var_M / err_M_abs^2  -->  sigma(chi) = chi * sqrt(2 / N_eff)
    return np.sqrt(2.0 * np.maximum(chi, 0.0) / (N * T)) * err_M_abs


def _cv(E2: np.ndarray, E: np.ndarray, N: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Cv = N (⟨e²⟩ − ⟨e⟩²) / T²   (per-spin, intensive)."""
    e2 = E2 / N**2   # ⟨e²⟩
    e  = E  / N      # ⟨e⟩
    return N * (e2 - e**2) / T**2


def _cv_err(
    E2: np.ndarray, err_E2: np.ndarray,
    E: np.ndarray, err_E: np.ndarray,
    N: np.ndarray, T: np.ndarray,
) -> np.ndarray:
    """Statistical error on Cv accounting for the covariance between ⟨e²⟩ and ⟨e⟩²."""
    var_E = np.maximum(E2 - E**2, 1e-12)
    cv = var_E / (N * T**2)
    # N_eff = var_E / err_E^2  -->  sigma(Cv) = Cv * sqrt(2 / N_eff)
    return np.sqrt(2.0 * np.maximum(cv, 0.0) / (N * T**2)) * err_E


def _binder(M4: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """U₄ = 1 − ⟨m⁴⟩ / (3 ⟨m²⟩²)   (dimensionless, N-independent)."""
    # Raw CSV gives extensive M2, M4 (sum over lattice); the ratio M4/M2²
    # cancels N automatically: (M4/N⁴)/(M2/N²)² = M4/M2².
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(M2 != 0, M4 / (3.0 * M2**2), np.nan)
    return 1.0 - ratio


def _binder_err(
    M4: np.ndarray, err_M4: np.ndarray,
    M2: np.ndarray, err_M2: np.ndarray,
) -> np.ndarray:
    """Propagated statistical error on U₄ (upper bound ignoring Cov(M4, M2)).

    Note: The summary CSV does not store Cov(M4, M2), which is strongly
    positive and cancels much of the variance in M4 / M2². As a result, this
    uncorrelated linear propagation overestimates single-point err_binder.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        dU_dM4 = -1.0 / (3.0 * M2**2)
        dU_dM2 = 2.0 * M4 / (3.0 * M2**3)
    return np.sqrt((dU_dM4 * err_M4)**2 + (dU_dM2 * err_M2)**2)


def compute_observables(df: pd.DataFrame) -> pd.DataFrame:
    """Add chi, cv, binder (and their errors) to the DataFrame.

    Works row-by-row on the *per-run* data first, then aggregates
    over independent runs via the ensemble mean and standard error.

    Returns
    -------
    pd.DataFrame indexed by (L, T) with columns:
        chi, err_chi, cv, err_cv, binder, err_binder,
        M_abs, err_M_abs, M2, err_M2, M4, err_M4, E, err_E,
        n_runs, is_thermalized_frac
    """
    # --- per-row (per independent run) computation -------------------------
    M2    = df["M2"].values
    M4    = df["M4"].values
    M_abs = df["M_abs"].values
    E     = df["E"].values
    E2    = df["E2"].values
    N     = df["N"].values
    T     = df["T"].values

    df = df.copy()
    df["chi"]      = _chi(M2, M_abs, N, T)
    df["err_chi"]  = _chi_err(M2, df["err_M2"].values, M_abs, df["err_M_abs"].values, N, T)
    df["cv"]       = _cv(E2, E, N, T)
    df["err_cv"]   = _cv_err(E2, df["err_E2"].values, E, df["err_E"].values, N, T)
    df["binder"]   = _binder(M4, M2)
    df["err_binder"] = _binder_err(M4, df["err_M4"].values, M2, df["err_M2"].values)

    # --- ensemble average over independent runs ----------------------------
    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        cols_mean = ["M_abs", "M2", "M4", "E", "E2",
                     "chi", "cv", "binder"]
        cols_err  = ["err_M_abs", "err_M2", "err_M4", "err_E", "err_E2",
                     "err_chi", "err_cv", "err_binder"]
        out: dict[str, float] = {}
        for c in cols_mean:
            out[c] = g[c].mean()
        # Combine statistical error from block averaging with run-to-run spread
        for c, ec in zip(cols_mean, cols_err):
            # Mean of per-run stat errors (correlated within a run)
            mean_stat_err = g[ec].mean()
            # Standard error of the mean across runs
            run_sem = g[c].std(ddof=1) / np.sqrt(n) if n > 1 else g[ec].mean()
            # Total: quadrature combination
            out[ec] = np.sqrt(mean_stat_err**2 + run_sem**2)
        out["n_runs"] = n
        out["is_thermalized_frac"] = g["is_thermalized"].mean()
        return pd.Series(out)

    agg = (
        df.groupby(["L", "T"], sort=True)
          .apply(_agg)
          .reset_index()
    )
    return agg


# ===========================================================================
# Finite-size scaling utilities
# ===========================================================================

def peak_temperature(
    df_agg: pd.DataFrame,
    obs: str = "chi",
    refine: bool = True,
) -> pd.DataFrame:
    """Find the temperature of the peak of *obs* for each lattice size L.

    Parameters
    ----------
    obs:
        Observable column name, e.g. ``"chi"`` or ``"cv"``.
    refine:
        If True (default), refines the peak location and amplitude with a
        sub-grid 3-point parabolic interpolation.

    Returns
    -------
    pd.DataFrame with columns: L, T_peak, peak_value
    """
    rows = []
    for L, grp in df_agg.groupby("L"):
        grp = grp.sort_values("T").reset_index(drop=True)
        idx = grp[obs].idxmax()
        T_peak = float(grp.loc[idx, "T"])
        peak_val = float(grp.loc[idx, obs])

        if refine and 0 < idx < len(grp) - 1:
            T0 = float(grp.loc[idx - 1, "T"])
            T1 = float(grp.loc[idx, "T"])
            T2 = float(grp.loc[idx + 1, "T"])
            y0 = float(grp.loc[idx - 1, obs])
            y1 = float(grp.loc[idx, obs])
            y2 = float(grp.loc[idx + 1, obs])
            h1 = T1 - T0
            h2 = T2 - T1

            # Check if grid spacing is uniform
            if abs(h1 - h2) < 1e-5:
                h = h1
                denom = 2.0 * (2.0 * y1 - y0 - y2)
                if denom > 0:
                    delta = h * (y2 - y0) / denom
                    if abs(delta) < h:
                        T_peak = T1 + delta
                        peak_val = y1 + 0.5 * (y2 - y0) * (delta / (2.0 * h))
            else:
                # Non-uniform spacing parabolic interpolation
                denom = (T0 - T1) * (T0 - T2) * (T1 - T2)
                if denom != 0:
                    a = (T2 * (y1 - y0) + T1 * (y0 - y2) + T0 * (y2 - y1)) / denom
                    b = (T2**2 * (y0 - y1) + T1**2 * (y2 - y0) + T0**2 * (y1 - y2)) / denom
                    if a < 0:
                        T_cand = -b / (2.0 * a)
                        if T0 <= T_cand <= T2:
                            T_peak = T_cand
                            c = y1 - a * T1**2 - b * T1
                            peak_val = a * T_peak**2 + b * T_peak + c

        rows.append({"L": L, "T_peak": T_peak, "peak_value": peak_val})
    return pd.DataFrame(rows)


def _median3(vals: np.ndarray) -> np.ndarray:
    """Apply a 3-point running median (leaving endpoints unchanged)."""
    if len(vals) < 3:
        return vals.copy()
    out = vals.copy()
    for i in range(1, len(vals) - 1):
        out[i] = float(np.median(vals[i - 1 : i + 2]))
    return out


def _find_crossing_candidates(
    T_common: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    u_min: float,
    u_max: float,
    require_upward: bool = True,
) -> list[tuple[float, float, float, int]]:
    """Find linear-interpolation crossings of ``b1 - b2`` inside ``[u_min, u_max]``.

    Returns list of ``(T_root, U_root, |d(b1-b2)/dT|, interval_idx)``.
    When ``require_upward=True``, keeps only crossings where ``b1 - b2`` goes
    from negative to positive (larger lattice L2 transitions from more ordered
    below Tc to less ordered above Tc).
    """
    diff = b1 - b2
    cands: list[tuple[float, float, float, int]] = []
    for k in range(len(diff) - 1):
        d1, d2 = float(diff[k]), float(diff[k + 1])
        if require_upward:
            is_cross = (d1 <= 0.0 < d2)
        else:
            is_cross = (d1 <= 0.0 < d2) or (d1 >= 0.0 > d2)
        if not is_cross or d2 == d1:
            continue

        T1, T2 = float(T_common[k]), float(T_common[k + 1])
        T_root = T1 - d1 * (T2 - T1) / (d2 - d1)
        w = (T_root - T1) / (T2 - T1)
        U_root = float(b1[k] * (1.0 - w) + b1[k + 1] * w)
        slope = abs((d2 - d1) / (T2 - T1))
        if u_min <= U_root <= u_max:
            cands.append((float(T_root), U_root, slope, k))
    return cands


def _fit_local_binder_curve(
    ts: np.ndarray,
    u4s: np.ndarray,
    errs: np.ndarray,
    t0: float,
    l2: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    """Fit a local quadratic (or linear if <5 pts) polynomial to U4(T) near t0."""
    if len(ts) < 2:
        return None
    idx = int(np.argmin(np.abs(ts - t0)))
    dt_lo = ts[idx] - ts[max(0, idx - 1)]
    dt_hi = ts[min(len(ts) - 1, idx + 1)] - ts[idx]
    dt = max(float(dt_lo), float(dt_hi), 1e-4)

    # Adaptive window scales inversely with L2 so we stay inside the smooth core
    # of the critical drop without curving into the saturation plateaus.
    hw = min(0.005, max(2.2 * dt, 2.5 / float(l2)))
    mask = np.abs(ts - t0) <= hw + 1e-9

    if int(mask.sum()) >= 5:
        ts_win, u4_win, er_win = ts[mask], u4s[mask], errs[mask]
    else:
        n_target = min(5, len(ts))
        lo = max(0, idx - n_target // 2)
        hi = lo + n_target
        if hi > len(ts):
            hi = len(ts)
            lo = max(0, hi - n_target)
        ts_win, u4_win, er_win = ts[lo:hi], u4s[lo:hi], errs[lo:hi]

    if len(ts_win) < 2:
        return None

    xs = ts_win - t0
    deg = 2 if len(xs) >= 5 else 1
    X = np.column_stack([xs ** p for p in range(deg + 1)])
    coeffs_raw, _, _, _ = np.linalg.lstsq(X, u4_win, rcond=None)
    inv_raw = np.linalg.pinv(X.T @ X)

    coeffs = np.zeros(3, dtype=float)
    coeffs[: deg + 1] = coeffs_raw
    inv3 = np.zeros((3, 3), dtype=float)
    inv3[: deg + 1, : deg + 1] = inv_raw

    resid = u4_win - (X @ coeffs_raw)
    dof = len(xs) - (deg + 1)
    s2_emp = float(np.sum(resid ** 2) / dof) if dof >= 1 else 0.0
    valid_e = er_win[np.isfinite(er_win) & (er_win > 0)]
    s2_rep = float(np.mean(valid_e) ** 2) if len(valid_e) > 0 else 0.0
    # Prefer empirical residual variance (dof >= 2) because uncorrelated CSV
    # error propagation omits Cov(M4, M2); fall back to reported error otherwise.
    s2_eff = s2_emp if (dof >= 2 and s2_emp > 0.0) else (s2_rep if s2_rep > 0.0 else s2_emp)
    return coeffs, inv3, s2_eff, float(ts_win.min()), float(ts_win.max())


def _solve_local_crossing(
    fit1: tuple[np.ndarray, np.ndarray, float, float, float],
    fit2: tuple[np.ndarray, np.ndarray, float, float, float],
    t0: float,
) -> tuple[float, float, float] | None:
    """Solve c1(dx) == c2(dx) and propagate uncertainty on T_cross."""
    c1, inv1, s2_1, tmin1, tmax1 = fit1
    c2, inv2, s2_2, tmin2, tmax2 = fit2
    da = float(c1[0] - c2[0])
    db = float(c1[1] - c2[1])
    dc = float(c1[2] - c2[2])

    if abs(dc) < 1e-12:
        if abs(db) < 1e-12:
            return None
        dx = -da / db
    else:
        disc = db * db - 4.0 * dc * da
        if disc < 0.0:
            return None
        sq = float(np.sqrt(disc))
        r1 = (-db + sq) / (2.0 * dc)
        r2 = (-db - sq) / (2.0 * dc)
        dx = r1 if abs(r1) <= abs(r2) else r2

    t_cross = float(t0 + dx)
    lo_b = max(tmin1, tmin2)
    hi_b = min(tmax1, tmax2)
    if not (lo_b - 1e-9 <= t_cross <= hi_b + 1e-9):
        return None

    slope_diff = db + 2.0 * dc * dx
    if slope_diff <= 0.0:
        return None

    u_cross = float(c1[0] + c1[1] * dx + c1[2] * dx * dx)
    v = np.array([1.0, dx, dx * dx], dtype=float)
    g1 = float(v @ inv1 @ v)
    g2 = float(v @ inv2 @ v)
    t_err = float(np.sqrt(max(0.0, s2_1 * g1 + s2_2 * g2)) / slope_diff)
    return t_cross, t_err, u_cross


def binder_crossing_details(
    df_agg: pd.DataFrame,
    u_min: float = 0.08,
    u_max: float = 0.662,
) -> dict[tuple[int, int], dict[str, float]]:
    """Estimate pairwise Binder cumulant crossings with uncertainties.

    Locates the physical transition crossing on 3-point median-filtered curves
    within ``u_min <= U4 <= u_max`` (excluding both the high-T disordered noise
    plateau U4 ≈ 0 and the low-T ferromagnetic ceiling U4 ≈ 2/3, while staying
    compatible with PBC, OBC, and symmetry-broken boundary conditions), then
    refines ``T_cross`` and ``T_err`` using adaptive local quadratic fits.

    Returns
    -------
    dict mapping ``(L1, L2)`` to ``{"T_cross", "T_err", "U_cross", "n_candidates"}``.
    """
    Ls = sorted(int(L) for L in df_agg["L"].unique())
    details: dict[tuple[int, int], dict[str, float]] = {}

    for i in range(len(Ls) - 1):
        L1, L2 = Ls[i], Ls[i + 1]
        g1 = df_agg[df_agg["L"] == L1].sort_values("T")
        g2 = df_agg[df_agg["L"] == L2].sort_values("T")
        t1, b1_raw = g1["T"].values.astype(float), g1["binder"].values.astype(float)
        t2, b2_raw = g2["T"].values.astype(float), g2["binder"].values.astype(float)
        e1_raw = g1["err_binder"].values.astype(float) if "err_binder" in g1 else np.full_like(b1_raw, np.nan)
        e2_raw = g2["err_binder"].values.astype(float) if "err_binder" in g2 else np.full_like(b2_raw, np.nan)

        T_min = max(t1.min(), t2.min())
        T_max = min(t1.max(), t2.max())
        if T_max <= T_min:
            continue

        T_common = np.unique(np.concatenate([
            t1[(t1 >= T_min) & (t1 <= T_max)],
            t2[(t2 >= T_min) & (t2 <= T_max)],
        ]))
        if len(T_common) < 2:
            continue

        b1_med = np.interp(T_common, t1, _median3(b1_raw))
        b2_med = np.interp(T_common, t2, _median3(b2_raw))

        cands = _find_crossing_candidates(T_common, b1_med, b2_med, u_min, u_max, require_upward=True)
        if not cands:
            cands = _find_crossing_candidates(T_common, b1_med, b2_med, 0.04, 0.665, require_upward=False)
        if not cands:
            b1_lin = np.interp(T_common, t1, b1_raw)
            b2_lin = np.interp(T_common, t2, b2_raw)
            cands = _find_crossing_candidates(T_common, b1_lin, b2_lin, 0.04, 0.665, require_upward=False)
        if not cands:
            continue

        # Most decisive crossing (largest differential slope in transition band)
        T0, U0, slope0, k0 = max(cands, key=lambda c: c[2])

        fit1 = _fit_local_binder_curve(t1, b1_raw, e1_raw, T0, L2)
        fit2 = _fit_local_binder_curve(t2, b2_raw, e2_raw, T0, L2)
        solved = _solve_local_crossing(fit1, fit2, T0) if (fit1 is not None and fit2 is not None) else None

        if solved is not None:
            T_cross, T_err, U_cross = solved
        else:
            T_cross, U_cross = T0, U0
            e1_c = float(np.interp(T0, t1, e1_raw)) if np.all(np.isfinite(e1_raw)) else np.nan
            e2_c = float(np.interp(T0, t2, e2_raw)) if np.all(np.isfinite(e2_raw)) else np.nan
            T_err = float(np.hypot(e1_c, e2_c) / slope0) if (slope0 > 0 and np.isfinite(e1_c) and np.isfinite(e2_c)) else np.nan

        details[(L1, L2)] = {
            "T_cross": float(T_cross),
            "T_err": float(T_err),
            "U_cross": float(U_cross),
            "n_candidates": float(len(cands)),
        }

    return details


def binder_crossing(
    df_agg: pd.DataFrame,
    u_min: float = 0.08,
    u_max: float = 0.662,
) -> dict[tuple[int, int], float]:
    """Estimate Tc from pairwise crossings of the Binder cumulant curves.

    Returns
    -------
    dict mapping ``(L1, L2)`` to estimated ``T_cross``.
    """
    return {
        pair: info["T_cross"]
        for pair, info in binder_crossing_details(df_agg, u_min=u_min, u_max=u_max).items()
    }


# ===========================================================================
# Critical temperature extrapolation
# ===========================================================================

def _fss_model(L: np.ndarray, Tc: float, a: float, inv_nu: float) -> np.ndarray:
    """Finite-size scaling ansatz:  T_peak(L) = Tc + a · L^{-1/ν}."""
    return Tc + a * L ** (-inv_nu)


def _fss_model_fixed_nu(L: np.ndarray, Tc: float, a: float) -> np.ndarray:
    """Same ansatz with ν = 1 fixed (exact 2D Ising value)."""
    return Tc + a * L ** (-1.0)


class ExtrapResult:
    """Container returned by :func:`extrapolate_tc`.

    Attributes
    ----------
    method : str
        Human-readable description of the method used.
    Tc : float
        Extrapolated critical temperature.
    Tc_err : float
        1-σ uncertainty on Tc (from covariance or spread of estimates).
    params : dict
        All fitted / derived parameters (method-dependent).
    meta : dict
        Extra information (e.g. chi-squared, degrees of freedom).
    """

    def __init__(self, method: str, Tc: float, Tc_err: float,
                 params: dict | None = None, meta: dict | None = None):
        self.method  = method
        self.Tc      = Tc
        self.Tc_err  = Tc_err
        self.params  = params or {}
        self.meta    = meta or {}

    def __repr__(self) -> str:
        return (f"ExtrapResult(method={self.method!r}, "
                f"Tc={self.Tc:.6f} ± {self.Tc_err:.6f})")

    def summary(self) -> str:
        lines = [
            f"  Method  : {self.method}",
            f"  Tc      : {self.Tc:.6f} ± {self.Tc_err:.6f}",
            f"  Tc(exact): {TC_EXACT:.6f}  "
            f"  deviation: {abs(self.Tc - TC_EXACT):.6f}",
        ]
        for k, v in self.params.items():
            if isinstance(v, float):
                lines.append(f"  {k:<10}: {v:.6f}")
            else:
                lines.append(f"  {k:<10}: {v}")
        for k, v in self.meta.items():
            lines.append(f"  [{k}]: {v}")
        return "\n".join(lines)


def extrapolate_tc(
    df_agg: pd.DataFrame,
    *,
    fix_nu: bool = True,
    min_sizes: int = 3,
    refine_peaks: bool = True,
    binder_omega: float = 1.0,
) -> dict[str, ExtrapResult]:
    """Extrapolate the critical temperature Tc using three independent methods.

    All methods are based on finite-size scaling (FSS):

    **Method 1 — χ peak scaling** (``"chi_peak"``)
        Fits  $T_{\\rm peak}^\\chi(L) = T_c + a \\cdot L^{-1/\\nu}$
        to the temperatures where χ is maximal for each L.

    **Method 2 — C_v peak scaling** (``"cv_peak"``)
        Same FSS ansatz applied to the specific-heat peak positions.

    **Method 3 — Binder crossing average** (``"binder"``)
        Collects pairwise Binder-cumulant crossing temperatures and errors via
        :func:`binder_crossing_details`, forms an inverse-variance weighted mean,
        and fits the finite-size crossing drift model
        $T_{\\rm cross}(L_1,L_2) = T_c + b \\cdot (L_1 L_2)^{-(1/\\nu + \\omega)/2}$
        (default $\\nu=1, \\omega=1 \\Rightarrow L_{\\rm geom}^{-2}$) when $\\ge 3$ crossings exist.

    **Combined estimate** (``"combined"``)
        Inverse-variance weighted mean of the three Tc values above.

    Parameters
    ----------
    df_agg:
        Aggregated observable DataFrame from :func:`compute_observables`.
    fix_nu:
        If True (default), fix the correlation-length exponent ν = 1
        (exact 2D Ising value). Set False to fit ν / drift exponent freely
        (needs ≥ 4 sizes/crossings for reliable convergence).
    min_sizes:
        Minimum number of distinct lattice sizes required to attempt a fit.
        Methods with fewer data points fall back to a linear 1/L estimate.
    binder_omega:
        Correction-to-scaling exponent ω used in the Binder crossing drift
        exponent ``1/nu + binder_omega`` (default 1.0 → L^{-2}; use 2.0 for
        leading bulk 2D Ising PBC corrections L^{-3}).

    Returns
    -------
    dict[str, ExtrapResult]
        Keys: ``"chi_peak"``, ``"cv_peak"``, ``"binder"``, ``"combined"``.
        Each value is an :class:`ExtrapResult` instance.
    """
    results: dict[str, ExtrapResult] = {}

    # ------------------------------------------------------------------
    # Helper: fit T_peak(L) = Tc + a * L^{-1/nu}
    # ------------------------------------------------------------------
    def _fit_peaks(obs: str, label: str) -> ExtrapResult:
        peaks = peak_temperature(df_agg, obs, refine=refine_peaks)
        Ls     = peaks["L"].values.astype(float)
        Tpeaks = peaks["T_peak"].values

        if len(Ls) < 2:
            return ExtrapResult(
                label, TC_EXACT, np.nan,
                meta={"warning": "fewer than 2 sizes, cannot fit"},
            )

        if len(Ls) < min_sizes or not HAS_SCIPY:
            # Fallback: linear regression in 1/L space
            inv_L  = 1.0 / Ls
            coeffs = np.polyfit(inv_L, Tpeaks, 1)
            Tc_est = float(coeffs[1])
            # Bootstrap uncertainty from the fit residuals
            resid  = Tpeaks - np.polyval(coeffs, inv_L)
            Tc_err = float(np.std(resid) / np.sqrt(len(Ls))) if len(Ls) > 2 else np.nan
            return ExtrapResult(
                f"{label} (linear 1/L fallback)",
                Tc_est, Tc_err,
                params={"a": float(coeffs[0]), "nu": 1.0, "inv_nu": 1.0},
                meta={"n_sizes": len(Ls), "method_detail": "polyfit(1/L, T_peak, 1)"},
            )

        # Non-linear LSQ with scipy.optimize.curve_fit
        if fix_nu:
            model = _fss_model_fixed_nu
            p0    = [TC_EXACT, 1.0]
            pnames = ["Tc", "a"]
        else:
            model = _fss_model
            p0    = [TC_EXACT, 1.0, 1.0]
            pnames = ["Tc", "a", "inv_nu"]

        try:
            popt, pcov = curve_fit(
                model, Ls, Tpeaks,
                p0=p0,
                maxfev=10_000,
                bounds=(
                    [TC_EXACT - 0.5, -10.0] + ([] if fix_nu else [0.1]),
                    [TC_EXACT + 0.5,  10.0] + ([] if fix_nu else [5.0]),
                ),
            )
        except RuntimeError as exc:
            # curve_fit failed to converge: fall back to linear
            inv_L  = 1.0 / Ls
            coeffs = np.polyfit(inv_L, Tpeaks, 1)
            return ExtrapResult(
                f"{label} (linear 1/L fallback — curve_fit failed)",
                float(coeffs[1]), np.nan,
                meta={"warning": str(exc)},
            )

        perr   = np.sqrt(np.diag(pcov))
        Tc_est = float(popt[0])
        Tc_err = float(perr[0])

        # Goodness-of-fit: reduced chi-squared (treat equal weights = 1)
        residuals = Tpeaks - model(Ls, *popt)
        dof       = max(1, len(Ls) - len(popt))
        chi2_red  = float(np.sum(residuals**2) / dof)

        params = {k: float(v) for k, v in zip(pnames, popt)}
        if fix_nu:
            params["nu"] = 1.0
        else:
            params["nu"] = float(1.0 / popt[2]) if popt[2] != 0 else np.nan

        return ExtrapResult(
            label, Tc_est, Tc_err,
            params=params,
            meta={"n_sizes": len(Ls), "chi2_red": chi2_red, "dof": dof,
                  "fix_nu": fix_nu},
        )

    results["chi_peak"] = _fit_peaks("chi", "χ peak FSS")
    results["cv_peak"]  = _fit_peaks("cv",  "Cv peak FSS")

    # ------------------------------------------------------------------
    # Method 3: Binder crossing
    # ------------------------------------------------------------------
    cross_details = binder_crossing_details(df_agg)
    if not cross_details:
        results["binder"] = ExtrapResult(
            "Binder crossing",
            TC_EXACT, np.nan,
            meta={"warning": "no crossings found"},
        )
    else:
        cross_keys = list(cross_details.keys())
        cross_vals = np.array([cross_details[k]["T_cross"] for k in cross_keys], dtype=float)
        cross_errs = np.array([cross_details[k]["T_err"] for k in cross_keys], dtype=float)
        cross_u    = np.array([cross_details[k]["U_cross"] for k in cross_keys], dtype=float)

        use_sigma = bool(np.all(np.isfinite(cross_errs)) and np.all(cross_errs > 0))
        if use_sigma:
            raw_w = 1.0 / (cross_errs ** 2)
            weights = raw_w / raw_w.sum()
            Tc_wmean = float(np.dot(weights, cross_vals))
            chi2_w = float(np.sum(raw_w * (cross_vals - Tc_wmean) ** 2))
            chi2_w_red = chi2_w / max(1, len(cross_vals) - 1) if len(cross_vals) > 1 else 1.0
            Tc_err = float(np.sqrt(1.0 / raw_w.sum()) * max(1.0, np.sqrt(chi2_w_red)))
        else:
            raw_w = np.array([L1 * L2 for (L1, L2) in cross_keys], dtype=float)
            weights = raw_w / raw_w.sum()
            Tc_wmean = float(np.dot(weights, cross_vals))
            Tc_wstd = float(np.sqrt(np.dot(weights, (cross_vals - Tc_wmean) ** 2)))
            Tc_err = Tc_wstd if len(cross_vals) > 1 else np.nan

        params: dict = {}
        for (L1, L2), v, e, u in zip(cross_keys, cross_vals, cross_errs, cross_u):
            err_str = f" ± {e:.5f}" if np.isfinite(e) else ""
            params[f"T_cross(L={L1},{L2})"] = f"{v:.6f}{err_str}  (U*={u:.4f})"
        params["U_cross_mean"] = float(np.mean(cross_u))
        params["weights"] = {f"({L1},{L2})": float(w)
                             for (L1, L2), w in zip(cross_keys, weights)}

        # Fit finite-size crossing drift:
        #   T_cross(L1, L2) = Tc + b * L_geom^{-(1/nu + omega)}
        drift_meta: dict = {}
        Tc_out = Tc_wmean
        Tc_out_err = Tc_err
        if len(cross_keys) >= 3:
            geom_L = np.array([np.sqrt(L1 * L2) for (L1, L2) in cross_keys], dtype=float)
            drift_power = 1.0 + float(binder_omega)
            fitted_free = False

            if not fix_nu and HAS_SCIPY and len(cross_keys) >= 4:
                try:
                    def _cross_model_free(x, Tc, b, p_exp):
                        return Tc + b * x ** (-p_exp)
                    cf_kwargs = {
                        "p0": [TC_EXACT, 1.0, drift_power],
                        "maxfev": 5000,
                        "bounds": ([TC_EXACT - 0.5, -1e4, 1.0], [TC_EXACT + 0.5, 1e4, 5.0]),
                    }
                    if use_sigma:
                        cf_kwargs["sigma"] = cross_errs
                        cf_kwargs["absolute_sigma"] = True
                    pc, pcc = curve_fit(_cross_model_free, geom_L, cross_vals, **cf_kwargs)
                    Tc_fit, b_fit, p_fit = [float(val) for val in pc]
                    resid = cross_vals - _cross_model_free(geom_L, Tc_fit, b_fit, p_fit)
                    dof = max(1, len(cross_keys) - 3)
                    chi2_red = float(np.sum((resid / cross_errs) ** 2) / dof) if use_sigma else float(np.sum(resid ** 2) / dof)
                    scale = max(1.0, float(np.sqrt(chi2_red))) if use_sigma else 1.0
                    Tc_fit_err = float(np.sqrt(pcc[0, 0]) * scale)
                    drift_meta = {
                        "drift_Tc": Tc_fit,
                        "drift_Tc_err": Tc_fit_err,
                        "drift_b": b_fit,
                        "drift_power": p_fit,
                        "drift_chi2_red": chi2_red,
                    }
                    if np.isfinite(Tc_fit_err) and Tc_fit_err > 0:
                        Tc_out = Tc_fit
                        Tc_out_err = Tc_fit_err
                        fitted_free = True
                except RuntimeError:
                    drift_meta = {"drift_warning": "free-exponent drift fit did not converge"}

            if not fitted_free:
                # Exact weighted linear least squares in x = L_geom^{-drift_power}
                x = geom_L ** (-drift_power)
                w_fit = (1.0 / (cross_errs ** 2)) if use_sigma else np.ones_like(cross_vals)
                S = float(w_fit.sum())
                Sx = float(np.dot(w_fit, x))
                Sy = float(np.dot(w_fit, cross_vals))
                Sxx = float(np.dot(w_fit, x * x))
                Sxy = float(np.dot(w_fit, x * cross_vals))
                det = S * Sxx - Sx * Sx
                if det > 0:
                    Tc_fit = (Sxx * Sy - Sx * Sxy) / det
                    b_fit = (S * Sxy - Sx * Sy) / det
                    resid = cross_vals - (Tc_fit + b_fit * x)
                    dof = max(1, len(cross_keys) - 2)
                    chi2_red = float(np.sum(w_fit * (resid ** 2)) / dof)
                    if use_sigma:
                        Tc_fit_err = float(np.sqrt(Sxx / det) * max(1.0, np.sqrt(chi2_red)))
                    else:
                        Tc_fit_err = float(np.sqrt(chi2_red * Sxx / det))
                    drift_meta.update({
                        "drift_Tc": float(Tc_fit),
                        "drift_Tc_err": Tc_fit_err,
                        "drift_b": float(b_fit),
                        "drift_power_fixed": drift_power,
                        "drift_chi2_red": chi2_red,
                        "wmean_Tc": Tc_wmean,
                        "wmean_Tc_err": Tc_err,
                    })
                    if np.isfinite(Tc_fit_err) and Tc_fit_err > 0:
                        Tc_out = float(Tc_fit)
                        Tc_out_err = Tc_fit_err

        results["binder"] = ExtrapResult(
            "Binder crossing (inv-var + drift fit)",
            Tc_out, Tc_out_err,
            params=params,
            meta={"n_crossings": len(cross_keys), **drift_meta},
        )

    # ------------------------------------------------------------------
    # Combined: inverse-variance weighted mean of the three estimates
    # ------------------------------------------------------------------
    sub = ["chi_peak", "cv_peak", "binder"]
    Tcs  = np.array([results[k].Tc     for k in sub])
    errs = np.array([results[k].Tc_err for k in sub])

    # Only include estimates with finite uncertainties
    valid = np.isfinite(errs) & (errs > 0)
    if valid.sum() == 0:
        # All errors unknown: plain mean
        Tc_comb = float(np.nanmean(Tcs))
        Tc_comb_err = float(np.nanstd(Tcs)) if valid.sum() < len(Tcs) else np.nan
    else:
        inv_var  = 1.0 / errs[valid] ** 2
        Tc_comb  = float(np.dot(inv_var, Tcs[valid]) / inv_var.sum())
        Tc_comb_err = float(1.0 / np.sqrt(inv_var.sum()))

    results["combined"] = ExtrapResult(
        "Combined (inverse-variance weighted)",
        Tc_comb, Tc_comb_err,
        params={"contributing_methods": [s for s, v in zip(sub, valid) if v]},
        meta={"n_valid": int(valid.sum())},
    )

    return results


# ===========================================================================
# Plotting
# ===========================================================================

def _color_map(Ls):
    """Return a color per lattice size from a sequential colormap."""
    try:
        cmap = matplotlib.colormaps["plasma"].resampled(len(Ls))
    except (AttributeError, KeyError):
        cmap = plt.get_cmap("plasma", len(Ls))
    return {L: cmap(i) for i, L in enumerate(sorted(Ls))}


def _add_tc_line(ax):
    ax.axvline(TC_EXACT, ls="--", lw=0.8, color="grey", label=f"$T_c^{{\\rm exact}}={TC_EXACT:.4f}$")


def plot_chi(df_agg: pd.DataFrame, ax=None, title: str = "Magnetic Susceptibility χ"):
    """Plot χ vs T for each lattice size L."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    Ls = sorted(df_agg["L"].unique())
    colors = _color_map(Ls)
    for L in Ls:
        g = df_agg[df_agg["L"] == L].sort_values("T")
        ax.errorbar(g["T"], g["chi"], yerr=g["err_chi"],
                    fmt="o-", ms=3, lw=1, capsize=2,
                    color=colors[L], label=f"L={L}")
    _add_tc_line(ax)
    ax.set_xlabel("T")
    ax.set_ylabel("χ / spin")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return ax


def plot_cv(df_agg: pd.DataFrame, ax=None, title: str = "Specific Heat $C_v$"):
    """Plot Cv vs T for each lattice size L."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    Ls = sorted(df_agg["L"].unique())
    colors = _color_map(Ls)
    for L in Ls:
        g = df_agg[df_agg["L"] == L].sort_values("T")
        ax.errorbar(g["T"], g["cv"], yerr=g["err_cv"],
                    fmt="s-", ms=3, lw=1, capsize=2,
                    color=colors[L], label=f"L={L}")
    _add_tc_line(ax)
    ax.set_xlabel("T")
    ax.set_ylabel("$C_v$ / spin")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return ax


def plot_binder(df_agg: pd.DataFrame, ax=None, title: str = "Binder Cumulant $U_4$"):
    """Plot U4 vs T for each lattice size L and mark estimated crossings."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    Ls = sorted(df_agg["L"].unique())
    colors = _color_map(Ls)
    for L in Ls:
        g = df_agg[df_agg["L"] == L].sort_values("T")
        ax.errorbar(g["T"], g["binder"], yerr=g["err_binder"],
                    fmt="^-", ms=3, lw=1, capsize=2,
                    color=colors[L], label=f"L={L}")
    crossings = binder_crossing(df_agg)
    for (L1, L2), T_cross in crossings.items():
        ax.axvline(T_cross, ls=":", lw=0.7, alpha=0.6,
                   label=f"cross L={L1}/{L2}: {T_cross:.4f}")
    _add_tc_line(ax)
    ax.set_xlabel("T")
    ax.set_ylabel("$U_4$")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    return ax


def plot_fss_peaks(df_agg: pd.DataFrame, obs: str = "chi",
                   extrap: dict | None = None,
                   ax=None, title: str | None = None,
                   refine_peaks: bool = True):
    """Plot T_peak(L) vs 1/L and overlay the FSS power-law fit.

    Parameters
    ----------
    extrap:
        Optional dict returned by :func:`extrapolate_tc`.  When provided the
        fitted curve and Tc estimate are drawn; otherwise a simple linear
        regression in 1/L is used as a quick visual guide.
    refine_peaks:
        Whether to use sub-grid parabolic peak refinement.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    peaks = peak_temperature(df_agg, obs, refine=refine_peaks)
    if len(peaks) < 2:
        ax.text(0.5, 0.5, "Not enough data for FSS", transform=ax.transAxes, ha="center")
        return ax
    Ls     = peaks["L"].values.astype(float)
    Tpeaks = peaks["T_peak"].values
    inv_L  = 1.0 / Ls

    ax.scatter(inv_L, Tpeaks, zorder=5, s=40, label="$T_{\\rm peak}(L)$")

    L_dense   = np.linspace(0, inv_L.max() * 1.1, 300)
    inv_dense = L_dense  # x-axis is 1/L

    key = f"{obs}_peak"
    if extrap is not None and key in extrap:
        er = extrap[key]
        Tc_est = er.Tc
        Tc_err = er.Tc_err
        a   = er.params.get("a", None)
        nu  = er.params.get("nu", 1.0)

        if a is not None:
            inv_nu  = 1.0 / nu if nu and nu > 0 else 1.0
            with np.errstate(divide="ignore"):
                L_for_curve = np.where(L_dense > 0, 1.0 / np.maximum(L_dense, 1e-15), np.inf)
            T_curve = _fss_model(L_for_curve, Tc_est, a, inv_nu)
            ax.plot(inv_dense, T_curve, "r-", lw=1.5,
                    label=f"FSS fit (ν={nu:.2f})")
            # 1-σ band from Tc uncertainty
            if np.isfinite(Tc_err):
                ax.fill_between(inv_dense,
                                T_curve - Tc_err, T_curve + Tc_err,
                                color="red", alpha=0.12, label=f"±1σ (Tc err)")
        label_tc = (f"$T_c = {Tc_est:.5f}$"
                    + (f" ± {Tc_err:.5f}" if np.isfinite(Tc_err) else ""))
        ax.axhline(Tc_est, ls="--", lw=1.0, color="red", label=label_tc)
    else:
        # Fallback: plain linear fit
        coeffs = np.polyfit(inv_L, Tpeaks, 1)
        Tc_est = coeffs[1]
        ax.plot(L_dense, np.polyval(coeffs, L_dense), "r--", lw=1.2,
                label=f"linear fit → $T_c ≈ {Tc_est:.4f}$")

    ax.axhline(TC_EXACT, ls=":", color="grey", lw=0.9,
               label=f"$T_c^{{\\rm exact}}={TC_EXACT:.5f}$")
    ax.set_xlabel("$1/L$")
    ax.set_ylabel("$T_{\\rm peak}$")
    ax.set_title(title or f"FSS of {obs} peak")
    ax.legend(fontsize=7)
    return ax


def plot_extrapolation(
    df_agg: pd.DataFrame,
    extrap: dict,
    ax=None,
    title: str = "Tc Extrapolation Summary",
):
    """Bar-chart + whisker plot comparing all Tc estimates.

    Shows the extrapolated Tc from each method side-by-side with the exact
    Onsager value as a horizontal reference line.

    Parameters
    ----------
    extrap:
        Dict returned by :func:`extrapolate_tc`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    order  = ["chi_peak", "cv_peak", "binder", "combined"]
    labels = ["χ peak\nFSS", "Cv peak\nFSS", "Binder\ncrossing", "Combined"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    Tcs  = [extrap[k].Tc     for k in order]
    errs = [extrap[k].Tc_err for k in order]

    xs = np.arange(len(order))
    for i, (x, Tc, err, col, lbl) in enumerate(zip(xs, Tcs, errs, colors, labels)):
        ax.bar(x, Tc - TC_EXACT + 0.0, bottom=TC_EXACT,  # show absolute T
               width=0.5, color=col, alpha=0.75, zorder=2)
        if np.isfinite(err):
            ax.errorbar(x, Tc, yerr=err, fmt="none",
                        color="black", capsize=5, lw=1.5, zorder=3)
        ax.text(x, Tc + (err if np.isfinite(err) else 0) + 0.001,
                f"{Tc:.5f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(TC_EXACT, ls="--", lw=1.2, color="grey",
               label=f"$T_c^{{\\rm exact}} = {TC_EXACT:.5f}$")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$T_c$")
    ax.set_title(title)

    # Y-axis window: zoom in around Tc while ensuring all bars/errorbars fit
    all_vals = [v for v in Tcs if np.isfinite(v)]
    all_errs = [e if np.isfinite(e) else 0.0 for e in errs]
    if all_vals:
        max_dev = max(abs(v - TC_EXACT) + e for v, e in zip(Tcs, all_errs) if np.isfinite(v))
        margin = max(0.01, max(all_errs, default=0.01) * 3, max_dev * 1.25)
        yc = TC_EXACT
        ax.set_ylim(yc - margin, yc + margin)

    ax.legend(fontsize=8)
    return ax


def plot_all(df_agg: pd.DataFrame, save_figs: bool = False, out_dir: str = ".",
             extrap: dict | None = None, refine_peaks: bool = True):
    """Produce a 2×3 panel: chi, cv, binder, FSS-chi, FSS-cv, Tc summary."""
    if not HAS_MPL:
        print("matplotlib not available — skipping plots.", file=sys.stderr)
        return

    if extrap is None:
        extrap = extrapolate_tc(df_agg, refine_peaks=refine_peaks)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("2D Ising Model — Critical Temperature Analysis", fontsize=13)

    plot_chi(df_agg, ax=axes[0, 0])
    plot_cv(df_agg, ax=axes[0, 1])
    plot_binder(df_agg, ax=axes[0, 2])
    plot_fss_peaks(df_agg, obs="chi", extrap=extrap, ax=axes[1, 0], refine_peaks=refine_peaks)
    plot_fss_peaks(df_agg, obs="cv",  extrap=extrap, ax=axes[1, 1], refine_peaks=refine_peaks)
    plot_extrapolation(df_agg, extrap=extrap, ax=axes[1, 2])

    plt.tight_layout()
    if save_figs:
        p = Path(out_dir) / "critical_analysis.png"
        fig.savefig(p, dpi=150)
        print(f"Figure saved: {p}")
    else:
        plt.show()
    return fig


# ===========================================================================
# Summary table
# ===========================================================================

def print_summary(df_agg: pd.DataFrame, extrap: dict | None = None, refine_peaks: bool = True) -> None:
    """Print a readable summary of peaks, Binder crossings, and Tc extrapolation.

    Parameters
    ----------
    extrap:
        Optional dict returned by :func:`extrapolate_tc`.  When provided a
        full extrapolation section is printed.  If omitted, :func:`extrapolate_tc`
        is called automatically.
    refine_peaks:
        Whether to use sub-grid parabolic peak refinement.
    """
    print("\n" + "=" * 60)
    print("  Finite-size scaling summary")
    print("=" * 60)

    for obs, label in [("chi", "χ"), ("cv", "Cv")]:
        peaks = peak_temperature(df_agg, obs, refine=refine_peaks)
        print(f"\n  {label} peaks:")
        print(f"    {'L':>6}  {'T_peak':>10}  {'peak_value':>14}")
        for _, row in peaks.iterrows():
            print(f"    {int(row['L']):6d}  {row['T_peak']:10.5f}  {row['peak_value']:14.4f}")

    cross_details = binder_crossing_details(df_agg)
    print("\n  Binder cumulant crossings (estimated Tc):")
    if cross_details:
        for (L1, L2), info in cross_details.items():
            err = info["T_err"]
            err_str = f" ± {err:.5f}" if np.isfinite(err) else ""
            print(f"    L={L1:4d} × L={L2:4d}  →  Tc ≈ {info['T_cross']:.5f}{err_str}  (U*={info['U_cross']:.4f})")
    else:
        print("    (not enough overlapping T values to find crossings)")

    # --- Tc extrapolation ---------------------------------------------------
    if extrap is None:
        extrap = extrapolate_tc(df_agg, refine_peaks=refine_peaks)

    print("\n" + "=" * 60)
    print("  Critical temperature extrapolation")
    print("=" * 60)
    for key in ("chi_peak", "cv_peak", "binder", "combined"):
        er = extrap.get(key)
        if er is None:
            continue
        print(f"\n  [{key}]")
        print(er.summary())

    print(f"\n  Exact Tc (Onsager) = {TC_EXACT:.8f}")
    print("=" * 60 + "\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv", help="Path to master_results.csv")
    p.add_argument(
        "--drop-unthermalized", action="store_true",
        help="Discard rows where is_thermalized == 0 (default: keep all rows)",
    )
    p.add_argument(
        "--keep-unthermalized", action="store_true", default=True,
        help="Include rows where is_thermalized == 0 (default behavior, kept for compatibility)",
    )
    p.add_argument(
        "--drop-outliers", action="store_true",
        help="Discard isolated trapped / non-equilibrated low-T outliers flagged by detect_outliers",
    )
    p.add_argument(
        "--no-refine-peaks", action="store_true",
        help="Disable sub-grid parabolic peak refinement and use discrete maximum",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Produce figures (requires matplotlib)",
    )
    p.add_argument(
        "--save-figs", action="store_true",
        help="Save figures to PNG instead of displaying",
    )
    p.add_argument(
        "--out-dir", default=".",
        help="Directory for saved figures [default: current dir]",
    )
    p.add_argument(
        "--save-csv", metavar="PATH",
        help="Save the aggregated observable table to this CSV path",
    )
    p.add_argument(
        "--free-nu", action="store_true",
        help="Fit ν / Binder drift exponent freely (needs ≥4 sizes/crossings)",
    )
    p.add_argument(
        "--binder-omega", type=float, default=1.0,
        help="Correction-to-scaling exponent ω in Binder crossing drift L^{-(1/ν+ω)} [default: 1.0]",
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    print(f"Loading {args.csv} …")
    df_raw = load(
        args.csv,
        drop_unthermalized=args.drop_unthermalized,
        drop_outliers=args.drop_outliers,
    )
    print(f"  {len(df_raw)} rows loaded  "
          f"({df_raw['L'].nunique()} sizes, {df_raw['T'].nunique()} temperatures, "
          f"{df_raw['run_tag'].nunique()} tags)")

    print("Computing observables (χ, Cv, Binder) …")
    df_agg = compute_observables(df_raw)

    refine_peaks = not args.no_refine_peaks

    print("Extrapolating Tc …")
    extrap = extrapolate_tc(
        df_agg,
        fix_nu=not args.free_nu,
        refine_peaks=refine_peaks,
        binder_omega=args.binder_omega,
    )

    print_summary(df_agg, extrap=extrap, refine_peaks=refine_peaks)

    if args.save_csv:
        df_agg.to_csv(args.save_csv, index=False)
        print(f"Aggregated table saved: {args.save_csv}")

    if args.plot:
        plot_all(df_agg, save_figs=args.save_figs, out_dir=args.out_dir,
                 extrap=extrap, refine_peaks=refine_peaks)


if __name__ == "__main__":
    main()
