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

def load(path: str | Path, drop_unthermalized: bool = True) -> pd.DataFrame:
    """Load and lightly validate the master CSV produced by run_shared.x.

    Parameters
    ----------
    path:
        Path to the CSV file.
    drop_unthermalized:
        If True (default) rows with ``is_thermalized == 0`` are discarded
        with a warning.

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
    if drop_unthermalized:
        bad = df["is_thermalized"] == 0
        if bad.any():
            warnings.warn(
                f"Dropping {bad.sum()} / {n_total} rows that did not thermalize.",
                stacklevel=2,
            )
            df = df[~bad].copy()

    df["N"] = df["L"] ** 2          # number of spins
    df["T"] = df["T"].round(6)      # normalise floating-point T labels
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
    """Propagated statistical error on χ (linear error propagation)."""
    dm2   = err_M2   / N**2
    dmabs = err_M_abs / N
    # ∂χ/∂⟨m²⟩ = N/T,  ∂χ/∂⟨|m|⟩ = -2N⟨|m|⟩/T
    mabs = M_abs / N
    return np.sqrt((N / T * dm2)**2 + (2 * N * mabs / T * dmabs)**2)


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
    """Propagated statistical error on Cv."""
    de2 = err_E2 / N**2
    de  = err_E  / N
    e   = E / N
    return np.sqrt((N / T**2 * de2)**2 + (2 * N * e / T**2 * de)**2)


def _binder(M4: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """U₄ = 1 − ⟨m⁴⟩ / (3 ⟨m²⟩²)   (dimensionless, N-independent)."""
    m4 = M4 / M2**0   # raw extensive moments cancel N: use M4/M2² = (M4/N⁴)/(M2/N²)²
    # But the raw CSV gives extensive M2, M4 (sum over lattice),
    # so the ratio M4/M2² already normalises correctly.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(M2 != 0, M4 / (3.0 * M2**2), np.nan)
    return 1.0 - ratio


def _binder_err(
    M4: np.ndarray, err_M4: np.ndarray,
    M2: np.ndarray, err_M2: np.ndarray,
) -> np.ndarray:
    """Propagated statistical error on U₄."""
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

def peak_temperature(df_agg: pd.DataFrame, obs: str = "chi") -> pd.DataFrame:
    """Find the temperature of the peak of *obs* for each lattice size L.

    Parameters
    ----------
    obs:
        Observable column name, e.g. ``"chi"`` or ``"cv"``.

    Returns
    -------
    pd.DataFrame with columns: L, T_peak, peak_value
    """
    rows = []
    for L, grp in df_agg.groupby("L"):
        idx = grp[obs].idxmax()
        rows.append({"L": L, "T_peak": grp.loc[idx, "T"], "peak_value": grp.loc[idx, obs]})
    return pd.DataFrame(rows)


def binder_crossing(df_agg: pd.DataFrame) -> dict:
    """Estimate Tc from the crossing of the Binder cumulant curves.

    Uses pairwise linear interpolation between adjacent (L1, L2) pairs.

    Returns
    -------
    dict mapping ``(L1, L2)`` to estimated ``T_cross``.
    """
    Ls = sorted(df_agg["L"].unique())
    crossings: dict[tuple, float] = {}
    for i in range(len(Ls) - 1):
        L1, L2 = Ls[i], Ls[i + 1]
        g1 = df_agg[df_agg["L"] == L1].sort_values("T")
        g2 = df_agg[df_agg["L"] == L2].sort_values("T")
        # Interpolate onto common T grid
        T_common = np.intersect1d(g1["T"].values, g2["T"].values)
        if len(T_common) < 2:
            continue
        b1 = np.interp(T_common, g1["T"].values, g1["binder"].values)
        b2 = np.interp(T_common, g2["T"].values, g2["binder"].values)
        diff = b1 - b2
        # Find sign change
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            continue
        k = sign_changes[0]
        # Linear interpolation for the root
        T1, T2 = T_common[k], T_common[k + 1]
        d1, d2 = diff[k], diff[k + 1]
        T_cross = T1 - d1 * (T2 - T1) / (d2 - d1)
        crossings[(L1, L2)] = T_cross
    return crossings


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
) -> dict[str, ExtrapResult]:
    """Extrapolate the critical temperature Tc using three independent methods.

    All methods are based on finite-size scaling (FSS):

    **Method 1 — χ peak scaling** (``"chi_peak"``)
        Fits  $T_{\\rm peak}^\\chi(L) = T_c + a \\cdot L^{-1/\\nu}$
        to the temperatures where χ is maximal for each L.

    **Method 2 — C_v peak scaling** (``"cv_peak"``)
        Same FSS ansatz applied to the specific-heat peak positions.

    **Method 3 — Binder crossing average** (``"binder"``)
        Collects all pairwise Binder-cumulant crossing temperatures and
        performs a weighted average, with weights $w \\propto (L_1 L_2)$
        (larger pairs are closer to the thermodynamic limit).
        Also fits a residual-drift model
        $T_{\\rm cross}(L_1,L_2) = T_c + b \\cdot (L_1 L_2)^{-1/\\nu}$
        when enough crossing points exist.

    **Combined estimate** (``"combined"``)
        Inverse-variance weighted mean of the three Tc values above.

    Parameters
    ----------
    df_agg:
        Aggregated observable DataFrame from :func:`compute_observables`.
    fix_nu:
        If True (default), fix the correlation-length exponent ν = 1
        (exact 2D Ising value). Set False to fit ν as a free parameter
        (needs ≥ 4 sizes for reliable convergence).
    min_sizes:
        Minimum number of distinct lattice sizes required to attempt a fit.
        Methods with fewer data points fall back to a linear 1/L estimate.

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
        peaks = peak_temperature(df_agg, obs)
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
    crossings = binder_crossing(df_agg)
    if not crossings:
        results["binder"] = ExtrapResult(
            "Binder crossing",
            TC_EXACT, np.nan,
            meta={"warning": "no crossings found"},
        )
    else:
        cross_vals = np.array(list(crossings.values()))
        cross_keys = list(crossings.keys())

        # Weight ∝ L1*L2 (larger pairs converge faster to Tc)
        weights = np.array([L1 * L2 for (L1, L2) in cross_keys], dtype=float)
        weights /= weights.sum()

        Tc_wmean = float(np.dot(weights, cross_vals))
        # Weighted standard deviation as uncertainty proxy
        Tc_wstd  = float(np.sqrt(np.dot(weights, (cross_vals - Tc_wmean) ** 2)))
        Tc_err   = Tc_wstd if len(cross_vals) > 1 else np.nan

        params = {f"T_cross(L={L1},{L2})": float(v)
                  for (L1, L2), v in crossings.items()}
        params["weights"] = {f"({L1},{L2})": float(w)
                             for (L1, L2), w in zip(cross_keys, weights)}

        # Optional: fit drift T_cross ~ Tc + b*(L1*L2)^{-1/nu}
        drift_meta: dict = {}
        if HAS_SCIPY and len(crossings) >= 3:
            geom_L = np.array([np.sqrt(L1 * L2) for (L1, L2) in cross_keys])
            try:
                if fix_nu:
                    def _cross_model(x, Tc, b):
                        return Tc + b * x ** (-1.0)
                    pc, pcc = curve_fit(
                        _cross_model, geom_L, cross_vals,
                        p0=[TC_EXACT, 1.0], maxfev=5000,
                    )
                    Tc_fit, b_fit = pc
                    Tc_fit_err = float(np.sqrt(pcc[0, 0]))
                    drift_meta = {
                        "drift_Tc": float(Tc_fit),
                        "drift_Tc_err": Tc_fit_err,
                        "drift_b": float(b_fit),
                        "drift_nu_fixed": 1.0,
                    }
                    # Prefer the drift fit if it converged well
                    if np.isfinite(Tc_fit_err) and Tc_fit_err < Tc_err or np.isnan(Tc_err):
                        Tc_wmean = float(Tc_fit)
                        Tc_err   = Tc_fit_err
                else:
                    def _cross_model_free(x, Tc, b, inv_nu):  # type: ignore[misc]
                        return Tc + b * x ** (-inv_nu)
                    pc, pcc = curve_fit(
                        _cross_model_free, geom_L, cross_vals,
                        p0=[TC_EXACT, 1.0, 1.0], maxfev=5000,
                    )
                    Tc_fit, b_fit, inv_nu_fit = pc
                    Tc_fit_err = float(np.sqrt(pcc[0, 0]))
                    drift_meta = {
                        "drift_Tc": float(Tc_fit),
                        "drift_Tc_err": Tc_fit_err,
                        "drift_b": float(b_fit),
                        "drift_nu": float(1.0 / inv_nu_fit) if inv_nu_fit != 0 else np.nan,
                    }
                    if np.isfinite(Tc_fit_err) and (np.isnan(Tc_err) or Tc_fit_err < Tc_err):
                        Tc_wmean = float(Tc_fit)
                        Tc_err   = Tc_fit_err
            except RuntimeError:
                drift_meta = {"drift_warning": "drift fit did not converge"}

        results["binder"] = ExtrapResult(
            "Binder crossing (weighted mean + drift fit)",
            Tc_wmean, Tc_err,
            params=params,
            meta={"n_crossings": len(crossings), **drift_meta},
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
    cmap = cm.get_cmap("plasma", len(Ls))
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
                   ax=None, title: str | None = None):
    """Plot T_peak(L) vs 1/L and overlay the FSS power-law fit.

    Parameters
    ----------
    extrap:
        Optional dict returned by :func:`extrapolate_tc`.  When provided the
        fitted curve and Tc estimate are drawn; otherwise a simple linear
        regression in 1/L is used as a quick visual guide.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    peaks = peak_temperature(df_agg, obs)
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
            T_curve = Tc_est + a * (1.0 / (L_dense + 1e-12)) ** (-inv_nu)  # L_dense in L space
            # Recalculate: L_dense as 1/L values → L = 1/L_dense
            L_for_curve = np.where(L_dense > 0, 1.0 / L_dense, np.inf)
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

    # Y-axis window: zoom in around Tc
    all_vals = [v for v in Tcs if np.isfinite(v)]
    all_errs = [e for e in errs if np.isfinite(e)]
    if all_vals:
        margin = max(0.01, max(all_errs, default=0.01) * 3)
        yc = TC_EXACT
        ax.set_ylim(yc - margin, yc + margin)

    ax.legend(fontsize=8)
    return ax


def plot_all(df_agg: pd.DataFrame, save_figs: bool = False, out_dir: str = ".",
             extrap: dict | None = None):
    """Produce a 2×3 panel: chi, cv, binder, FSS-chi, FSS-cv, Tc summary."""
    if not HAS_MPL:
        print("matplotlib not available — skipping plots.", file=sys.stderr)
        return

    if extrap is None:
        extrap = extrapolate_tc(df_agg)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("2D Ising Model — Critical Temperature Analysis", fontsize=13)

    plot_chi(df_agg, ax=axes[0, 0])
    plot_cv(df_agg, ax=axes[0, 1])
    plot_binder(df_agg, ax=axes[0, 2])
    plot_fss_peaks(df_agg, obs="chi", extrap=extrap, ax=axes[1, 0])
    plot_fss_peaks(df_agg, obs="cv",  extrap=extrap, ax=axes[1, 1])
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

def print_summary(df_agg: pd.DataFrame, extrap: dict | None = None) -> None:
    """Print a readable summary of peaks, Binder crossings, and Tc extrapolation.

    Parameters
    ----------
    extrap:
        Optional dict returned by :func:`extrapolate_tc`.  When provided a
        full extrapolation section is printed.  If omitted, :func:`extrapolate_tc`
        is called automatically.
    """
    print("\n" + "=" * 60)
    print("  Finite-size scaling summary")
    print("=" * 60)

    for obs, label in [("chi", "χ"), ("cv", "Cv")]:
        peaks = peak_temperature(df_agg, obs)
        print(f"\n  {label} peaks:")
        print(f"    {'L':>6}  {'T_peak':>10}  {'peak_value':>14}")
        for _, row in peaks.iterrows():
            print(f"    {int(row['L']):6d}  {row['T_peak']:10.5f}  {row['peak_value']:14.4f}")

    crossings = binder_crossing(df_agg)
    print("\n  Binder cumulant crossings (estimated Tc):")
    if crossings:
        for (L1, L2), T_cross in crossings.items():
            print(f"    L={L1:4d} × L={L2:4d}  →  Tc ≈ {T_cross:.5f}")
    else:
        print("    (not enough overlapping T values to find crossings)")

    # --- Tc extrapolation ---------------------------------------------------
    if extrap is None:
        extrap = extrapolate_tc(df_agg)

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
        "--keep-unthermalized", action="store_true",
        help="Include rows where is_thermalized == 0",
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
        help="Fit ν as a free parameter instead of fixing ν=1 (needs ≥4 sizes)",
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    print(f"Loading {args.csv} …")
    df_raw = load(args.csv, drop_unthermalized=not args.keep_unthermalized)
    print(f"  {len(df_raw)} rows loaded  "
          f"({df_raw['L'].nunique()} sizes, {df_raw['T'].nunique()} temperatures, "
          f"{df_raw['run_tag'].nunique()} tags)")

    print("Computing observables (χ, Cv, Binder) …")
    df_agg = compute_observables(df_raw)

    print("Extrapolating Tc …")
    extrap = extrapolate_tc(df_agg, fix_nu=not args.free_nu)

    print_summary(df_agg, extrap=extrap)

    if args.save_csv:
        df_agg.to_csv(args.save_csv, index=False)
        print(f"Aggregated table saved: {args.save_csv}")

    if args.plot:
        plot_all(df_agg, save_figs=args.save_figs, out_dir=args.out_dir,
                 extrap=extrap)


if __name__ == "__main__":
    main()
