"""Diagnostics for a fitted OLS model.

Two kinds of evidence are produced and kept apart:

* Visual diagnostics (residual plots, Q-Q plot): the primary tool.
* Formal tests (Breusch–Pagan, Shapiro–Wilk / Jarque–Bera, RESET): supporting
  evidence. A non-significant result means "no evidence of a violation was
  found by this test", never "the assumption holds". Small samples give tests
  little power; very large samples make them flag trivial deviations.

All test statistics come from statsmodels / SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from datalens.regression import PERFECT_FIT_TOL, RegressionResult

SHAPIRO_MAX_N = 5000          # SciPy: p-values may be inaccurate beyond this
LOW_POWER_N = 30
LARGE_N = 1000
VIF_MODERATE, VIF_HIGH = 5.0, 10.0
CONDITION_HIGH = 30.0         # Belsley et al., on unit-length scaled columns
STUDENT_RESID_LIMIT = 3.0
RESET_POWER = 3
LOWESS_FRAC = 2 / 3
LOWESS_MIN_N = 10


class DiagnosticsError(ValueError):
    """Raised when diagnostics are not meaningful for the given model."""


@dataclass(frozen=True)
class AssumptionTest:
    assumption: str
    test: str
    h0: str
    statistic: float | None
    df: str | None
    p_value: float | None
    alpha: float
    note: str = ""

    @property
    def available(self) -> bool:
        return self.p_value is not None

    @property
    def conclusion(self) -> str:
        if not self.available:
            return "Tidak tersedia"
        if self.p_value < self.alpha:
            return f"Ada bukti melawan H0 (p < {self.alpha:g})"
        return f"Tidak cukup bukti untuk menolak H0 (p ≥ {self.alpha:g}). Ini bukan bukti bahwa asumsi terpenuhi."


@dataclass(frozen=True)
class Diagnostics:
    # Index = original row labels. Columns: fitted, resid, std_resid
    # (internally studentized), student_resid (externally studentized),
    # leverage, cooks_d, sqrt_abs_std_resid.
    observations: pd.DataFrame
    lowess: np.ndarray | None          # shape (m, 2): sorted fitted, smoothed residual
    scale_lowess: np.ndarray | None    # same, for sqrt(|std_resid|) (scale-location plot)
    qq: pd.DataFrame                   # theoretical, sample (sorted standardized residuals)
    qq_line: tuple[float, float]       # intercept, slope through the quartiles
    tests: tuple[AssumptionTest, ...]
    durbin_watson: float
    vif: pd.DataFrame | None           # None with a single predictor
    condition_number: float
    thresholds: dict[str, float]
    influential: pd.DataFrame
    notes: tuple[str, ...]


def diagnose(result: RegressionResult, alpha: float = 0.05) -> Diagnostics:
    if result.r_squared > 1 - PERFECT_FIT_TOL:
        raise DiagnosticsError("Model fit sempurna (residual ≈ 0), sehingga diagnostik residual tidak bermakna.")

    fitted = result.fitted
    n, p = result.n_obs, result.df_model + 1
    exog = fitted.model.exog

    df_resid = n - p
    with np.errstate(divide="ignore", invalid="ignore"):
        influence = fitted.get_influence()
        std_resid = influence.resid_studentized_internal
        # Exact identity for externally studentized residuals,
        # t_i = r_i · sqrt((n − p − 1) / (n − p − r_i²)). statsmodels' own
        # property refits the model n times (leave-one-out), ~10 s at n = 10k.
        student = (
            std_resid * np.sqrt((df_resid - 1) / (df_resid - std_resid**2))
            if df_resid > 1
            else np.full(n, np.nan)
        )
        obs = pd.DataFrame(
            {
                "fitted": fitted.fittedvalues,
                "resid": fitted.resid,
                "std_resid": std_resid,
                "student_resid": student,
                "leverage": influence.hat_matrix_diag,
                "cooks_d": influence.cooks_distance[0],
            }
        )

    obs["sqrt_abs_std_resid"] = np.sqrt(obs["std_resid"].abs())

    def smooth_of(col: str) -> np.ndarray | None:
        valid = obs[["fitted", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < LOWESS_MIN_N:
            return None
        # delta = 1% of the x-range (R's default) interpolates between nearby
        # points instead of refitting at each one; without it n = 10k takes ~10 s.
        delta = 0.01 * float(np.ptp(valid["fitted"]))
        return lowess(valid[col], valid["fitted"], frac=LOWESS_FRAC, delta=delta, return_sorted=True)
    qq, qq_line = _qq(obs["std_resid"])

    thresholds = {"cooks_d": 4 / n, "leverage": 2 * p / n, "student_resid": STUDENT_RESID_LIMIT}
    flags = (
        (obs["cooks_d"] > thresholds["cooks_d"])
        | (obs["leverage"] > thresholds["leverage"])
        | (obs["student_resid"].abs() > thresholds["student_resid"])
    )
    influential = obs[flags].sort_values("cooks_d", ascending=False)

    notes = []
    if n < LOW_POWER_N:
        notes.append(
            f"n = {n} kecil: uji formal memiliki daya rendah, jadi pelanggaran asumsi mudah tidak terdeteksi."
        )
    if n >= LARGE_N:
        notes.append(
            f"n = {n:,} besar: uji formal dapat menolak H0 karena penyimpangan kecil yang tidak penting "
            "secara praktis. Utamakan plot."
        )

    return Diagnostics(
        observations=obs,
        lowess=smooth_of("resid"),
        scale_lowess=smooth_of("sqrt_abs_std_resid"),
        qq=qq,
        qq_line=qq_line,
        tests=(
            _breusch_pagan(fitted, exog, alpha),
            _normality(fitted.resid.to_numpy(), alpha),
            _reset(fitted, result.df_resid, alpha),
        ),
        durbin_watson=float(durbin_watson(fitted.resid)),
        vif=_vif(exog, list(result.predictors)),
        condition_number=_scaled_condition_number(exog),
        thresholds=thresholds,
        influential=influential,
        notes=tuple(notes),
    )


def _qq(std_resid: pd.Series) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Normal Q-Q points and a reference line through the 1st and 3rd quartiles."""
    sample = std_resid.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    (theoretical, ordered), _ = stats.probplot(sample, dist="norm", fit=True)
    xq = stats.norm.ppf([0.25, 0.75])
    yq = np.quantile(sample, [0.25, 0.75])
    slope = (yq[1] - yq[0]) / (xq[1] - xq[0])
    intercept = yq[0] - slope * xq[0]
    return pd.DataFrame({"theoretical": theoretical, "sample": ordered}), (float(intercept), float(slope))


def _breusch_pagan(fitted, exog: np.ndarray, alpha: float) -> AssumptionTest:
    lm, lm_p, _, _ = het_breuschpagan(fitted.resid, exog, robust=True)
    return AssumptionTest(
        assumption="Homoskedastisitas (varians galat konstan)",
        test="Breusch–Pagan (Koenker, studentized)",
        h0="Varians galat tidak bergantung pada prediktor",
        statistic=float(lm),
        df=f"{exog.shape[1] - 1}",
        p_value=float(lm_p),
        alpha=alpha,
    )


def _normality(resid: np.ndarray, alpha: float) -> AssumptionTest:
    n = len(resid)
    common = dict(
        assumption="Normalitas galat",
        h0="Galat berdistribusi normal",
        alpha=alpha,
    )
    if n < 3:
        return AssumptionTest(test="Shapiro–Wilk", statistic=None, df=None, p_value=None,
                              note="Membutuhkan n ≥ 3.", **common)
    if n <= SHAPIRO_MAX_N:
        w, p = stats.shapiro(resid)
        return AssumptionTest(test="Shapiro–Wilk", statistic=float(w), df=None, p_value=float(p), **common)
    jb, p, _, _ = jarque_bera(resid)
    return AssumptionTest(
        test="Jarque–Bera", statistic=float(jb), df="2", p_value=float(p),
        note=f"n > {SHAPIRO_MAX_N}: Shapiro–Wilk diganti Jarque–Bera.", **common,
    )


def _reset(fitted, df_resid: int, alpha: float) -> AssumptionTest:
    common = dict(
        assumption="Linearitas / bentuk fungsional",
        test=f"Ramsey RESET (ŷ² … ŷ^{RESET_POWER}, uji F)",
        h0="Tidak ada nonlinearitas yang terabaikan dalam bentuk pangkat ŷ",
        alpha=alpha,
    )
    extra = RESET_POWER - 1
    if df_resid - extra < 1:
        return AssumptionTest(statistic=None, df=None, p_value=None,
                              note="Derajat bebas residual tidak cukup untuk menambah suku ŷ².", **common)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = linear_reset(fitted, power=RESET_POWER, test_type="fitted", use_f=True)
    return AssumptionTest(
        statistic=float(np.squeeze(res.fvalue)),
        df=f"{int(res.df_num)}, {int(res.df_denom)}",
        p_value=float(np.squeeze(res.pvalue)),
        **common,
    )


def _vif(exog: np.ndarray, predictors: list[str]) -> pd.DataFrame | None:
    """VIF_j = 1 / (1 − R²_j), R²_j from regressing predictor j on the others (with intercept)."""
    if len(predictors) < 2:
        return None
    with np.errstate(divide="ignore"):
        values = [variance_inflation_factor(exog, i + 1) for i in range(len(predictors))]
    out = pd.DataFrame({"vif": values}, index=pd.Index(predictors, name="prediktor"))
    out["tingkat"] = pd.cut(
        out["vif"], [-np.inf, VIF_MODERATE, VIF_HIGH, np.inf], labels=["rendah", "sedang", "tinggi"], right=False
    ).astype(str)
    return out


def _scaled_condition_number(exog: np.ndarray) -> float:
    scaled = exog / np.linalg.norm(exog, axis=0)
    return float(np.linalg.cond(scaled))
