"""Ordinary least squares regression with explicit input validation.

Design decisions (see README "Statistical decisions"):

* Estimation and inference are delegated to statsmodels; nothing inferential
  is hand-rolled here.
* Missing values: complete-case analysis (listwise deletion) over the model
  columns only. Rows with +/-inf are dropped too and counted separately.
* Variables are used on their original scale, so each slope keeps its
  "per one unit of X" interpretation. Nothing is standardised.
* Standard errors are the classical (non-robust) OLS ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_bool_dtype, is_numeric_dtype

INTERCEPT = "(Intercept)"

# Rule of thumb, not a hard requirement: fewer than ~10 observations per
# predictor makes estimates unstable and inference fragile.
MIN_OBS_PER_PREDICTOR = 10

# R^2 this close to 1 means residuals are numerically zero.
PERFECT_FIT_TOL = 1e-10


class ModelSpecError(ValueError):
    """Raised when the requested model cannot be estimated from the data."""


@dataclass(frozen=True)
class ModelData:
    y: pd.Series
    X: pd.DataFrame  # predictors only, intercept not included
    n_dropped_missing: int
    n_dropped_nonfinite: int


@dataclass(frozen=True)
class RegressionResult:
    target: str
    predictors: tuple[str, ...]
    n_obs: int
    n_dropped_missing: int
    n_dropped_nonfinite: int
    df_model: int
    df_resid: int
    conf_level: float
    # Index: term (INTERCEPT first, then predictors in the order given).
    # Columns: estimate, std_error, t_value, p_value, ci_lower, ci_upper.
    coefficients: pd.DataFrame
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    residual_std_error: float
    warnings: tuple[str, ...]
    fitted: sm.regression.linear_model.RegressionResultsWrapper = field(repr=False, compare=False)

    @property
    def n_dropped(self) -> int:
        return self.n_dropped_missing + self.n_dropped_nonfinite

    @property
    def equation(self) -> str:
        """Fitted equation, e.g. ``ŷ = 2.2 + 0.6 × x``."""
        est = self.coefficients["estimate"]
        b0 = est[INTERCEPT]
        parts = [("−" if b0 < 0 else "") + _fmt(abs(b0))]
        for name in self.predictors:
            b = est[name]
            sign = "−" if b < 0 else "+"
            parts.append(f"{sign} {_fmt(abs(b))} × {name}")
        return "ŷ = " + " ".join(parts)


def prepare_model_data(df: pd.DataFrame, target: str, predictors: Sequence[str]) -> ModelData:
    """Validate the specification and return complete-case model data."""
    predictors = list(predictors)
    if not predictors:
        raise ModelSpecError("Pilih minimal satu prediktor.")
    if len(set(predictors)) != len(predictors):
        raise ModelSpecError("Setiap prediktor hanya boleh dipilih satu kali.")
    if target in predictors:
        raise ModelSpecError(f"Variabel dependen '{target}' tidak boleh sekaligus menjadi prediktor.")

    columns = [target, *predictors]
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise ModelSpecError(f"Kolom tidak ditemukan: {', '.join(absent)}.")
    if INTERCEPT in columns:
        raise ModelSpecError(f"Nama kolom '{INTERCEPT}' dicadangkan untuk intersep; ganti nama kolom tersebut.")
    non_numeric = [c for c in columns if not is_numeric_dtype(df[c]) or is_bool_dtype(df[c])]
    if non_numeric:
        raise ModelSpecError(
            f"Kolom berikut bukan numerik: {', '.join(non_numeric)}. "
            "Versi ini hanya mendukung variabel numerik."
        )

    data = df[columns].astype(float)
    has_nan = data.isna().any(axis=1)
    has_inf = np.isinf(data).any(axis=1) & ~has_nan
    data = data[~(has_nan | has_inf)]
    return ModelData(
        y=data[target],
        X=data[predictors],
        n_dropped_missing=int(has_nan.sum()),
        n_dropped_nonfinite=int(has_inf.sum()),
    )


def fit_ols(
    df: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    conf_level: float = 0.95,
) -> RegressionResult:
    """Fit ``target ~ 1 + predictors`` by OLS after validating the input.

    Raises ModelSpecError for any specification that cannot be estimated
    (instead of letting statsmodels return NaNs or a silently pseudo-inverted
    singular design).
    """
    if not 0 < conf_level < 1:
        raise ModelSpecError("Tingkat kepercayaan harus berada di antara 0 dan 1.")

    data = prepare_model_data(df, target, predictors)
    n = len(data.y)
    k = data.X.shape[1]
    n_params = k + 1

    if n == 0:
        raise ModelSpecError("Tidak ada baris lengkap setelah baris dengan nilai hilang dihapus.")
    if n <= n_params:
        raise ModelSpecError(
            f"Observasi lengkap tidak cukup: n = {n}, sedangkan model memiliki {n_params} parameter. "
            f"Dibutuhkan minimal {n_params + 1} observasi agar derajat bebas residual ≥ 1."
        )
    if data.y.nunique() <= 1:
        raise ModelSpecError(
            f"Variabel dependen '{target}' konstan pada data lengkap; tidak ada variasi yang dapat dijelaskan."
        )
    constant = [c for c in data.X.columns if data.X[c].nunique() <= 1]
    if constant:
        raise ModelSpecError(
            f"Prediktor konstan: {', '.join(constant)}. Prediktor tanpa variasi identik dengan intersep "
            "sehingga koefisiennya tidak dapat diestimasi."
        )

    design = data.X.copy()
    design.insert(0, INTERCEPT, 1.0)
    _check_full_rank(design)

    with np.errstate(divide="ignore", invalid="ignore"):
        fitted = sm.OLS(data.y, design).fit()
        alpha = 1 - conf_level
        ci = fitted.conf_int(alpha=alpha)
        coefficients = pd.DataFrame(
            {
                "estimate": fitted.params,
                "std_error": fitted.bse,
                "t_value": fitted.tvalues,
                "p_value": fitted.pvalues,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }
        )
    coefficients.index.name = "term"

    warnings: list[str] = []
    if data.n_dropped_missing or data.n_dropped_nonfinite:
        warnings.append(
            f"{data.n_dropped_missing + data.n_dropped_nonfinite} baris dikeluarkan karena nilai hilang "
            "atau tak hingga pada variabel model (complete-case analysis). Estimasi tetap tidak bias bila "
            "data hilang sepenuhnya acak (MCAR) atau hanya bergantung pada prediktor; bila ketidaklengkapan "
            "berkaitan dengan nilai variabel dependen, estimasi dapat bias."
        )
    if n < MIN_OBS_PER_PREDICTOR * k:
        warnings.append(
            f"Ukuran sampel kecil relatif terhadap jumlah prediktor (n = {n}, k = {k}; aturan praktis "
            f"≥ {MIN_OBS_PER_PREDICTOR} observasi per prediktor). Estimasi bisa tidak stabil."
        )
    if fitted.rsquared > 1 - PERFECT_FIT_TOL:
        warnings.append(
            "Model fit sempurna (residual ≈ 0). Galat baku, uji t, dan p-value tidak bermakna. "
            "Periksa apakah variabel dependen dihitung langsung dari prediktor."
        )

    return RegressionResult(
        target=target,
        predictors=tuple(data.X.columns),
        n_obs=n,
        n_dropped_missing=data.n_dropped_missing,
        n_dropped_nonfinite=data.n_dropped_nonfinite,
        df_model=int(fitted.df_model),
        df_resid=int(fitted.df_resid),
        conf_level=conf_level,
        coefficients=coefficients,
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        f_statistic=float(fitted.fvalue),
        f_pvalue=float(fitted.f_pvalue),
        residual_std_error=float(np.sqrt(fitted.scale)),
        warnings=tuple(warnings),
        fitted=fitted,
    )


def _check_full_rank(design: pd.DataFrame) -> None:
    """Reject singular designs and name the redundant columns.

    Columns are scaled to unit length first so that variables measured on very
    different scales (e.g. years vs. millions) do not distort the rank test.
    """
    norms = np.linalg.norm(design.to_numpy(), axis=0)
    scaled = design / norms
    if np.linalg.matrix_rank(scaled.to_numpy()) == design.shape[1]:
        return

    # Add columns one at a time; a column that does not raise the rank is a
    # linear combination of the intercept and the columns kept before it.
    kept = [INTERCEPT]
    redundant = []
    for col in design.columns[1:]:
        if np.linalg.matrix_rank(scaled[[*kept, col]].to_numpy()) > len(kept):
            kept.append(col)
        else:
            redundant.append(col)
    raise ModelSpecError(
        "Matriks desain singular (multikolinearitas sempurna): "
        f"{', '.join(redundant)} merupakan kombinasi linear dari intersep dan prediktor lain. "
        "Hapus salah satu variabel yang redundan."
    )


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def standardized_coefficients(result: RegressionResult) -> pd.DataFrame:
    """Standardised slopes β_j = b_j · sd(x_j) / sd(y) on the estimation sample.

    Equivalent to refitting the model on z-scored variables; the interval is the
    unstandardised interval multiplied by the same positive factor. Used only to
    compare effect sizes across predictors measured in different units.
    """
    model = result.fitted.model
    X = pd.DataFrame(model.exog, columns=model.exog_names)
    sd_y = float(np.std(model.endog, ddof=1))
    rows = {}
    for name in result.predictors:
        factor = float(X[name].std(ddof=1)) / sd_y
        coef = result.coefficients.loc[name]
        rows[name] = {
            "beta": coef["estimate"] * factor,
            "ci_lower": coef["ci_lower"] * factor,
            "ci_upper": coef["ci_upper"] * factor,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "term"
    return out
