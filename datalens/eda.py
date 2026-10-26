"""Descriptive statistics and correlations for exploratory analysis.

Conventions (shown to the user in the UI):

* Standard deviation uses the sample formula (ddof = 1).
* Quartiles use linear interpolation (Hyndman & Fan type 7, the pandas/NumPy
  default). Other software may use a different definition and differ slightly
  on small samples.
* Skewness and excess kurtosis are the bias-adjusted sample estimators
  (pandas ``skew``/``kurt``; scipy ``bias=False``). Normal distribution → 0.
* Outlier flags use Tukey's fences (Q1 − 1.5·IQR, Q3 + 1.5·IQR). A flag means
  "inspect this", not "delete this".
* Correlations use pairwise deletion: every cell is computed on the rows where
  *both* variables are present, so cells can rest on different n.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from datalens.data import numeric_columns

CorrMethod = Literal["pearson", "spearman"]

TUKEY_K = 1.5
# Below this many complete pairs a correlation is not computed: with n = 2,
# Pearson's r is always ±1 regardless of the data.
MIN_PAIRS = 3


def _validate_numeric(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    columns = list(columns)
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(absent)}.")
    numeric = set(numeric_columns(df))
    non_numeric = [c for c in columns if c not in numeric]
    if non_numeric:
        raise ValueError(f"Kolom berikut bukan numerik: {', '.join(non_numeric)}.")
    return columns


def describe_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """One row per column with location, spread, shape and missingness."""
    columns = _validate_numeric(df, columns)
    rows = {}
    for col in columns:
        x = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = x.dropna()
        q1, median, q3 = valid.quantile([0.25, 0.5, 0.75]) if len(valid) else (np.nan,) * 3
        iqr = q3 - q1
        outliers = ((valid < q1 - TUKEY_K * iqr) | (valid > q3 + TUKEY_K * iqr)).sum()
        rows[col] = {
            "n": len(valid),
            "n_hilang": int(x.isna().sum()),
            "mean": valid.mean(),
            "sd": valid.std(ddof=1),
            "min": valid.min(),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": valid.max(),
            "iqr": iqr,
            "skewness": valid.skew(),
            "kurtosis": valid.kurt(),
            "n_outlier": int(outliers),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "variabel"
    return out


@dataclass(frozen=True)
class CorrelationResult:
    method: CorrMethod
    r: pd.DataFrame        # correlation coefficients; NaN where undefined
    n: pd.DataFrame        # pairwise complete-observation counts
    constant: tuple[str, ...]  # columns with zero variance (r undefined)

    @property
    def pairwise_n_varies(self) -> bool:
        off_diag = self.n.to_numpy()[~np.eye(len(self.n), dtype=bool)]
        return off_diag.size > 0 and off_diag.min() != off_diag.max()


def correlation_matrix(
    df: pd.DataFrame, columns: Sequence[str], method: CorrMethod = "pearson"
) -> CorrelationResult:
    columns = _validate_numeric(df, columns)
    if len(columns) < 2:
        raise ValueError("Korelasi membutuhkan minimal dua variabel numerik.")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Metode korelasi tidak dikenal: {method}.")

    data = df[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    present = data.notna().astype(int)
    n = present.T @ present

    r = data.corr(method=method, min_periods=MIN_PAIRS)
    constant = tuple(c for c in columns if data[c].nunique(dropna=True) <= 1)
    # pandas may return ±1 or NaN for zero-variance columns depending on
    # version; make "undefined" explicit.
    r.loc[list(constant), :] = np.nan
    r.loc[:, list(constant)] = np.nan
    return CorrelationResult(method=method, r=r, n=n, constant=constant)


def strongest_pairs(result: CorrelationResult, top: int = 10) -> pd.DataFrame:
    """Unique variable pairs ordered by |r|, largest first; undefined r omitted."""
    cols = list(result.r.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = result.r.loc[a, b]
            if not np.isnan(r):
                rows.append({"variabel_1": a, "variabel_2": b, "r": r, "n": int(result.n.loc[a, b])})
    out = pd.DataFrame(rows, columns=["variabel_1", "variabel_2", "r", "n"])
    if out.empty:
        return out
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def histogram_bins(values: pd.Series, max_bins: int = 100) -> np.ndarray:
    """Bin edges by NumPy's 'auto' rule (max of Sturges and Freedman–Diaconis).

    Freedman–Diaconis can produce thousands of bins for heavy-tailed data, so
    the count is capped at ``max_bins`` equal-width bins.
    """
    x = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if x.size == 0:
        raise ValueError("Tidak ada nilai valid untuk histogram.")
    edges = np.histogram_bin_edges(x, bins="auto")
    if len(edges) - 1 > max_bins:
        edges = np.histogram_bin_edges(x, bins=max_bins)
    return edges
