"""Comparison of two linear models: in-sample fit vs. cross-validated error.

Design decisions:

* Both models (and the intercept-only baseline) are evaluated on the *same*
  complete-case rows: the union of their columns. Otherwise a model could look
  better just because it was scored on different observations.
* Cross-validation uses repeated k-fold with a fixed seed. Every model sees
  identical splits, so per-fold differences are paired.
* Leakage: each estimator is a scikit-learn Pipeline that is cloned and refit
  inside every training fold. The only preprocessing (listwise deletion) is a
  per-row rule that uses no information from other rows, so applying it before
  splitting leaks nothing. Any future learned preprocessing (scaling,
  imputation) must be added as a Pipeline step, not applied up front.
* Fold-to-fold SD describes variability across splits. It is not a standard
  error: folds share training data, so they are not independent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from datalens.regression import ModelSpecError, RegressionResult, fit_ols, prepare_model_data

BASELINE = "Baseline (rata-rata)"


@dataclass(frozen=True)
class NestedFTest:
    restricted: str
    full: str
    f_statistic: float
    df_num: int
    df_denom: int
    p_value: float


@dataclass(frozen=True)
class ComparisonResult:
    target: str
    names: tuple[str, str]
    models: tuple[RegressionResult, RegressionResult]
    n_obs: int
    n_dropped: int
    n_splits: int
    n_repeats: int
    seed: int
    in_sample: pd.DataFrame     # index: model name
    cv_folds: pd.DataFrame      # long: model, split, rmse, mae
    cv_summary: pd.DataFrame    # index: model (baseline first); rmse/mae mean & sd
    paired_rmse_diff: pd.Series  # per split: RMSE(B) − RMSE(A)
    nested_f: NestedFTest | None
    warnings: tuple[str, ...]


def make_estimator() -> Pipeline:
    """OLS as a Pipeline. Any learned preprocessing belongs here, so it is refit per fold."""
    return Pipeline([("ols", LinearRegression())])


def formula(target: str, predictors: Sequence[str]) -> str:
    return f"{target} ~ {' + '.join(predictors)}"


def compare_models(
    df: pd.DataFrame,
    target: str,
    predictors_a: Sequence[str],
    predictors_b: Sequence[str],
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
    names: tuple[str, str] = ("A", "B"),
) -> ComparisonResult:
    predictors_a, predictors_b = list(predictors_a), list(predictors_b)
    if set(predictors_a) == set(predictors_b):
        raise ModelSpecError("Kedua model memiliki prediktor yang sama; pilih spesifikasi yang berbeda.")

    union = list(dict.fromkeys([*predictors_a, *predictors_b]))
    data = prepare_model_data(df, target, union)
    common = pd.concat([data.y, data.X], axis=1)
    n = len(common)

    # Reuses all fit_ols validation (constant/singular/too few rows) on the shared rows.
    model_a = fit_ols(common, target, predictors_a)
    model_b = fit_ols(common, target, predictors_b)

    if not 2 <= n_splits <= n:
        raise ModelSpecError(f"Jumlah fold harus antara 2 dan n = {n}.")
    if n_repeats < 1:
        raise ModelSpecError("Jumlah pengulangan minimal 1.")
    p_max = max(len(predictors_a), len(predictors_b)) + 1
    min_train = n - math.ceil(n / n_splits)
    if min_train <= p_max:
        raise ModelSpecError(
            f"Data terlalu sedikit untuk {n_splits}-fold CV: fold latih terkecil berisi {min_train} baris, "
            f"sedangkan model terbesar memiliki {p_max} parameter. Kurangi jumlah fold atau prediktor."
        )

    y = common[target].to_numpy()
    splits = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(common))
    specs = {BASELINE: [], names[0]: predictors_a, names[1]: predictors_b}

    rows, rank_deficient = [], {name: 0 for name in names}
    for name, predictors in specs.items():
        X = common[predictors].to_numpy() if predictors else np.zeros((n, 1))
        for i, (train, test) in enumerate(splits):
            if predictors:
                design = np.column_stack([np.ones(len(train)), X[train]])
                if np.linalg.matrix_rank(design) < design.shape[1]:
                    rank_deficient[name] += 1
                est = clone(make_estimator())
            else:
                est = DummyRegressor(strategy="mean")
            est.fit(X[train], y[train])
            pred = est.predict(X[test])
            rows.append(
                {
                    "model": name,
                    "split": i,
                    "rmse": root_mean_squared_error(y[test], pred),
                    "mae": mean_absolute_error(y[test], pred),
                }
            )
    folds = pd.DataFrame(rows)
    summary = folds.groupby("model", sort=False)[["rmse", "mae"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]

    by_model = folds.pivot(index="split", columns="model", values="rmse")
    paired = by_model[names[1]] - by_model[names[0]]

    warnings = []
    for name, count in rank_deficient.items():
        if count:
            warnings.append(
                f"Model {name}: {count} dari {len(splits)} fold latih memiliki matriks desain singular "
                "(misalnya prediktor konstan dalam fold). Estimasi pada fold tersebut tidak unik."
            )

    return ComparisonResult(
        target=target,
        names=names,
        models=(model_a, model_b),
        n_obs=n,
        n_dropped=data.n_dropped_missing + data.n_dropped_nonfinite,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        in_sample=_in_sample_table(names, (model_a, model_b)),
        cv_folds=folds,
        cv_summary=summary,
        paired_rmse_diff=paired,
        nested_f=_nested_f(names, model_a, model_b),
        warnings=tuple(warnings),
    )


def _in_sample_table(names: tuple[str, str], models: tuple[RegressionResult, RegressionResult]) -> pd.DataFrame:
    rows = {}
    for name, m in zip(names, models):
        rows[name] = {
            "formula": formula(m.target, m.predictors),
            "k": len(m.predictors),
            "r_squared": m.r_squared,
            "adj_r_squared": m.adj_r_squared,
            "aic": float(m.fitted.aic),
            "bic": float(m.fitted.bic),
            "rmse_in_sample": float(np.sqrt(m.fitted.ssr / m.n_obs)),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "model"
    return out


def _nested_f(names: tuple[str, str], a: RegressionResult, b: RegressionResult) -> NestedFTest | None:
    """Partial F-test when one model's predictors are a strict subset of the other's."""
    sa, sb = set(a.predictors), set(b.predictors)
    if sa < sb:
        (restricted, r_name), (full, f_name) = (a, names[0]), (b, names[1])
    elif sb < sa:
        (restricted, r_name), (full, f_name) = (b, names[1]), (a, names[0])
    else:
        return None
    f, p, df_diff = full.fitted.compare_f_test(restricted.fitted)
    return NestedFTest(
        restricted=r_name,
        full=f_name,
        f_statistic=float(f),
        df_num=int(df_diff),
        df_denom=int(full.df_resid),
        p_value=float(p),
    )
