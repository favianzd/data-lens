import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from statsmodels.stats.anova import anova_lm

from conftest import make_linear
from datalens.comparison import BASELINE, compare_models
from datalens.plots import cv_error_figure
from datalens.regression import ModelSpecError


@pytest.fixture
def cmp_df() -> pd.DataFrame:
    df = make_linear(200, seed=11)
    df["noise"] = np.random.default_rng(12).normal(size=len(df))
    return df


def test_cv_matches_sklearn_cross_val_score(cmp_df):
    res = compare_models(cmp_df, "y", ["x1"], ["x1", "x2"], n_splits=5, n_repeats=3, seed=7)
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=7)
    for name, cols in [("A", ["x1"]), ("B", ["x1", "x2"])]:
        scores = -cross_val_score(LinearRegression(), cmp_df[cols], cmp_df["y"], cv=cv,
                                  scoring="neg_root_mean_squared_error")
        assert res.cv_summary.loc[name, "rmse_mean"] == pytest.approx(scores.mean(), rel=1e-10)
        assert res.cv_summary.loc[name, "rmse_std"] == pytest.approx(scores.std(ddof=1), rel=1e-10)
    base = -cross_val_score(DummyRegressor(), cmp_df[["x1"]], cmp_df["y"], cv=cv,
                            scoring="neg_mean_absolute_error")
    assert res.cv_summary.loc[BASELINE, "mae_mean"] == pytest.approx(base.mean(), rel=1e-10)
    assert len(res.cv_folds) == 3 * 5 * 3


def test_relevant_predictor_improves_cv_error(cmp_df):
    res = compare_models(cmp_df, "y", ["x1"], ["x1", "x2"])
    rmse = res.cv_summary["rmse_mean"]
    assert rmse["B"] < rmse["A"] < rmse[BASELINE]
    # Noise SD is 1, so out-of-sample RMSE of the true model is close to 1.
    assert rmse["B"] == pytest.approx(1.0, abs=0.15)
    assert (res.paired_rmse_diff < 0).mean() > 0.9


def test_noise_predictor_raises_r2_but_not_cv_performance(cmp_df):
    res = compare_models(cmp_df, "y", ["x1", "x2"], ["x1", "x2", "noise"])
    ins = res.in_sample
    assert ins.loc["B", "r_squared"] >= ins.loc["A", "r_squared"]   # always, for nested OLS
    assert ins.loc["B", "bic"] > ins.loc["A", "bic"]                # penalty outweighs the gain
    assert res.cv_summary.loc["B", "rmse_mean"] == pytest.approx(res.cv_summary.loc["A", "rmse_mean"], rel=0.02)


def test_models_scored_on_identical_rows(cmp_df):
    df = cmp_df.copy()
    df.loc[:9, "x2"] = np.nan  # only model B uses x2
    res = compare_models(df, "y", ["x1"], ["x1", "x2"])
    assert res.n_obs == len(df) - 10 and res.n_dropped == 10
    assert res.models[0].n_obs == res.models[1].n_obs == res.n_obs


def test_in_sample_metrics_match_model_results(cmp_df):
    res = compare_models(cmp_df, "y", ["x1"], ["x1", "x2"])
    for name, m in zip(res.names, res.models):
        row = res.in_sample.loc[name]
        assert row["r_squared"] == m.r_squared
        assert row["aic"] == pytest.approx(m.fitted.aic)
        assert row["rmse_in_sample"] == pytest.approx(np.sqrt(np.mean(m.fitted.resid**2)))
        assert row["formula"] == f"y ~ {' + '.join(m.predictors)}"


def test_nested_f_matches_anova(cmp_df):
    res = compare_models(cmp_df, "y", ["x1"], ["x1", "x2"])
    table = anova_lm(res.models[0].fitted, res.models[1].fitted)
    f = res.nested_f
    assert (f.restricted, f.full) == ("A", "B")
    assert f.f_statistic == pytest.approx(table["F"].iloc[1])
    assert f.p_value == pytest.approx(table["Pr(>F)"].iloc[1])
    assert (f.df_num, f.df_denom) == (1, len(cmp_df) - 3)


def test_nested_f_either_direction_and_absent_when_not_nested(cmp_df):
    assert compare_models(cmp_df, "y", ["x1", "x2"], ["x1"]).nested_f.full == "A"
    assert compare_models(cmp_df, "y", ["x1"], ["x2"]).nested_f is None


def test_deterministic_given_seed(cmp_df):
    r1 = compare_models(cmp_df, "y", ["x1"], ["x1", "x2"], seed=3)
    r2 = compare_models(cmp_df, "y", ["x1"], ["x1", "x2"], seed=3)
    pd.testing.assert_frame_equal(r1.cv_folds, r2.cv_folds)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(predictors_a=["x1"], predictors_b=["x1"]), "sama"),
        (dict(predictors_a=["x1"], predictors_b=["x1", "x2"], n_splits=1), "fold"),
        (dict(predictors_a=["x1"], predictors_b=["x1", "x2"], n_repeats=0), "pengulangan"),
        (dict(predictors_a=["x1"], predictors_b=["y"]), "tidak boleh sekaligus"),
    ],
)
def test_invalid_comparisons_rejected(cmp_df, kwargs, match):
    with pytest.raises(ModelSpecError, match=match):
        compare_models(cmp_df, "y", **kwargs)


def test_fold_size_must_exceed_largest_model_parameters():
    df = make_linear(5, seed=2)  # largest model has 3 parameters
    # 5 folds → smallest training fold = 5 − 1 = 4 rows > 3: allowed.
    compare_models(df, "y", ["x1"], ["x1", "x2"], n_splits=5, n_repeats=1)
    # 2 folds → smallest training fold = 5 − 3 = 2 rows ≤ 3: rejected.
    with pytest.raises(ModelSpecError, match="fold latih terkecil berisi 2 baris"):
        compare_models(df, "y", ["x1"], ["x1", "x2"], n_splits=2, n_repeats=1)
    with pytest.raises(ModelSpecError, match="antara 2 dan n"):
        compare_models(df, "y", ["x1"], ["x1", "x2"], n_splits=6, n_repeats=1)


def test_rank_deficient_training_folds_warned():
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({"x1": rng.normal(size=n), "rare": np.r_[np.ones(1), np.zeros(n - 1)]})
    df["y"] = df["x1"] + rng.normal(size=n)
    res = compare_models(df, "y", ["x1"], ["x1", "rare"], n_splits=5, n_repeats=2)
    assert any("singular" in w for w in res.warnings)


def test_cv_figure_shows_fold_values_and_means(cmp_df):
    res = compare_models(cmp_df, "y", ["x1"], ["x1", "x2"], n_repeats=2)
    fig = cv_error_figure(res)
    folds = res.cv_folds
    np.testing.assert_allclose(fig.data[0].x, folds.loc[folds.model == BASELINE, "rmse"])
    assert fig.data[3].x[0] == pytest.approx(res.cv_summary.loc["A", "rmse_mean"])
