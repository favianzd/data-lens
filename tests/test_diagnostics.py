import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from conftest import make_linear
from datalens.diagnostics import SHAPIRO_MAX_N, AssumptionTest, DiagnosticsError, diagnose
from datalens.plots import cooks_distance, qq_plot, residual_histogram, residuals_vs_fitted, scale_location
from datalens.regression import fit_ols


def _design(df, predictors):
    return np.column_stack([np.ones(len(df)), df[predictors].to_numpy()])


# --- Influence measures against an independent NumPy computation ------------


def test_influence_measures_match_hat_matrix_formulas(small_df):
    r = fit_ols(small_df, "y", ["x1", "x2"])
    d = diagnose(r)
    X = _design(small_df, ["x1", "x2"])
    n, p = X.shape
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    e = small_df["y"].to_numpy() - H @ small_df["y"].to_numpy()
    s2 = e @ e / (n - p)
    std = e / np.sqrt(s2 * (1 - h))
    cooks = std**2 / p * h / (1 - h)
    # External: leave-one-out variance s²_(i) = ((n−p)s² − e_i²/(1−h_i)) / (n−p−1)
    s2_i = ((n - p) * s2 - e**2 / (1 - h)) / (n - p - 1)
    student = e / np.sqrt(s2_i * (1 - h))

    obs = d.observations
    np.testing.assert_allclose(obs["leverage"], h, rtol=1e-10)
    np.testing.assert_allclose(obs["std_resid"], std, rtol=1e-8)
    np.testing.assert_allclose(obs["student_resid"], student, rtol=1e-8)
    np.testing.assert_allclose(obs["cooks_d"], cooks, rtol=1e-8)
    assert d.thresholds["cooks_d"] == pytest.approx(4 / n)
    assert d.thresholds["leverage"] == pytest.approx(2 * p / n)


def test_planted_outlier_is_flagged(small_df):
    df = small_df.copy()
    df.loc[len(df)] = {"x1": 6.0, "x2": 0.0, "y": -40.0}  # high leverage and a huge residual
    d = diagnose(fit_ols(df, "y", ["x1", "x2"]))
    assert d.influential.index[0] == len(df) - 1


def test_residuals_keep_original_row_labels(small_df):
    df = small_df.copy()
    df.loc[3, "y"] = np.nan
    d = diagnose(fit_ols(df, "y", ["x1"]))
    assert 3 not in d.observations.index
    assert len(d.observations) == len(df) - 1


# --- VIF and condition number --------------------------------------------------


def test_vif_matches_auxiliary_regressions():
    rng = np.random.default_rng(3)
    n = 500
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(scale=0.6, size=n)
    c = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "y": a + b + c + rng.normal(size=n)})
    d = diagnose(fit_ols(df, "y", ["a", "b", "c"]))
    for j, others in [("a", ["b", "c"]), ("b", ["a", "c"]), ("c", ["a", "b"])]:
        r2 = sm.OLS(df[j], sm.add_constant(df[others])).fit().rsquared
        assert d.vif.loc[j, "vif"] == pytest.approx(1 / (1 - r2), rel=1e-10)
    assert d.vif.loc["c", "vif"] < 1.1
    assert d.vif.loc["a", "vif"] > 2


def test_vif_levels_and_condition_number_for_near_collinearity():
    rng = np.random.default_rng(4)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.05, size=300)})
    df["y"] = df["a"] + rng.normal(size=300)
    d = diagnose(fit_ols(df, "y", ["a", "b"]))
    assert (d.vif["tingkat"] == "tinggi").all()
    assert d.condition_number > 30


def test_vif_absent_for_single_predictor(small_df):
    assert diagnose(fit_ols(small_df, "y", ["x1"])).vif is None


# --- Formal tests ------------------------------------------------------------


def test_breusch_pagan_matches_koenker_lm_and_detects_heteroscedasticity():
    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(1, 10, n)
    df = pd.DataFrame({"x": x, "y": 2 + 3 * x + rng.normal(scale=0.5 * x)})  # SD grows with x
    r = fit_ols(df, "y", ["x"])
    bp = diagnose(r).tests[0]
    # Koenker's studentized LM = n · R² of e² regressed on the predictors.
    aux = sm.OLS(r.fitted.resid**2, sm.add_constant(df["x"])).fit()
    assert bp.statistic == pytest.approx(n * aux.rsquared, rel=1e-8)
    assert bp.p_value == pytest.approx(stats.chi2.sf(n * aux.rsquared, 1), rel=1e-8)
    assert bp.p_value < 0.001
    assert bp.conclusion.startswith("Ada bukti")


def test_normality_uses_shapiro_and_matches_scipy(small_df):
    r = fit_ols(small_df, "y", ["x1", "x2"])
    t = diagnose(r).tests[1]
    w, p = stats.shapiro(r.fitted.resid)
    assert t.test == "Shapiro–Wilk"
    assert (t.statistic, t.p_value) == pytest.approx((w, p))


def test_normality_switches_to_jarque_bera_for_large_n():
    r = fit_ols(make_linear(SHAPIRO_MAX_N + 1), "y", ["x1", "x2"])
    t = diagnose(r).tests[1]
    assert t.test == "Jarque–Bera"
    assert t.p_value == pytest.approx(stats.jarque_bera(r.fitted.resid).pvalue)


def test_normality_flags_heavy_tailed_errors():
    rng = np.random.default_rng(6)
    x = rng.normal(size=500)
    df = pd.DataFrame({"x": x, "y": x + rng.standard_t(df=2, size=500)})
    assert diagnose(fit_ols(df, "y", ["x"])).tests[1].p_value < 0.001


def test_reset_detects_omitted_curvature():
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 4, 300)
    df = pd.DataFrame({"x": x, "y": x**2 + rng.normal(scale=0.5, size=300)})
    reset = diagnose(fit_ols(df, "y", ["x"])).tests[2]
    assert reset.df == "2, 296"
    assert reset.p_value < 0.001


def test_reset_unavailable_when_residual_df_too_small(small_df):
    d = diagnose(fit_ols(small_df.head(5), "y", ["x1", "x2"]))  # df_resid = 2, RESET needs 3
    reset = d.tests[2]
    assert not reset.available
    assert reset.conclusion == "Tidak tersedia"
    assert reset.note
    assert any("daya rendah" in n for n in d.notes)


def test_non_significant_result_never_claims_assumption_holds():
    t = AssumptionTest("Normalitas galat", "Shapiro–Wilk", "normal", 0.99, None, 0.6, 0.05)
    assert "bukan bukti bahwa asumsi terpenuhi" in t.conclusion
    assert "terpenuhi" not in t.conclusion.replace("bukan bukti bahwa asumsi terpenuhi", "")


def test_large_n_note(known_df):
    assert any("n = 10,000 besar" in n for n in diagnose(fit_ols(known_df, "y", ["x1"])).notes)


def test_perfect_fit_rejected():
    df = pd.DataFrame({"x": np.arange(10.0)})
    df["y"] = 1 + 2 * df["x"]
    with pytest.raises(DiagnosticsError):
        diagnose(fit_ols(df, "y", ["x"]))


# --- Q-Q and figures show the computed values --------------------------------


def test_qq_line_near_identity_for_normal_errors(known_df):
    d = diagnose(fit_ols(known_df, "y", ["x1", "x2"]))
    intercept, slope = d.qq_line
    assert intercept == pytest.approx(0, abs=0.05)
    assert slope == pytest.approx(1, abs=0.05)
    assert d.qq["sample"].is_monotonic_increasing


def test_diagnostic_figures_plot_computed_values(small_df):
    d = diagnose(fit_ols(small_df, "y", ["x1", "x2"]))
    obs = d.observations
    np.testing.assert_allclose(residuals_vs_fitted(d).data[0].y, obs["resid"])
    np.testing.assert_allclose(scale_location(d).data[0].y, np.sqrt(obs["std_resid"].abs()))
    np.testing.assert_allclose(qq_plot(d).data[0].y, d.qq["sample"])
    assert sum(residual_histogram(d).data[0].y) == len(obs)
    np.testing.assert_allclose(cooks_distance(d).data[0].y, obs["cooks_d"])


def test_tiny_model_diagnostics_do_not_crash(small_df):
    d = diagnose(fit_ols(small_df.head(4), "y", ["x1", "x2"]))  # df_resid = 1
    assert d.lowess is None
    residuals_vs_fitted(d), qq_plot(d), cooks_distance(d)
