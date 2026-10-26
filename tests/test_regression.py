import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from conftest import TRUE_COEFS, TRUE_INTERCEPT
from datalens.regression import INTERCEPT, ModelSpecError, fit_ols, standardized_coefficients
from datalens.sample_data import TARGET, TRUE_COEFFICIENTS, make_study_dataset

# --- Correctness against known / independent results -----------------------


def test_textbook_simple_regression():
    # Hand-computed: x̄ = 3, ȳ = 4, Sxy = 6, Sxx = 10 → b1 = 0.6, b0 = 2.2,
    # SST = 6, SSreg = b1²·Sxx = 3.6 → R² = 0.6.
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    r = fit_ols(df, "y", ["x"])
    assert r.coefficients.loc["x", "estimate"] == pytest.approx(0.6)
    assert r.coefficients.loc[INTERCEPT, "estimate"] == pytest.approx(2.2)
    assert r.r_squared == pytest.approx(0.6)
    assert r.n_obs == 5
    assert r.df_resid == 3


def test_recovers_known_coefficients(known_df):
    r = fit_ols(known_df, "y", ["x1", "x2"])
    est = r.coefficients["estimate"]
    assert est[INTERCEPT] == pytest.approx(TRUE_INTERCEPT, abs=0.05)
    for name, beta in TRUE_COEFS.items():
        assert est[name] == pytest.approx(beta, abs=0.05)
        row = r.coefficients.loc[name]
        assert row.ci_lower < beta < row.ci_upper


def test_r_squared_matches_population_value(known_df):
    # Var(signal) = 3²·1 + 1.5²·(4²/12) = 9 + 3 = 12, Var(ε) = 1 → R²_pop = 12/13.
    r = fit_ols(known_df, "y", ["x1", "x2"])
    assert r.r_squared == pytest.approx(12 / 13, abs=0.01)


def test_matches_independent_numpy_calculation(small_df):
    conf = 0.90
    r = fit_ols(small_df, "y", ["x1", "x2"], conf_level=conf)

    X = np.column_stack([np.ones(len(small_df)), small_df[["x1", "x2"]].to_numpy()])
    y = small_df["y"].to_numpy()
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    pvals = 2 * stats.t.sf(np.abs(t), df=n - p)
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, df=n - p)
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    adj = 1 - (1 - r2) * (n - 1) / (n - p)

    c = r.coefficients
    np.testing.assert_allclose(c["estimate"], beta, rtol=1e-10)
    np.testing.assert_allclose(c["std_error"], se, rtol=1e-10)
    np.testing.assert_allclose(c["t_value"], t, rtol=1e-10)
    np.testing.assert_allclose(c["p_value"], pvals, rtol=1e-8)
    np.testing.assert_allclose(c["ci_lower"], beta - tcrit * se, rtol=1e-10)
    np.testing.assert_allclose(c["ci_upper"], beta + tcrit * se, rtol=1e-10)
    assert r.r_squared == pytest.approx(r2, rel=1e-12)
    assert r.adj_r_squared == pytest.approx(adj, rel=1e-12)
    assert r.residual_std_error == pytest.approx(np.sqrt(sigma2), rel=1e-12)


def test_longley_nist_certified_values():
    # NIST StRD "Longley" — a deliberately ill-conditioned reference problem.
    # https://www.itl.nist.gov/div898/strd/lls/data/Longley.shtml
    data = sm.datasets.longley.load_pandas()
    df = pd.concat([data.exog, data.endog], axis=1)
    predictors = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]
    r = fit_ols(df, "TOTEMP", predictors)

    certified_beta = [
        -3482258.63459582, 15.0618722713733, -0.358191792925910e-01,
        -2.02022980381683, -1.03322686717359, -0.511041056535807e-01, 1829.15146461355,
    ]
    certified_se = [
        890420.383607373, 84.9149257747669, 0.334910077722432e-01,
        0.488399681651699, 0.214274163161675, 0.226073200069370, 455.478499142212,
    ]
    np.testing.assert_allclose(r.coefficients["estimate"], certified_beta, rtol=1e-7)
    np.testing.assert_allclose(r.coefficients["std_error"], certified_se, rtol=1e-7)
    assert r.r_squared == pytest.approx(0.995479004577296, rel=1e-9)
    assert r.residual_std_error == pytest.approx(304.854073561965, rel=1e-8)


def test_demo_dataset_estimates_cover_documented_coefficients():
    r = fit_ols(make_study_dataset(n=2_000), TARGET, list(TRUE_COEFFICIENTS), conf_level=0.99)
    for name, beta in TRUE_COEFFICIENTS.items():
        row = r.coefficients.loc[name]
        assert row.ci_lower < beta < row.ci_upper


def test_equation_string():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    assert fit_ols(df, "y", ["x"]).equation == "ŷ = 2.2 + 0.6 × x"
    df["y_neg"] = -df["y"]
    assert fit_ols(df, "y_neg", ["x"]).equation == "ŷ = −2.2 − 0.6 × x"


# --- Missing and non-finite values ------------------------------------------


def test_listwise_deletion_only_on_model_columns(small_df):
    df = small_df.copy()
    df.loc[0, "y"] = np.nan
    df.loc[1, "x1"] = np.nan
    df.loc[2, "x2"] = np.inf
    df["unused"] = np.nan  # must not cause rows to be dropped

    r = fit_ols(df, "y", ["x1", "x2"])
    assert r.n_obs == len(df) - 3
    assert r.n_dropped_missing == 2
    assert r.n_dropped_nonfinite == 1
    assert any("complete-case" in w for w in r.warnings)

    expected = fit_ols(small_df.drop(index=[0, 1, 2]), "y", ["x1", "x2"])
    np.testing.assert_allclose(r.coefficients["estimate"], expected.coefficients["estimate"], rtol=1e-12)


def test_no_missing_means_no_missing_warning(small_df):
    r = fit_ols(small_df, "y", ["x1", "x2"])
    assert r.n_dropped == 0
    assert not any("complete-case" in w for w in r.warnings)


# --- Invalid specifications ---------------------------------------------------


def test_constant_predictor_rejected(small_df):
    small_df["c"] = 5.0
    with pytest.raises(ModelSpecError, match="konstan.*c"):
        fit_ols(small_df, "y", ["x1", "c"])


def test_predictor_constant_after_listwise_deletion_rejected(small_df):
    # Non-constant overall, but constant on the complete cases.
    small_df["c"] = 1.0
    small_df.loc[0, "c"] = 2.0
    small_df.loc[0, "y"] = np.nan
    with pytest.raises(ModelSpecError, match="konstan"):
        fit_ols(small_df, "y", ["x1", "c"])


def test_constant_target_rejected(small_df):
    small_df["y"] = 1.0
    with pytest.raises(ModelSpecError, match="dependen.*konstan"):
        fit_ols(small_df, "y", ["x1"])


def test_perfect_collinearity_rejected_and_named(small_df):
    small_df["x3"] = small_df["x1"] + 2 * small_df["x2"]
    with pytest.raises(ModelSpecError, match="singular.*x3"):
        fit_ols(small_df, "y", ["x1", "x2", "x3"])


def test_collinear_with_intercept_rejected(small_df):
    small_df["x1_shifted"] = small_df["x1"] + 10  # x1_shifted = x1 + 10·intercept
    with pytest.raises(ModelSpecError, match="x1_shifted"):
        fit_ols(small_df, "y", ["x1", "x1_shifted"])


@pytest.mark.parametrize("n_rows", [1, 2, 3])
def test_insufficient_observations_rejected(n_rows, small_df):
    # Two predictors + intercept = 3 parameters → need ≥ 4 complete rows.
    with pytest.raises(ModelSpecError, match="tidak cukup"):
        fit_ols(small_df.head(n_rows), "y", ["x1", "x2"])


def test_minimum_observations_accepted_with_warning(small_df):
    r = fit_ols(small_df.head(4), "y", ["x1", "x2"])
    assert r.df_resid == 1
    assert any("sampel kecil" in w for w in r.warnings)


def test_all_rows_missing_rejected(small_df):
    small_df["y"] = np.nan
    with pytest.raises(ModelSpecError, match="Tidak ada baris lengkap"):
        fit_ols(small_df, "y", ["x1"])


def test_non_numeric_predictor_rejected(small_df):
    small_df["grp"] = "a"
    with pytest.raises(ModelSpecError, match="bukan numerik.*grp"):
        fit_ols(small_df, "y", ["x1", "grp"])


def test_bool_predictor_rejected(small_df):
    small_df["flag"] = small_df["x1"] > 0
    with pytest.raises(ModelSpecError, match="bukan numerik"):
        fit_ols(small_df, "y", ["flag"])


@pytest.mark.parametrize(
    ("target", "predictors", "match"),
    [
        ("y", [], "minimal satu prediktor"),
        ("y", ["x1", "x1"], "satu kali"),
        ("y", ["y", "x1"], "tidak boleh sekaligus"),
        ("y", ["nope"], "tidak ditemukan"),
        ("nope", ["x1"], "tidak ditemukan"),
    ],
)
def test_invalid_specification_rejected(small_df, target, predictors, match):
    with pytest.raises(ModelSpecError, match=match):
        fit_ols(small_df, target, predictors)


@pytest.mark.parametrize("conf", [0, 1, 1.5, -0.1])
def test_invalid_confidence_level_rejected(small_df, conf):
    with pytest.raises(ModelSpecError, match="kepercayaan"):
        fit_ols(small_df, "y", ["x1"], conf_level=conf)


def test_perfect_fit_warns_that_inference_is_meaningless():
    df = pd.DataFrame({"x": np.arange(10.0)})
    df["y"] = 1 + 2 * df["x"]
    r = fit_ols(df, "y", ["x"])
    assert r.coefficients.loc["x", "estimate"] == pytest.approx(2)
    assert any("fit sempurna" in w for w in r.warnings)


def test_standardized_coefficients_match_fit_on_zscored_data(small_df):
    res = fit_ols(small_df, "y", ["x1", "x2"])
    std = standardized_coefficients(res)
    z = (small_df - small_df.mean()) / small_df.std(ddof=1)
    ref = fit_ols(z, "y", ["x1", "x2"])
    np.testing.assert_allclose(std["beta"], ref.coefficients.loc[["x1", "x2"], "estimate"], rtol=1e-10)
    np.testing.assert_allclose(std["ci_lower"], ref.coefficients.loc[["x1", "x2"], "ci_lower"], rtol=1e-10)
