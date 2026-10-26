import numpy as np
import pandas as pd
import pytest
from scipy import stats

from datalens.eda import correlation_matrix, describe_numeric, histogram_bins, strongest_pairs
from datalens.plots import correlation_heatmap, distribution_figure


@pytest.fixture
def eda_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(10, 2, n)
    df = pd.DataFrame(
        {
            "x": x,
            "y_lin": 3 * x + rng.normal(0, 1, n),
            "y_neg": -x,
            "y_exp": np.exp(x / 2),      # monotone but nonlinear
            "skewed": rng.exponential(1.0, n),
            "label": rng.choice(["a", "b"], n),
        }
    )
    df.loc[[0, 5, 9], "y_lin"] = np.nan
    df.loc[[1, 2], "skewed"] = np.nan
    return df


# --- Descriptive statistics --------------------------------------------------


def test_describe_matches_numpy_and_scipy(eda_df):
    desc = describe_numeric(eda_df, ["y_lin", "skewed"])
    for col in ["y_lin", "skewed"]:
        x = eda_df[col].dropna().to_numpy()
        row = desc.loc[col]
        assert row["n"] == len(x)
        assert row["n_hilang"] == eda_df[col].isna().sum()
        assert row["mean"] == pytest.approx(x.mean())
        assert row["sd"] == pytest.approx(x.std(ddof=1))
        q1, med, q3 = np.percentile(x, [25, 50, 75])  # NumPy default = linear (type 7)
        assert (row["q1"], row["median"], row["q3"]) == pytest.approx((q1, med, q3))
        assert row["iqr"] == pytest.approx(q3 - q1)
        assert (row["min"], row["max"]) == pytest.approx((x.min(), x.max()))
        assert row["skewness"] == pytest.approx(stats.skew(x, bias=False))
        assert row["kurtosis"] == pytest.approx(stats.kurtosis(x, fisher=True, bias=False))
        expected_outliers = ((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum()
        assert row["n_outlier"] == expected_outliers


def test_describe_known_values():
    df = pd.DataFrame({"v": [1.0, 2, 3, 4, 100]})
    row = describe_numeric(df, ["v"]).loc["v"]
    assert row["mean"] == pytest.approx(22.0)
    assert row["median"] == pytest.approx(3.0)
    assert (row["q1"], row["q3"]) == pytest.approx((2.0, 4.0))
    assert row["n_outlier"] == 1           # fence = 4 + 1.5·2 = 7 < 100
    assert row["skewness"] > 0


def test_describe_ignores_infinite_values():
    df = pd.DataFrame({"v": [1.0, 2.0, np.inf, 3.0, -np.inf]})
    row = describe_numeric(df, ["v"]).loc["v"]
    assert row["n"] == 3
    assert row["n_hilang"] == 2
    assert row["mean"] == pytest.approx(2.0)


def test_describe_small_samples_give_nan_shape_not_errors():
    row = describe_numeric(pd.DataFrame({"v": [1.0, 2.0]}), ["v"]).loc["v"]
    assert np.isnan(row["skewness"]) and np.isnan(row["kurtosis"])


def test_describe_rejects_non_numeric(eda_df):
    with pytest.raises(ValueError, match="bukan numerik"):
        describe_numeric(eda_df, ["label"])


# --- Correlation -------------------------------------------------------------


def test_pearson_matches_scipy_with_pairwise_deletion(eda_df):
    res = correlation_matrix(eda_df, ["x", "y_lin", "skewed"], "pearson")
    for a, b in [("x", "y_lin"), ("x", "skewed"), ("y_lin", "skewed")]:
        pair = eda_df[[a, b]].dropna()
        expected = stats.pearsonr(pair[a], pair[b]).statistic
        assert res.r.loc[a, b] == pytest.approx(expected)
        assert res.r.loc[b, a] == pytest.approx(expected)
        assert res.n.loc[a, b] == len(pair)
    assert res.n.loc["x", "x"] == len(eda_df)
    assert res.pairwise_n_varies


def test_spearman_matches_scipy(eda_df):
    res = correlation_matrix(eda_df, ["x", "y_lin"], "spearman")
    pair = eda_df[["x", "y_lin"]].dropna()
    assert res.r.loc["x", "y_lin"] == pytest.approx(stats.spearmanr(pair["x"], pair["y_lin"]).statistic)


def test_known_correlations(eda_df):
    p = correlation_matrix(eda_df, ["x", "y_neg", "y_exp"], "pearson")
    s = correlation_matrix(eda_df, ["x", "y_neg", "y_exp"], "spearman")
    assert p.r.loc["x", "y_neg"] == pytest.approx(-1.0)
    # Monotone nonlinear relation: Spearman is exactly 1, Pearson is not.
    assert s.r.loc["x", "y_exp"] == pytest.approx(1.0)
    assert p.r.loc["x", "y_exp"] < 0.99
    assert not p.pairwise_n_varies


def test_constant_column_correlation_is_undefined(eda_df):
    eda_df["c"] = 4.0
    res = correlation_matrix(eda_df, ["x", "c"], "pearson")
    assert res.constant == ("c",)
    assert np.isnan(res.r.loc["x", "c"])
    assert strongest_pairs(res).empty


def test_too_few_pairs_gives_nan():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [2.0, 1.0, 3.0, np.nan]})
    res = correlation_matrix(df, ["a", "b"])
    assert res.n.loc["a", "b"] == 2
    assert np.isnan(res.r.loc["a", "b"])


@pytest.mark.parametrize(
    ("columns", "method", "match"),
    [(["x"], "pearson", "minimal dua"), (["x", "y_lin"], "kendall", "tidak dikenal"), (["x", "label"], "pearson", "bukan numerik")],
)
def test_correlation_invalid_input(eda_df, columns, method, match):
    with pytest.raises(ValueError, match=match):
        correlation_matrix(eda_df, columns, method)


def test_strongest_pairs_sorted_by_absolute_r(eda_df):
    res = correlation_matrix(eda_df, ["x", "y_lin", "y_neg", "skewed"])
    pairs = strongest_pairs(res)
    assert len(pairs) == 6
    assert list(pairs["r"].abs()) == sorted(pairs["r"].abs(), reverse=True)
    assert {pairs.iloc[0]["variabel_1"], pairs.iloc[0]["variabel_2"]} == {"x", "y_neg"}


# --- Histogram bins and figures show the computed values --------------------


def test_histogram_bins_cover_data_and_are_capped():
    x = pd.Series(np.r_[np.zeros(1000), np.linspace(0, 1e6, 5)])  # IQR = 0, huge range
    edges = histogram_bins(x, max_bins=50)
    assert edges[0] <= x.min() and edges[-1] >= x.max()
    assert len(edges) - 1 <= 50


def test_histogram_empty_raises():
    with pytest.raises(ValueError):
        histogram_bins(pd.Series([np.nan, np.inf]))


def test_distribution_figure_uses_computed_counts_and_quartiles(eda_df):
    fig = distribution_figure(eda_df["skewed"], "skewed")
    x = eda_df["skewed"].dropna()
    counts, _ = np.histogram(x, bins=histogram_bins(x))
    bars, box = fig.data[0], fig.data[1]
    assert list(bars.y) == list(counts)
    assert sum(bars.y) == len(x)
    desc = describe_numeric(eda_df, ["skewed"]).loc["skewed"]
    assert (box.q1[0], box.median[0], box.q3[0]) == pytest.approx((desc["q1"], desc["median"], desc["q3"]))
    if desc["n_outlier"]:
        assert len(fig.data[2].x) == desc["n_outlier"]


def test_heatmap_shows_correlation_values_strict_lower_triangle(eda_df):
    cols = ["x", "y_lin", "skewed"]
    res = correlation_matrix(eda_df, cols)
    trace = correlation_heatmap(res).data[0]
    z = np.array(trace.z, dtype=float)
    assert list(trace.y) == cols[1:] and list(trace.x) == cols[:-1]
    for i, row in enumerate(trace.y):
        for j, col in enumerate(trace.x):
            if cols.index(col) < cols.index(row):
                assert z[i, j] == pytest.approx(res.r.loc[row, col])
                assert trace.customdata[i][j] == res.n.loc[row, col]
            else:
                assert np.isnan(z[i, j])
