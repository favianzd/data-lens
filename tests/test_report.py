from datetime import datetime
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from conftest import make_linear
from datalens.comparison import compare_models
from datalens.diagnostics import diagnose
from datalens.formatting import fmt_num, fmt_pvalue
from datalens.regression import INTERCEPT, fit_ols
from datalens.report import (
    ReportInputs,
    build_report,
    coefficient_findings,
    fit_finding,
    limitations,
    report_filename,
)

WHEN = datetime(2026, 10, 22, 9, 30)


class _Values(HTMLParser):
    """Collects the text of every element carrying data-key."""

    def __init__(self):
        super().__init__()
        self.values: dict[str, str] = {}
        self._key: str | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-key" in attrs:
            self._key = attrs["data-key"]
            self.values[self._key] = ""

    def handle_endtag(self, tag):
        if tag == "span":
            self._key = None

    def handle_data(self, data):
        if self._key is not None:
            self.values[self._key] += data


def _values(html: str) -> dict[str, str]:
    p = _Values()
    p.feed(html)
    return p.values


def _p_inline(p):
    s = fmt_pvalue(p)
    return s if s.startswith("<") else f"= {s}"


@pytest.fixture
def report_df() -> pd.DataFrame:
    df = make_linear(120, seed=21)
    df["noise"] = np.random.default_rng(22).normal(size=len(df))
    df.loc[[3, 7], "x2"] = np.nan
    return df


def _build(df, result, comparison=None, figures=False):
    return build_report(ReportInputs("data uji.csv", df, result, comparison, figures, WHEN))


# --- Every reported number equals the underlying result ----------------------


def test_report_values_match_regression_result(report_df):
    r = fit_ols(report_df, "y", ["x1", "x2"])
    v = _values(_build(report_df, r))

    assert v["n_obs"] == f"{r.n_obs:,}"
    assert v["n_dropped"] == "2"
    assert v["r_squared"] == fmt_num(r.r_squared)
    assert v["adj_r_squared"] == fmt_num(r.adj_r_squared)
    assert v["residual_std_error"] == fmt_num(r.residual_std_error)
    assert v["f_statistic"] == fmt_num(r.f_statistic)
    assert v["f_pvalue"] == _p_inline(r.f_pvalue)
    assert v["equation"] == r.equation
    for term in [INTERCEPT, "x1", "x2"]:
        row = r.coefficients.loc[term]
        assert v[f"coef:{term}:estimate"] == fmt_num(row.estimate)
        assert v[f"coef:{term}:std_error"] == fmt_num(row.std_error)
        assert v[f"coef:{term}:t_value"] == fmt_num(row.t_value)
        assert v[f"coef:{term}:p_value"] == fmt_pvalue(row.p_value)
        assert v[f"coef:{term}:ci_lower"] == fmt_num(row.ci_lower)
        assert v[f"coef:{term}:ci_upper"] == fmt_num(row.ci_upper)


def test_report_values_match_dataset_and_diagnostics(report_df):
    r = fit_ols(report_df, "y", ["x1", "x2"])
    d = diagnose(r)
    v = _values(_build(report_df, r))

    assert v["rows"] == "120" and v["cols"] == "4"
    assert v["rows_missing"] == "2"
    assert v["desc:x2:n"] == "118" and v["desc:x2:n_hilang"] == "2"
    assert v["desc:y:mean"] == fmt_num(report_df["y"].mean())
    for i, t in enumerate(d.tests):
        assert v[f"test:{i}:statistic"] == fmt_num(t.statistic)
        assert v[f"test:{i}:p_value"] == fmt_pvalue(t.p_value)
    assert v["durbin_watson"] == f"{d.durbin_watson:.3f}"
    assert v["vif:x1"] == fmt_num(d.vif.loc["x1", "vif"])
    assert v["n_influential"] == str(len(d.influential))


def test_report_values_match_comparison(report_df):
    r = fit_ols(report_df, "y", ["x1", "x2"])
    c = compare_models(report_df, "y", ["x1"], ["x1", "x2"], n_repeats=2)
    v = _values(_build(report_df, r, c))
    for name in c.names:
        assert v[f"cmp:{name}:r_squared"] == fmt_num(c.in_sample.loc[name, "r_squared"])
        assert v[f"cmp:{name}:bic"] == f"{c.in_sample.loc[name, 'bic']:.2f}"
    for m in c.cv_summary.index:
        assert v[f"cv:{m}:rmse_mean"] == fmt_num(c.cv_summary.loc[m, "rmse_mean"])
        assert v[f"cv:{m}:mae_std"] == fmt_num(c.cv_summary.loc[m, "mae_std"], 3)
    assert v["nested_f"] == fmt_num(c.nested_f.f_statistic)
    assert v["cmp_n"] == f"{c.n_obs:,}"


# --- Rule-based text only says what the data supports ------------------------


def test_coefficient_findings_use_association_language_and_ci(report_df):
    r = fit_ols(report_df, "y", ["x1", "x2", "noise"])
    findings = coefficient_findings(r)
    assert len(findings) == 3
    for text in findings:
        assert "berasosiasi" in text
        assert "menyebabkan" not in text and "pengaruh" not in text
        assert "prediktor lain dalam model dibuat tetap" in text
    noise = r.coefficients.loc["noise"]
    expected = "mencakup" if noise.ci_lower <= 0 <= noise.ci_upper else "tidak mencakup"
    assert findings[2].endswith(f"Interval ini {expected} nol.")
    assert "tidak mencakup nol" in findings[0]  # x1 has a true slope of 3


def test_simple_regression_findings_do_not_mention_other_predictors(report_df):
    text = coefficient_findings(fit_ols(report_df, "y", ["x1"]))[0]
    assert "prediktor lain" not in text


def test_fit_finding_reports_r_squared_as_in_sample(report_df):
    r = fit_ols(report_df, "y", ["x1"])
    text = fit_finding(r)
    assert f"{r.r_squared:.1%}" in text
    assert "bukan akurasi prediksi" in text


def test_limitations_only_mention_violations_that_were_detected():
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 10, 300)
    hetero = pd.DataFrame({"x": x, "y": 2 + 3 * x + rng.normal(scale=0.6 * x)})
    r = fit_ols(hetero, "y", ["x"])
    items = " ".join(limitations(r, diagnose(r), None))
    assert "heteroskedastisitas" in items
    assert "asosiasi, bukan hubungan sebab-akibat" in items

    clean = make_linear(300, seed=4)
    rc = fit_ols(clean, "y", ["x1", "x2"])
    dc = diagnose(rc)
    assert dc.tests[0].p_value >= 0.05  # precondition for this check
    clean_items = " ".join(limitations(rc, dc, None))
    assert "heteroskedastisitas" not in clean_items
    assert "VIF" not in clean_items


def test_limitations_include_model_warnings_and_unavailable_tests(small_df):
    df = small_df.head(5).copy()
    df.loc[0, "x1"] = np.nan
    r = fit_ols(df, "y", ["x1", "x2"])  # 4 complete rows, df_resid = 1
    items = limitations(r, diagnose(r), None)
    assert any("complete-case" in i for i in items)
    assert any(i.startswith("Uji Ramsey RESET") and "tidak dijalankan" in i for i in items)


# --- Robustness ------------------------------------------------------------------


def test_column_names_are_html_escaped():
    df = make_linear(60, seed=3).rename(columns={"x1": "<script>alert(1)</script>", "y": 'nilai "y" & co'})
    r = fit_ols(df, 'nilai "y" & co', ["<script>alert(1)</script>", "x2"])
    html = build_report(ReportInputs("<img src=x onerror=alert(1)>.csv", df, r, None, False, WHEN))
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script" not in html.lower()  # no figures → no scripts at all


def test_report_without_figures_has_no_plotly_and_is_small(report_df):
    html = _build(report_df, fit_ols(report_df, "y", ["x1", "x2"]), figures=False)
    assert "Plotly" not in html.split("<footer")[0]
    assert len(html) < 100_000


def test_report_with_figures_embeds_plotly_once(report_df):
    r = fit_ols(report_df, "y", ["x1", "x2"])
    c = compare_models(report_df, "y", ["x1"], ["x1", "x2"], n_repeats=1)
    html = _build(report_df, r, c, figures=True)
    assert html.count("plotly.js v") == 1  # library header appears once
    assert html.count('class="plotly-graph-div"') == 4  # 3 diagnostics + CV figure


def test_report_when_diagnostics_unavailable():
    df = pd.DataFrame({"x": np.arange(10.0)})
    df["y"] = 1 + 2 * df["x"]
    r = fit_ols(df, "y", ["x"])  # perfect fit
    html = _build(df, r)
    assert "3. Diagnostik model</h2><p>Tidak tersedia" in html
    assert "fit sempurna" in html
    assert "<h2>4. Keterbatasan</h2>" in html


def test_report_section_numbering_with_comparison(report_df):
    r = fit_ols(report_df, "y", ["x1", "x2"])
    c = compare_models(report_df, "y", ["x1"], ["x1", "x2"], n_repeats=1)
    html = _build(report_df, r, c)
    assert "<h2>4. Perbandingan model</h2>" in html and "<h2>5. Keterbatasan</h2>" in html


def test_report_filename_is_safe():
    name = report_filename("../data penjualan 2024.csv", WHEN)
    assert name == "datalens_laporan____data_penjualan_2024_20261022_0930.html"
    assert "/" not in name and " " not in name
