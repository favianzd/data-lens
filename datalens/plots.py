"""Plotly figures for the EDA views.

Every number drawn here comes from datalens.eda (or NumPy on the same data),
not from Plotly's own binning/quartile logic, so what the chart shows matches
the tables exactly. Chrome (grid, fonts, background) is left to the Streamlit
theme; only data marks get explicit colors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from datalens.comparison import BASELINE, ComparisonResult
from datalens.diagnostics import Diagnostics
from datalens.eda import TUKEY_K, CorrelationResult, histogram_bins

# DataLens palette (design system, "Data" colours): observed data in navy,
# model / fit / reference lines in emerald, flagged points in copper. The
# diverging scale runs copper (−1) through warm neutral to navy (+1).
SERIES = "#2D4A6E"
SERIES_2 = "#0B7A57"
FLAG = "#B45A1E"
NEUTRAL = "#8A909B"
INK_SECONDARY = "#3A4659"
GRID = "#ECEAE3"
AXIS = "#CFCCC1"
FONT = "Inter, system-ui, sans-serif"
MONO = "JetBrains Mono, ui-monospace, monospace"
WEBGL_THRESHOLD = 5000  # switch scatter traces to WebGL above this many points
DIVERGING = [
    [0.0, "#8E4416"],
    [0.25, "#D9A27C"],
    [0.5, "#F4F3EF"],
    [0.75, "#8FA6C1"],
    [1.0, "#1F3A5F"],
]
MAX_ANNOTATED_CELLS = 12  # above this many variables, cell labels become noise


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        hoverlabel=dict(align="left", font=dict(family=MONO, size=12)),
    )
    axis = dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(family=MONO, size=11),
                title_font=dict(family=FONT, size=12, color=INK_SECONDARY))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def distribution_figure(values: pd.Series, name: str) -> go.Figure:
    """Histogram (top) with a Tukey box plot (bottom) on a shared x-axis."""
    x = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    edges = histogram_bins(x)
    counts, _ = np.histogram(x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    q1, median, q3 = x.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - TUKEY_K * iqr, q3 + TUKEY_K * iqr
    inside = x[(x >= lo) & (x <= hi)]
    outliers = x[(x < lo) | (x > hi)]
    mean = x.mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.04)
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=widths * 0.94,  # thin surface gap between adjacent bins
            marker=dict(color=SERIES, cornerradius=3),
            customdata=np.column_stack([edges[:-1], edges[1:]]),
            hovertemplate="[%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>frekuensi = %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # Precomputed box statistics so Plotly does not re-derive quartiles.
    fig.add_trace(
        go.Box(
            q1=[q1], median=[median], q3=[q3],
            lowerfence=[inside.min()], upperfence=[inside.max()],
            mean=[mean],
            orientation="h", y=[name],
            marker_color=SERIES, line=dict(width=1.5, color=SERIES),
            fillcolor="rgba(45,74,110,0.10)",
            hoverinfo="x",
        ),
        row=2,
        col=1,
    )
    if len(outliers):
        fig.add_trace(
            go.Scatter(
                x=outliers, y=[name] * len(outliers), mode="markers",
                marker=dict(size=8, color=SERIES, line=dict(width=1, color="white")),
                hovertemplate="outlier (Tukey): %{x:.4g}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    # Each label sits on the side facing away from the other line so the two
    # never collide, however close mean and median are.
    mean_side, median_side = ("right", "left") if mean >= median else ("left", "right")
    for value, label, dash, color, side in (
        (mean, "rata-rata", "dash", SERIES_2, mean_side),
        (median, "median", "dot", INK_SECONDARY, median_side),
    ):
        fig.add_vline(
            x=value, line=dict(color=color, width=1.5, dash=dash), row=1, col=1,
            annotation=dict(text=f"{label} {value:.4g}", font=dict(size=11, color=INK_SECONDARY)),
            annotation_position=f"top {side}",
        )
    fig.update_yaxes(title_text="frekuensi", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text=name, row=2, col=1)
    return _base_layout(fig, height=380)


def correlation_heatmap(result: CorrelationResult) -> go.Figure:
    """Strict lower-triangle heatmap on a fixed [-1, 1] diverging scale.

    The diagonal (always 1) and the mirrored upper triangle carry no
    information, so rows start at the 2nd variable and columns stop at the
    second-to-last.
    """
    cols = list(result.r.columns)
    k = len(cols)
    z = result.r.to_numpy(copy=True)
    z[np.triu_indices(k)] = np.nan
    z, n = z[1:, :-1], result.n.to_numpy()[1:, :-1]

    fig = go.Figure(
        go.Heatmap(
            z=z, x=cols[:-1], y=cols[1:],
            zmin=-1, zmax=1, zmid=0, colorscale=DIVERGING,
            xgap=2, ygap=2, hoverongaps=False,
            customdata=n,
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<br>n = %{customdata}<extra></extra>",
            texttemplate="%{z:.2f}" if k <= MAX_ANNOTATED_CELLS else None,
            colorbar=dict(title=dict(text="r"), thickness=12, tickvals=[-1, -0.5, 0, 0.5, 1]),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _base_layout(fig, height=max(260, 56 * (k - 1) + 120))


def scatter_figure(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Raw scatter of two variables over their complete pairs."""
    data = df[[x, y]].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    fig = go.Figure(
        go.Scatter(
            x=data[x], y=data[y], mode="markers",
            marker=dict(size=8, color=SERIES, opacity=0.7, line=dict(width=1, color="white")),
            hovertemplate=f"{x} = %{{x:.4g}}<br>{y} = %{{y:.4g}}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return _base_layout(fig, height=380)


# --- Regression diagnostics ---------------------------------------------------


def _scatter_trace(**kwargs) -> go.Scatter | go.Scattergl:
    cls = go.Scattergl if len(kwargs["x"]) > WEBGL_THRESHOLD else go.Scatter
    return cls(**kwargs)


def _points(x, y, name: str, hover: str, color=SERIES) -> go.Scatter | go.Scattergl:
    return _scatter_trace(
        x=x, y=y, mode="markers", name=name,
        marker=dict(size=7, color=color, opacity=0.7, line=dict(width=1, color="white")),
        customdata=np.asarray(x.index if hasattr(x, "index") else np.arange(len(x)), dtype=str),
        hovertemplate=hover + "<extra></extra>",
    )


def _smoother(curve: np.ndarray | None) -> list[go.Scatter]:
    if curve is None:
        return []
    return [go.Scatter(x=curve[:, 0], y=curve[:, 1], mode="lines", name="LOWESS",
                       line=dict(color=SERIES_2, width=2), hoverinfo="skip")]


def _with_legend(fig: go.Figure, height: int = 320) -> go.Figure:
    _base_layout(fig, height)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        margin=dict(t=36),
    )
    return fig


def residuals_vs_fitted(diag: Diagnostics) -> go.Figure:
    obs = diag.observations
    # Points beyond the externally studentized residual threshold are drawn in
    # the flag colour; they are the same rows listed under "Observasi berpengaruh".
    flagged = obs["student_resid"].abs() > diag.thresholds["student_resid"]
    fig = go.Figure([
        _points(obs["fitted"], obs["resid"], "Residual",
                "baris %{customdata}<br>ŷ = %{x:.4g}<br>residual = %{y:.4g}",
                color=np.where(flagged, FLAG, SERIES).tolist()),
        *_smoother(diag.lowess),
    ])
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.update_xaxes(title_text="Nilai prediksi (ŷ)")
    fig.update_yaxes(title_text="Residual")
    return _with_legend(fig)


def scale_location(diag: Diagnostics) -> go.Figure:
    obs = diag.observations
    fig = go.Figure([
        _points(obs["fitted"], obs["sqrt_abs_std_resid"], "√|residual terstandar|",
                "baris %{customdata}<br>ŷ = %{x:.4g}<br>√|r| = %{y:.3f}"),
        *_smoother(diag.scale_lowess),
    ])
    fig.update_xaxes(title_text="Nilai prediksi (ŷ)")
    fig.update_yaxes(title_text="√|residual terstandar|", rangemode="tozero")
    return _with_legend(fig)


def qq_plot(diag: Diagnostics) -> go.Figure:
    qq = diag.qq
    intercept, slope = diag.qq_line
    x_line = np.array([qq["theoretical"].min(), qq["theoretical"].max()])
    fig = go.Figure([
        _scatter_trace(
            x=qq["theoretical"], y=qq["sample"], mode="markers", name="Residual terstandar",
            marker=dict(size=7, color=SERIES, opacity=0.7, line=dict(width=1, color="white")),
            hovertemplate="kuantil teoretis = %{x:.3f}<br>kuantil sampel = %{y:.3f}<extra></extra>",
        ),
        go.Scatter(x=x_line, y=intercept + slope * x_line, mode="lines", name="Garis acuan (Q1–Q3)",
                   line=dict(color=SERIES_2, width=2), hoverinfo="skip"),
    ])
    fig.update_xaxes(title_text="Kuantil teoretis N(0, 1)")
    fig.update_yaxes(title_text="Kuantil residual terstandar")
    return _with_legend(fig)


def residual_histogram(diag: Diagnostics) -> go.Figure:
    r = diag.observations["std_resid"].replace([np.inf, -np.inf], np.nan).dropna()
    edges = histogram_bins(r)
    counts, _ = np.histogram(r, bins=edges)
    widths = np.diff(edges)
    grid = np.linspace(edges[0], edges[-1], 200)
    # N(0, 1) density scaled to expected counts per bin of the (median) width.
    expected = stats.norm.pdf(grid) * len(r) * np.median(widths)
    fig = go.Figure([
        go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=widths * 0.94, name="Residual terstandar",
               marker=dict(color=SERIES, cornerradius=3),
               hovertemplate="frekuensi = %{y}<extra></extra>"),
        go.Scatter(x=grid, y=expected, mode="lines", name="Kurva N(0, 1)",
                   line=dict(color=SERIES_2, width=2), hoverinfo="skip"),
    ])
    fig.update_xaxes(title_text="Residual terstandar")
    fig.update_yaxes(title_text="frekuensi")
    return _with_legend(fig)


def cooks_distance(diag: Diagnostics) -> go.Figure:
    obs = diag.observations
    position = np.arange(1, len(obs) + 1)
    fig = go.Figure(
        _scatter_trace(
            x=position, y=obs["cooks_d"], mode="markers", name="Cook's D",
            marker=dict(size=7, color=SERIES, line=dict(width=1, color="white")),
            customdata=np.asarray(obs.index, dtype=str),
            hovertemplate="baris %{customdata}<br>Cook's D = %{y:.4f}<extra></extra>",
        )
    )
    threshold = diag.thresholds["cooks_d"]
    fig.add_hline(y=threshold, line=dict(color=FLAG, width=1, dash="dash"),
                  annotation=dict(text=f"4/n = {threshold:.3g}", font=dict(size=11, color=INK_SECONDARY)),
                  annotation_position="top right")
    fig.update_xaxes(title_text="Urutan observasi")
    fig.update_yaxes(title_text="Cook's distance", rangemode="tozero")
    return _base_layout(fig, height=300)


# --- Model comparison ---------------------------------------------------------


def cv_error_figure(result: ComparisonResult) -> go.Figure:
    """Per-split RMSE (dots) with the mean ± SD across splits (ink bar) per model."""
    order = [BASELINE, *result.names]
    colors = {BASELINE: NEUTRAL, result.names[0]: SERIES, result.names[1]: SERIES_2}
    rng = np.random.default_rng(0)  # jitter only; does not touch the data
    fig = go.Figure()
    for i, name in enumerate(order):
        rmse = result.cv_folds.loc[result.cv_folds["model"] == name, "rmse"].to_numpy()
        fig.add_trace(go.Scatter(
            x=rmse, y=i + rng.uniform(-0.18, 0.18, len(rmse)), mode="markers", name=name,
            marker=dict(size=8, color=colors[name], opacity=0.55, line=dict(width=1, color="white")),
            hovertemplate=f"{name}<br>RMSE fold = %{{x:.4g}}<extra></extra>",
        ))
        mean, sd = result.cv_summary.loc[name, ["rmse_mean", "rmse_std"]]
        fig.add_trace(go.Scatter(
            x=[mean], y=[i], mode="markers", name=f"{name}: rata-rata ± SD",
            marker=dict(size=12, color=INK_SECONDARY, symbol="line-ns", line=dict(width=3, color=INK_SECONDARY)),
            error_x=dict(type="data", array=[sd], color=INK_SECONDARY, thickness=2, width=0),
            hovertemplate=f"{name}<br>rata-rata RMSE = {mean:.4g}<br>SD = {sd:.4g}<extra></extra>",
            showlegend=False,
        ))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=order, autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text=f"RMSE pada fold uji (satuan {result.target})")
    return _base_layout(fig, height=110 + 70 * len(order))


# --- Compact and regression-result figures -----------------------------------


def mini_histogram(values: pd.Series, name: str) -> go.Figure:
    """Small histogram with a dashed mean line, for the small-multiples grid."""
    x = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    edges = histogram_bins(x)
    counts, _ = np.histogram(x, bins=edges)
    fig = go.Figure(
        go.Bar(
            x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges) * 0.92,
            marker=dict(color=SERIES, cornerradius=2),
            customdata=np.column_stack([edges[:-1], edges[1:]]),
            hovertemplate="[%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>frekuensi = %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=x.mean(), line=dict(color=SERIES_2, width=1.5, dash="dash"))
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text=None, nticks=4)
    fig = _base_layout(fig, height=200)
    fig.update_layout(margin=dict(l=4, r=4, t=8, b=4))
    return fig


def standardized_effects_figure(effects: pd.DataFrame, conf_level: float) -> go.Figure:
    """Dot-and-interval plot of standardised slopes (β) with their intervals.

    ``effects`` comes from datalens.regression.standardized_coefficients, so
    the plotted values are exactly the ones in its table.
    """
    order = effects.reindex(effects["beta"].abs().sort_values().index)
    fig = go.Figure(
        go.Scatter(
            x=order["beta"], y=list(order.index), mode="markers",
            marker=dict(size=11, color=SERIES_2, line=dict(width=1, color="white")),
            error_x=dict(
                type="data", symmetric=False,
                array=order["ci_upper"] - order["beta"], arrayminus=order["beta"] - order["ci_lower"],
                color=INK_SECONDARY, thickness=2, width=5,
            ),
            customdata=order[["ci_lower", "ci_upper"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>β = %{x:.3f}<br>"
                f"CI {conf_level:.0%}: " + "[%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.update_xaxes(title_text="β (perubahan SD y per 1 SD x)")
    fig.update_yaxes(showgrid=False, tickfont=dict(family=MONO, size=12, color="#0F1D33"))
    return _base_layout(fig, height=max(220, 60 * len(order) + 90))


def actual_vs_predicted(actual: pd.Series, predicted: pd.Series, target: str) -> go.Figure:
    """Observed y against fitted ŷ with the y = ŷ reference line."""
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    fig = go.Figure([
        _scatter_trace(
            x=predicted, y=actual, mode="markers",
            marker=dict(size=7, color=SERIES, opacity=0.7, line=dict(width=1, color="white")),
            customdata=np.asarray(actual.index, dtype=str),
            hovertemplate="baris %{customdata}<br>ŷ = %{x:.4g}<br>y = %{y:.4g}<extra></extra>",
        ),
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=SERIES_2, width=1.5, dash="dash"),
                   hoverinfo="skip"),
    ])
    fig.update_xaxes(title_text="Nilai prediksi (ŷ)")
    fig.update_yaxes(title_text=f"Nilai aktual ({target})")
    return _base_layout(fig, height=300)
