"""Step 02 'Regresi': model specification, OLS results and model comparison.

The specification widgets keep their values in session state (keys
``reg_target``, ``reg_predictors``, ``reg_conf``). Results are marked stale
whenever the specification on screen differs from the one that was fitted,
so the numbers shown never silently disagree with the inputs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.comparison import formula
from datalens.data import numeric_columns
from datalens.formatting import fmt_num, fmt_pvalue
from datalens.plots import actual_vs_predicted, standardized_effects_figure
from datalens.regression import (
    INTERCEPT,
    ModelSpecError,
    RegressionResult,
    fit_ols,
    prepare_model_data,
    standardized_coefficients,
)
from views import comparison, ui
from views.components import p_label, render_safely, step_header

CONF_LEVELS = [0.90, 0.95, 0.99]
DEFAULT_CONF = 0.95


def _spec() -> tuple[str, tuple[str, ...], float]:
    ss = st.session_state
    return ss.reg_target, tuple(ss.reg_predictors), ss.reg_conf or DEFAULT_CONF


def _result_spec(r: RegressionResult) -> tuple[str, tuple[str, ...], float]:
    return r.target, r.predictors, r.conf_level


def _init_spec_state(numeric: list[str], result: RegressionResult | None) -> None:
    """Seed the widgets from the fitted model (or defaults) and drop stale options."""
    ss = st.session_state
    if ss.get("reg_target") not in numeric:
        ss.reg_target = result.target if result and result.target in numeric else numeric[-1]
    if "reg_predictors" not in ss:
        ss.reg_predictors = list(result.predictors) if result else []
    ss.reg_predictors = [p for p in ss.reg_predictors if p in numeric and p != ss.reg_target]
    if ss.get("reg_conf") not in CONF_LEVELS:
        ss.reg_conf = result.conf_level if result and result.conf_level in CONF_LEVELS else DEFAULT_CONF


def _fit(df: pd.DataFrame) -> None:
    target, predictors, conf = _spec()
    try:
        st.session_state.result = fit_ols(df, target, list(predictors), conf_level=conf)
        st.session_state.fit_error = None
    except ModelSpecError as exc:
        st.session_state.result = None
        st.session_state.fit_error = ((target, predictors, conf), str(exc))


def render(df: pd.DataFrame) -> None:
    numeric = numeric_columns(df)
    result: RegressionResult | None = st.session_state.get("result")
    if len(numeric) >= 2:
        _init_spec_state(numeric, result)
    fresh = result is not None and len(numeric) >= 2 and _spec() == _result_spec(result)

    step_header(
        "regresi",
        "Studio Regresi",
        "Tentukan model regresi linear (OLS), estimasi, lalu baca estimasinya bersama ketidakpastiannya.",
        cta=("Periksa diagnostik", "diagnostik") if fresh else None,
    )
    if len(numeric) < 2:
        st.warning("Regresi membutuhkan minimal dua kolom numerik (satu dependen, satu prediktor).")
        return

    tab_model, tab_cmp = st.tabs(["Model", "Perbandingan model"])
    with tab_model:
        spec_col, result_col = st.columns([1, 2.4], gap="large")
        with spec_col:
            _spec_panel(df, numeric, fresh)
        with result_col:
            _results_area(result, fresh)
    with tab_cmp:
        render_safely(comparison.render, df)


def _spec_panel(df: pd.DataFrame, numeric: list[str], fresh: bool) -> None:
    with st.container(border=True, key="panel-spec"):
        ui.panel_title("Spesifikasi model", "OLS dengan intersep · hanya kolom numerik")
        target = st.selectbox("Variabel dependen (Y)", numeric, key="reg_target", persist_state="session")
        predictors = st.multiselect(
            "Prediktor (X)",
            [c for c in numeric if c != target],
            key="reg_predictors",
            persist_state="session",
            placeholder="Pilih satu atau lebih prediktor",
            help="Satu prediktor = regresi sederhana; lebih dari satu = regresi berganda.",
        )
        st.segmented_control(
            "Tingkat kepercayaan", CONF_LEVELS, key="reg_conf", required=True, persist_state="session",
            format_func=lambda v: f"{v:.0%}", width="stretch",
        )
        if predictors:
            st.html(
                '<div class="dl-mono" style="background:#F7F6F2;border:1px solid var(--dl-line);border-radius:4px;'
                f'padding:9px 11px;font-size:12px;line-height:1.6;word-break:break-word">'
                f"{ui.esc(formula(target, predictors))}</div>"
            )
            try:
                used = len(prepare_model_data(df, target, predictors).y)
            except ModelSpecError:
                used = None
            if used is not None:
                dropped = len(df) - used
                st.html(
                    '<div style="display:flex;justify-content:space-between;font-size:12.5px;color:var(--dl-ink2)">'
                    f'<span>Baris dipakai</span><span class="dl-mono">{used:,} dari {len(df):,}'
                    + (f' <span style="color:var(--dl-amber)">(−{dropped:,})</span>' if dropped else "")
                    + "</span></div>"
                )
        st.button(
            "Estimasi model",
            type="secondary" if fresh else "primary",
            icon=":material/refresh:" if fresh else ":material/play_arrow:",
            width="stretch",
            on_click=_fit,
            args=(df,),
        )
        if fresh:
            st.html('<span class="dl-status ok">' + ui.ICON_OK + "Hasil sesuai spesifikasi ini</span>")


def _results_area(result: RegressionResult | None, fresh: bool) -> None:
    error = st.session_state.get("fit_error")
    if error and error[0] == _spec():
        st.error(error[1], icon=":material/error:")
    if result is None:
        if not error:
            ui.empty_state(
                "Belum ada model",
                "Pilih variabel dependen dan minimal satu prediktor, lalu klik Estimasi model.",
                eyebrow="Hasil",
            )
        return
    if not fresh:
        st.warning(
            "**Spesifikasi berubah.** Hasil di bawah milik model sebelumnya "
            f"(`{formula(result.target, result.predictors)}`). Klik **Estimasi model** untuk memperbarui.",
            icon=":material/sync_problem:",
        )
        with st.container(key="stale-results"):
            render_result(result)
        return
    render_result(result)


def _equation_html(r: RegressionResult) -> str:
    est = r.coefficients["estimate"]
    b0 = est[INTERCEPT]
    parts = [f'<span class="dl-mono">{"−" if b0 < 0 else ""}{fmt_num(abs(b0))}</span>']
    for name in r.predictors:
        b = est[name]
        sign = "−" if b < 0 else "+"
        parts.append(f'{sign} <span class="dl-mono pos">{fmt_num(abs(b))}</span> · <i>{ui.esc(name)}</i>')
    return '<p class="dl-eq"><i>ŷ</i> = ' + " ".join(parts) + "</p>"


def _latex(r: RegressionResult) -> str:
    est = r.coefficients["estimate"]
    out = r"\hat{y} = " + fmt_num(est[INTERCEPT])
    for name in r.predictors:
        b = est[name]
        out += f" {'-' if b < 0 else '+'} {fmt_num(abs(b))} \\cdot \\text{{{name.replace('_', chr(92) + '_')}}}"
    return out


def _interpretation(r: RegressionResult, effects: pd.DataFrame) -> str:
    name = effects["beta"].abs().idxmax()
    row = r.coefficients.loc[name]
    held = "Dengan prediktor lain dibuat tetap, kenaikan" if len(r.predictors) > 1 else "Kenaikan"
    direction = "lebih tinggi" if row.estimate >= 0 else "lebih rendah"
    return (
        f"{held} 1 unit <span class='dl-mono'>{ui.esc(name)}</span> berasosiasi dengan "
        f"<b style='color:var(--dl-ink);font-weight:600'>{ui.esc(r.target)} rata-rata {fmt_num(abs(row.estimate))} "
        f"unit {direction}</b> (CI {r.conf_level:.0%}: {fmt_num(row.ci_lower)} sampai {fmt_num(row.ci_upper)}). "
        "Prediktor ini memiliki efek terstandar terbesar dalam model."
    )


def render_result(r: RegressionResult) -> None:
    fitted = r.fitted
    n_total = r.n_obs + r.n_dropped
    ui.stat_strip(
        [
            ui.Stat("R²", fmt_num(r.r_squared, 3), f"{r.r_squared:.0%} variasi {r.target} dijelaskan",
                    bar=r.r_squared),
            ui.Stat("Adjusted R²", fmt_num(r.adj_r_squared, 3), f"Dikoreksi untuk {len(r.predictors)} prediktor",
                    bar=max(r.adj_r_squared, 0.0)),
            ui.Stat("Observasi", f"{r.n_obs:,}",
                    f"dari {n_total:,} baris · {r.n_dropped:,} dikeluarkan" if r.n_dropped else "semua baris lengkap",
                    bar=r.n_obs / n_total if n_total else None),
        ],
        big=True,
        footer=[
            f'Uji F({r.df_model}, {r.df_resid}) <span class="dl-mono">= {ui.esc(fmt_num(r.f_statistic))}</span>, '
            f'<span class="dl-mono">p {ui.esc(p_label(r.f_pvalue))}</span>',
            f'Galat baku residual <span class="dl-mono">{ui.esc(fmt_num(r.residual_std_error))}</span>',
            f'AIC <span class="dl-mono">{ui.esc(fmt_num(fitted.aic, 5))}</span>',
            f'BIC <span class="dl-mono">{ui.esc(fmt_num(fitted.bic, 5))}</span>',
        ],
    )

    effects = standardized_coefficients(r)
    with st.container(border=True, key="panel-equation"):
        st.html(
            '<span class="dl-eyebrow">Persamaan model</span>' + _equation_html(r)
            + f'<div class="dl-note">{_interpretation(r, effects)}</div>'
        )
        with st.expander("Salin persamaan (teks atau LaTeX)", icon=":material/content_copy:"):
            st.code(r.equation, language=None, wrap_lines=True)
            st.code(_latex(r), language="latex", wrap_lines=True)
        st.caption(f"ŷ adalah nilai prediksi rata-rata **{r.target}** untuk kombinasi nilai prediktor tertentu.")

    for w in r.warnings:
        st.warning(w, icon=":material/warning:")

    with st.container(border=True, key="panel-coefficients"):
        head, action = st.columns([3, 1], vertical_alignment="center")
        with head:
            st.subheader("Tabel koefisien")
        with action, st.container(horizontal=True, horizontal_alignment="right"):
            st.download_button(
                "CSV", data=r.coefficients.to_csv().encode("utf-8"), file_name="datalens_koefisien.csv",
                mime="text/csv", icon=":material/download:", type="tertiary", key="coef_csv",
            )
        pct = f"{r.conf_level:.0%}"
        table = pd.DataFrame(
            {
                "Estimasi": r.coefficients["estimate"].map(fmt_num),
                "Galat baku": r.coefficients["std_error"].map(fmt_num),
                "t": r.coefficients["t_value"].map(lambda v: fmt_num(v, 3)),
                "p-value": r.coefficients["p_value"].map(fmt_pvalue),
                f"CI {pct} bawah": r.coefficients["ci_lower"].map(fmt_num),
                f"CI {pct} atas": r.coefficients["ci_upper"].map(fmt_num),
            }
        )
        table.index.name = "Term"
        st.dataframe(table, width="stretch")
        st.caption(
            f"Galat baku klasik (non-robust) · uji t dua sisi dengan {r.df_resid} derajat bebas · "
            "p-value di bawah 0.001 ditulis < 0.001."
        )

    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True, key="panel-effects", height="stretch"):
        ui.panel_title("Efek terstandar", f"β per 1 SD perubahan prediktor, dengan CI {pct} · sebanding antarsatuan")
        st.plotly_chart(standardized_effects_figure(effects, r.conf_level), width="stretch")
    with right, st.container(border=True, key="panel-avp", height="stretch"):
        ui.panel_title("Aktual vs prediksi", "Titik pada garis putus-putus diprediksi dengan tepat")
        actual = fitted.fittedvalues + fitted.resid
        st.plotly_chart(actual_vs_predicted(actual, fitted.fittedvalues, r.target), width="stretch")

    with st.expander("Statistik model lengkap", icon=":material/table_rows:"):
        rows = [
            ("Variabel dependen", r.target),
            ("Metode", "Kuadrat terkecil (OLS)"),
            ("Kovarians", "klasik (non-robust)"),
            ("df model / residual", f"{r.df_model} / {r.df_resid}"),
            ("Statistik F", fmt_num(r.f_statistic, 6)),
            ("Prob (F)", f"{r.f_pvalue:.3g}"),
            ("Log-likelihood", fmt_num(fitted.llf, 6)),
            ("AIC", fmt_num(fitted.aic, 6)),
            ("BIC", fmt_num(fitted.bic, 6)),
            ("Galat baku residual", fmt_num(r.residual_std_error, 6)),
            ("Baris dikeluarkan (hilang / tak hingga)", f"{r.n_dropped_missing} / {r.n_dropped_nonfinite}"),
            ("Condition number (statsmodels)", f"{fitted.condition_number:.3g}"),
        ]
        cells = "".join(
            '<div style="display:flex;justify-content:space-between;gap:12px;padding:8px 0;'
            f'border-bottom:1px solid var(--dl-line)"><span style="color:var(--dl-muted)">{ui.esc(k)}</span>'
            f'<span class="dl-mono">{ui.esc(v)}</span></div>'
            for k, v in rows
        )
        st.html(
            '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));column-gap:32px;'
            f'font-size:13px">{cells}</div>'
        )

    with st.expander("Cara membaca hasil ini", icon=":material/menu_book:"):
        st.markdown(
            f"""
**Koefisien (slope).** Perubahan rata-rata yang diprediksi pada *{r.target}* untuk kenaikan satu unit
prediktor, **dengan prediktor lain dalam model dibuat tetap**. Ini adalah *asosiasi* dalam data, bukan
efek kausal: tanpa desain eksperimen atau strategi identifikasi kausal, variabel perancu yang tidak
diamati dapat menjelaskan hubungan tersebut.

**Intersep.** Nilai prediksi *{r.target}* ketika semua prediktor bernilai nol. Sering tidak bermakna
secara substantif jika nol berada di luar rentang data.

**Galat baku & interval kepercayaan {pct}.** Mengukur ketidakpastian estimasi akibat variasi sampel.
Prosedur interval ini, jika diulang pada banyak sampel, mencakup parameter sebenarnya {pct} dari waktu
— *dengan syarat asumsi model terpenuhi* (linearitas, independensi, varians galat konstan; normalitas
galat penting terutama untuk n kecil).

**p-value.** Probabilitas memperoleh statistik t setidaknya seekstrem ini *jika* koefisien populasi
sama dengan nol dan asumsi model benar. p-value **bukan** probabilitas bahwa hipotesis nol benar.
Signifikansi statistik juga **bukan** kepentingan praktis: dengan n besar, efek yang sangat kecil pun
bisa signifikan. Nilai pentingnya efek dinilai dari besar koefisien dan interval kepercayaannya dalam
satuan aslinya.

**Efek terstandar (β).** Koefisien yang dikalikan SD prediktor dan dibagi SD *{r.target}* pada sampel
estimasi. Berguna untuk membandingkan besar efek antar-prediktor yang satuannya berbeda; tidak mengubah
uji maupun kesimpulan.

**R².** Proporsi variasi *{r.target}* dalam sampel yang dijelaskan model. R² selalu naik (atau tetap)
saat prediktor ditambahkan, sehingga tidak cocok untuk membandingkan model dengan jumlah prediktor
berbeda. R² tinggi tidak membuktikan model benar atau kausal, dan tidak menjamin akurasi prediksi pada
data baru.

**Adjusted R².** R² yang dikoreksi untuk jumlah prediktor; dapat turun jika prediktor tambahan tidak
cukup menambah daya jelas.

**Galat baku residual.** Perkiraan simpangan baku galat, dalam satuan *{r.target}*: seberapa jauh
observasi biasanya menyimpang dari garis regresi.

**Uji F.** Menguji apakah *semua* slope sama dengan nol secara bersamaan.
"""
        )
    st.caption(
        f"Metode: OLS (statsmodels), galat baku klasik (non-robust). Nilai hilang: complete-case — "
        f"{r.n_dropped_missing} baris dengan nilai hilang dan {r.n_dropped_nonfinite} baris dengan nilai "
        "tak hingga pada variabel model dikeluarkan. Variabel tidak distandardisasi dalam estimasi."
    )
