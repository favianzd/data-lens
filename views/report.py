"""Step 04 'Laporan': on-screen analysis summary plus the downloadable HTML report.

The summary reuses the rule-based sentences from datalens.report, so what is
shown here matches the exported file.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from datalens.comparison import ComparisonResult, formula
from datalens.diagnostics import Diagnostics, DiagnosticsError, diagnose
from datalens.formatting import fmt_num, fmt_pvalue
from datalens.regression import RegressionResult
from datalens.report import (
    ReportInputs,
    build_report,
    coefficient_findings,
    fit_finding,
    limitations,
    now_wib,
    report_filename,
)
from views import ui
from views.components import step_header
from views.diagnostics import assumption_checks
from views.state import is_synthetic

MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
          "November", "Desember"]


def _date_id(when: datetime) -> str:
    return f"{when.day} {MONTHS[when.month - 1]} {when.year}"


def _summary_html(r: RegressionResult, diag: Diagnostics | None, cmp: ComparisonResult | None) -> str:
    name = st.session_state.get("dataset_name", "dataset")
    pct = f"{r.conf_level:.0%}"
    findings = [*coefficient_findings(r), fit_finding(r)]
    items = "".join(f'<li><span class="n">{i}</span><span>{ui.esc(f)}</span></li>' for i, f in enumerate(findings, 1))

    rows = "".join(
        f'<tr><td class="dl-mono">{ui.esc(term)}</td><td class="num">{fmt_num(c.estimate)}</td>'
        f'<td class="num">{fmt_num(c.std_error)}</td><td class="num">[{fmt_num(c.ci_lower)}, {fmt_num(c.ci_upper)}]</td>'
        f'<td class="num">{fmt_pvalue(c.p_value)}</td></tr>'
        for term, c in r.coefficients.iterrows()
    )
    table = (
        f"<table><thead><tr><th>Term</th><th class='num'>Estimasi</th><th class='num'>Galat baku</th>"
        f"<th class='num'>CI {pct}</th><th class='num'>p</th></tr></thead><tbody>{rows}</tbody></table>"
        f'<p style="font-size:12px;color:var(--dl-muted);margin:8px 0 0">Tabel 1. Estimasi OLS untuk '
        f"{ui.esc(r.target)} (n = {r.n_obs:,}). Galat baku klasik.</p>"
    )

    if diag is None:
        checks = "<p>Diagnostik residual tidak tersedia untuk model ini.</p>"
    else:
        cells = "".join(
            f'<tr><td>{ui.esc(c.title)}</td><td><span class="dl-status {c.tone}">{ui.STATUS[c.tone][0]}'
            f"{ui.STATUS[c.tone][1]}</span></td><td class='num'>{ui.esc(c.detail)}</td></tr>"
            for c in assumption_checks(diag)
        )
        checks = f"<table><tbody>{cells}</tbody></table>"

    limits = list(limitations(r, diag, cmp))
    if is_synthetic():
        limits.insert(0, "Data sintetis: hasil menunjukkan cara kerja metode, bukan perilaku nyata.")
    limit_items = "".join(f"<li>{ui.esc(x)}</li>" for x in limits)
    chip = ' <span class="dl-chip dl-chip-amber">Sintetis</span>' if is_synthetic() else ""
    predictors = ", ".join(r.predictors)

    return f"""
<article class="dl-paper">
  <header>
    <span class="dl-eyebrow">Ringkasan DataLens · {_date_id(now_wib())}</span>
    <h2>Regresi OLS untuk {ui.esc(r.target)}</h2>
    <p class="lede">Regresi kuadrat terkecil {ui.esc(r.target)} pada {ui.esc(predictors)}.</p>
    <dl>
      <div><dt>Dataset</dt><dd>{ui.esc(name)}{chip}</dd></div>
      <div><dt>Observasi</dt><dd>{r.n_obs:,} dari {r.n_obs + r.n_dropped:,}</dd></div>
      <div><dt>Estimator</dt><dd>OLS · galat baku klasik</dd></div>
      <div><dt>Kepercayaan</dt><dd>{pct}</dd></div>
    </dl>
  </header>
  <h3 style="margin-top:0">Temuan utama</h3>
  <ol class="find">{items}</ol>
  <h3>Bukti pendukung</h3>
  {table}
  <h3>Diagnostik</h3>
  {checks}
  <h3>Keterbatasan</h3>
  <ul>{limit_items}</ul>
  <footer>Dibuat oleh DataLens · semua nilai dihitung dari file yang dimuat ·
    <span class="dl-mono">{ui.esc(formula(r.target, r.predictors))}</span></footer>
</article>
"""


def _build(df: pd.DataFrame, result: RegressionResult, comparison: ComparisonResult | None,
           include_fig: bool, key: tuple) -> None:
    now = now_wib()
    name = st.session_state.get("dataset_name", "dataset")
    with st.spinner("Menyusun laporan…"):
        html = build_report(
            ReportInputs(dataset_name=name, df=df, result=result, comparison=comparison,
                         include_figures=include_fig, generated_at=now)
        )
    st.session_state.report = {"key": key, "html": html, "name": report_filename(name, now)}


def render(df: pd.DataFrame) -> None:
    result: RegressionResult = st.session_state["result"]
    comparison: ComparisonResult | None = st.session_state.get("comparison")
    step_header(
        "laporan",
        "Ringkasan Analisis",
        "Apa yang ditunjukkan model, bukti di baliknya, dan batas-batasnya. Tinjau ringkasan, lalu ekspor laporannya.",
    )
    try:
        diag = diagnose(result)
    except DiagnosticsError:
        diag = None

    left, right = st.columns([2.3, 1], gap="large")
    with right, st.container(border=True, key="panel-export"):
        ui.panel_title("Ekspor laporan", "Satu file HTML mandiri dari hasil yang tampil di aplikasi")
        include_cmp = st.checkbox(
            "Sertakan perbandingan model",
            value=comparison is not None,
            disabled=comparison is None,
            help="Jalankan perbandingan di langkah Regresi (tab Perbandingan model) untuk mengaktifkan opsi ini.",
        )
        include_fig = st.checkbox(
            "Sertakan grafik interaktif",
            value=True,
            help="Menyematkan plotly.js (~4,8 MB) agar grafik tetap berfungsi tanpa internet. Matikan untuk file kecil.",
        )
        if include_cmp and comparison is not None and comparison.target != result.target:
            st.caption(
                f"Catatan: perbandingan model memakai variabel dependen `{comparison.target}`, berbeda dari model utama."
            )

        key = (id(result), id(comparison) if include_cmp else None, include_fig)
        report = st.session_state.get("report")
        ready = bool(report and report["key"] == key)
        st.button(
            "Buat laporan", type="secondary" if ready else "primary", icon=":material/description:", width="stretch",
            on_click=_build, args=(df, result, comparison if include_cmp else None, include_fig, key),
        )
        report = st.session_state.get("report")
        ready = bool(report and report["key"] == key)
        if ready:
            size_mb = len(report["html"].encode("utf-8")) / 1e6
            st.success(f"Laporan siap · `{report['name']}` · {size_mb:.1f} MB", icon=":material/check_circle:")
            st.download_button(
                "Unduh laporan HTML",
                data=report["html"],
                file_name=report["name"],
                mime="text/html",
                icon=":material/download:",
                type="primary",
                width="stretch",
            )
        elif report:
            st.caption("Model atau pengaturan berubah sejak laporan terakhir dibuat. Klik **Buat laporan** lagi.")
        st.divider()
        st.download_button(
            "Unduh tabel koefisien (CSV)",
            data=result.coefficients.to_csv().encode("utf-8"),
            file_name="datalens_koefisien.csv",
            mime="text/csv",
            icon=":material/table_view:",
            type="tertiary",
        )

    with left:
        st.html(_summary_html(result, diag, comparison if include_cmp else None))

    if ready:
        with st.expander("Pratinjau file laporan HTML", icon=":material/visibility:"):
            components.html(report["html"], height=900, scrolling=True)
