"""Step 03 'Diagnostik': residual plots, formal assumption tests, VIF and influence."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.comparison import formula
from datalens.diagnostics import (
    CONDITION_HIGH,
    VIF_MODERATE,
    AssumptionTest,
    Diagnostics,
    DiagnosticsError,
    diagnose,
)
from datalens.formatting import fmt_num, fmt_pvalue
from datalens.plots import cooks_distance, qq_plot, residual_histogram, residuals_vs_fitted, scale_location
from datalens.regression import RegressionResult
from views import ui
from views.components import metric_row, step_header

MAX_INFLUENTIAL_ROWS = 20

LIMITATIONS = [
    "Membaca plot memerlukan penilaian. Dua analis bisa menafsirkan pola residual yang sama secara berbeda.",
    "Uji formal bergantung pada ukuran sampel. Pada n besar, penyimpangan kecil menjadi “signifikan”; pada n "
    "kecil, masalah nyata bisa lolos.",
    "Lolos semua pemeriksaan tidak berarti model sudah benar. Variabel yang terlewat dan galat pengukuran "
    "tidak terlihat di sini.",
    "VIF hanya mendeteksi tumpang tindih linear di antara prediktor yang Anda pilih.",
    "Tidak ada pemeriksaan di sini yang mengubah asosiasi pada data observasional menjadi efek kausal.",
]


def _test_check(title: str, t: AssumptionTest) -> ui.Check:
    if not t.available:
        return ui.Check(title, t.note or "Uji tidak dijalankan", "na")
    name = t.test.split(" (")[0]
    tone = "review" if t.p_value < t.alpha else "ok"
    return ui.Check(title, f"{name} p {'< 0.001' if t.p_value < 0.001 else '= ' + fmt_pvalue(t.p_value)}", tone)


def assumption_checks(diag: Diagnostics) -> list[ui.Check]:
    bp, normality, reset = diag.tests
    checks = [
        _test_check("Linearitas", reset),
        _test_check("Varians konstan", bp),
        _test_check("Normalitas galat", normality),
    ]
    if diag.vif is None:
        checks.append(ui.Check("Multikolinearitas", "Tidak relevan untuk satu prediktor", "na"))
    else:
        top = float(diag.vif["vif"].max())
        checks.append(ui.Check("Multikolinearitas", f"VIF maksimum {fmt_num(top, 3)}",
                               "review" if top >= VIF_MODERATE else "ok"))
    n_flag = len(diag.influential)
    checks.append(ui.Check("Observasi berpengaruh",
                           f"{n_flag} observasi melewati ambang" if n_flag else "Tidak ada yang melewati ambang",
                           "review" if n_flag else "ok"))
    return checks


def render(df: pd.DataFrame) -> None:
    result: RegressionResult = st.session_state["result"]
    step_header(
        "diagnostik",
        "Diagnostik Model",
        "Periksa apakah asumsi di balik estimasi OLS masuk akal. Plot adalah bukti utama; uji formal sebagai pendukung.",
        cta=("Susun laporan", "laporan"),
    )
    st.caption(f"Model: `{formula(result.target, result.predictors)}` · n = {result.n_obs:,}")
    try:
        diag = diagnose(result)
    except DiagnosticsError as exc:
        st.warning(str(exc))
        return

    ui.check_strip(assumption_checks(diag))
    st.caption(
        "“Tidak ada temuan” berarti uji atau ambang praktis tidak menemukan bukti pelanggaran (α = 0.05), bukan "
        "bukti bahwa asumsi terpenuhi. Selalu baca bersama plot di bawah."
    )
    for note in diag.notes:
        st.info(note, icon=":material/info:")

    _visual_section(diag)
    _tests_section(diag)
    _collinearity_section(diag)
    _influence_section(diag)
    ui.aside("Baca dengan hati-hati", "Yang tidak dapat disampaikan diagnostik ini", LIMITATIONS)


def _plot_panel(key: str, title: str, look_for: str, fig, reading: str) -> None:
    with st.container(border=True, key=f"panel-{key}", height="stretch"):
        ui.panel_title(title, f"Yang dicari: {look_for}")
        st.plotly_chart(fig, width="stretch")
        st.caption(reading)


def _visual_section(diag: Diagnostics) -> None:
    st.subheader("Diagnostik visual")
    st.caption(
        "Kurva LOWESS (hijau) adalah penghalus visual untuk membantu melihat pola, bukan model. Titik tembaga "
        "pada plot residual adalah observasi dengan |residual terstudentisasi| > 3."
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _plot_panel(
            "rvf", "Residual vs nilai prediksi", "lengkungan atau pola corong", residuals_vs_fitted(diag),
            "Yang diharapkan: titik tersebar acak di sekitar 0 tanpa pola. Lengkungan menandakan hubungan "
            "nonlinear yang belum dimodelkan. Pola corong menandakan varians galat tidak konstan.",
        )
    with c2:
        _plot_panel(
            "qq", "Normal Q-Q plot", "ujung titik yang menjauh dari garis", qq_plot(diag),
            "Yang diharapkan: titik mengikuti garis acuan. Ekor yang melengkung menjauh menandakan ekor lebih "
            "tebal atau tipis daripada normal. Pada n besar, inferensi OLS cukup tahan terhadap "
            "ketidaknormalan sedang.",
        )
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        _plot_panel(
            "hist", "Distribusi residual terstandar", "asimetri atau lebih dari satu puncak",
            residual_histogram(diag), "Kurva N(0, 1) hanya sebagai pembanding visual.",
        )
    with c4:
        _plot_panel(
            "scale", "Scale-location", "tren naik atau turun", scale_location(diag),
            "Yang diharapkan: sebaran vertikal dan kurva LOWESS kira-kira datar. Tren naik atau turun "
            "menandakan heteroskedastisitas.",
        )


def _tests_section(diag: Diagnostics) -> None:
    st.subheader("Uji formal")
    st.caption(
        "Uji formal adalah bukti pendukung, bukan pengganti plot. p ≥ α **tidak** membuktikan asumsi "
        "terpenuhi: bisa jadi uji kurang berdaya (n kecil) atau pelanggarannya berbentuk lain."
    )
    table = pd.DataFrame(
        [
            {
                "Asumsi": t.assumption,
                "Uji": t.test,
                "H0": t.h0,
                "Statistik": fmt_num(t.statistic) if t.available else "—",
                "df": t.df or "—",
                "p-value": fmt_pvalue(t.p_value) if t.available else "—",
                "Kesimpulan (α = 0.05)": t.conclusion,
            }
            for t in diag.tests
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    for t in diag.tests:
        if t.note:
            st.caption(f"{t.test}: {t.note}")
    st.markdown(
        f"**Independensi galat.** Durbin–Watson = {diag.durbin_watson:.3f}. Statistik ini hanya bermakna jika "
        "urutan baris memiliki arti (misalnya urutan waktu): ≈ 2 berarti tidak ada autokorelasi orde-1, "
        "mendekati 0 berarti autokorelasi positif, dan mendekati 4 berarti negatif. Untuk data potong-lintang, "
        "independensi dinilai dari cara data dikumpulkan, bukan dari uji."
    )


def _collinearity_section(diag: Diagnostics) -> None:
    st.subheader("Multikolinearitas")
    if diag.vif is None:
        st.caption("VIF hanya relevan untuk model dengan minimal dua prediktor.")
        return
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.dataframe(
            diag.vif,
            width="stretch",
            column_config={
                "_index": "Prediktor",
                "vif": st.column_config.ProgressColumn("VIF (skala 0–10)", format="%.3f", min_value=0, max_value=10),
                "tingkat": "Tingkat",
            },
        )
    with right:
        metric_row([("Condition number (terskala)", fmt_num(diag.condition_number, 3), None)])
        if diag.condition_number > CONDITION_HIGH:
            st.warning(f"Condition number > {CONDITION_HIGH:g} menandakan multikolinearitas yang kuat.")
    st.caption(
        "VIF_j = 1 / (1 − R²_j), dengan R²_j dari regresi prediktor j pada prediktor lain. VIF = 1 berarti "
        "tidak berkorelasi. Ambang 5 (patut diperiksa) dan 10 (biasanya bermasalah) hanyalah aturan praktis. "
        "Multikolinearitas memperbesar galat baku (estimasi koefisien individual menjadi tidak presisi), tetapi "
        "tidak membuat koefisien bias dan tidak merusak prediksi. Condition number (Belsley) dihitung dari matriks "
        "desain, termasuk kolom intersep, yang tiap kolomnya diskalakan ke panjang satu tanpa pemusatan. Karena "
        "itu nilainya bisa tinggi untuk prediktor yang jauh dari nol (misalnya tahun) walaupun VIF rendah."
    )


def _influence_section(diag: Diagnostics) -> None:
    st.subheader("Observasi berpengaruh")
    with st.container(border=True, key="panel-cooks"):
        st.plotly_chart(cooks_distance(diag), width="stretch")
    th = diag.thresholds
    obs = diag.observations
    n_flag = len(diag.influential)
    n_cooks = int((obs["cooks_d"] > th["cooks_d"]).sum())
    n_lev = int((obs["leverage"] > th["leverage"]).sum())
    n_res = int((obs["student_resid"].abs() > th["student_resid"]).sum())
    st.caption(
        f"Ditandai bila Cook's D > 4/n ({th['cooks_d']:.3g}): {n_cooks} observasi; leverage > 2p/n "
        f"({th['leverage']:.3g}): {n_lev}; |residual terstudentisasi eksternal| > {th['student_resid']:g}: "
        f"{n_res}. Total {n_flag} observasi unik ditandai. "
        "Ambang ini aturan praktis. Jangan menghapus observasi hanya karena melewatinya: periksa kesalahan "
        "data, lalu bandingkan hasil dengan dan tanpa observasi tersebut dan laporkan keduanya."
    )
    if n_flag:
        shown = diag.influential.head(MAX_INFLUENTIAL_ROWS)[
            ["fitted", "resid", "student_resid", "leverage", "cooks_d"]
        ]
        shown.index.name = "baris"
        st.dataframe(
            shown,
            width="stretch",
            column_config={
                "fitted": st.column_config.NumberColumn("ŷ", format="%.4g"),
                "resid": st.column_config.NumberColumn("residual", format="%.4g"),
                "student_resid": st.column_config.NumberColumn("resid. terstudentisasi", format="%.3f"),
                "leverage": st.column_config.NumberColumn("leverage", format="%.3f"),
                "cooks_d": st.column_config.NumberColumn("Cook's D", format="%.4f"),
            },
        )
        if n_flag > MAX_INFLUENTIAL_ROWS:
            st.caption(f"Menampilkan {MAX_INFLUENTIAL_ROWS} dari {n_flag} observasi dengan Cook's D terbesar.")
