"""Exploration (Data step): descriptive statistics, distributions and correlations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.data import numeric_columns
from datalens.eda import correlation_matrix, describe_numeric, strongest_pairs
from datalens.formatting import fmt_num
from datalens.plots import correlation_heatmap, distribution_figure, mini_histogram, scatter_figure
from views import ui

DEFAULT_MAX_VARS = 12
MAX_SMALL_MULTIPLES = 8
METHODS = {"Pearson": "pearson", "Spearman": "spearman"}

_NUM = st.column_config.NumberColumn
DESCRIBE_COLUMNS = {
    "n": _NUM("n", help="Jumlah nilai valid (tidak hilang)."),
    "n_hilang": _NUM("hilang"),
    "mean": _NUM("mean", format="%.4g"),
    "sd": _NUM("SD", format="%.4g", help="Simpangan baku sampel (pembagi n − 1)."),
    "min": _NUM("min", format="%.4g"),
    "q1": _NUM("Q1", format="%.4g"),
    "median": _NUM("median", format="%.4g"),
    "q3": _NUM("Q3", format="%.4g"),
    "max": _NUM("max", format="%.4g"),
    "iqr": _NUM("IQR", format="%.4g", help="Q3 − Q1."),
    "skewness": _NUM("skewness", format="%.3f"),
    "kurtosis": _NUM("kurtosis", format="%.3f", help="Kurtosis berlebih: distribusi normal = 0."),
    "n_outlier": _NUM("outlier", help="Jumlah nilai di luar pagar Tukey (Q1 − 1,5·IQR, Q3 + 1,5·IQR)."),
}


def render(df: pd.DataFrame) -> None:
    numeric = numeric_columns(df)
    if not numeric:
        st.info("Dataset ini tidak memiliki kolom numerik untuk dieksplorasi.")
        return

    selected = st.multiselect(
        "Variabel numerik yang dianalisis",
        numeric,
        default=numeric[:DEFAULT_MAX_VARS],
        placeholder="Pilih variabel",
        help="Pilihan ini berlaku untuk statistik deskriptif, distribusi, dan korelasi di bawah.",
        key="explore_vars",
        persist_state="session",
    )
    if not selected:
        st.info("Pilih minimal satu variabel numerik.")
        return

    _descriptive_section(df, selected)
    st.divider()
    _distribution_section(df, selected)
    st.divider()
    _correlation_section(df, selected)


def _descriptive_section(df: pd.DataFrame, selected: list[str]) -> None:
    st.subheader("Statistik deskriptif")
    st.dataframe(describe_numeric(df, selected), width="stretch", column_config=DESCRIBE_COLUMNS)
    with st.expander("Cara membaca statistik deskriptif", icon=":material/menu_book:"):
        st.markdown(
            """
- **Mean vs median.** Mean peka terhadap nilai ekstrem; median tidak. Selisih yang besar di antara
  keduanya menandakan distribusi yang miring atau adanya outlier.
- **SD & IQR.** SD mengukur sebaran di sekitar mean (pembagi n − 1). IQR adalah rentang 50% data di
  tengah dan lebih tahan terhadap outlier.
- **Kuartil** dihitung dengan interpolasi linear (definisi default pandas/NumPy). Perangkat lunak lain
  bisa memakai definisi berbeda sehingga hasilnya sedikit berbeda pada sampel kecil.
- **Skewness** (estimator sampel terkoreksi): positif berarti ekor kanan lebih panjang, negatif berarti
  ekor kiri lebih panjang, dan 0 untuk distribusi simetris. **Kurtosis** di sini adalah kurtosis berlebih
  (normal = 0); nilai positif berarti ekor lebih tebal daripada normal. Keduanya tidak stabil pada sampel kecil.
- **Outlier (pagar Tukey)** adalah tanda untuk *diperiksa*, bukan untuk dihapus otomatis. Nilai ekstrem
  bisa sah, dan pada distribusi miring wajar muncul banyak nilai di luar pagar.
- Semua statistik dihitung dari nilai yang tersedia pada tiap kolom. Nilai hilang dan tak hingga diabaikan.
"""
        )


def _distribution_section(df: pd.DataFrame, selected: list[str]) -> None:
    head, toggle = st.columns([2, 1], vertical_alignment="bottom")
    head.subheader("Distribusi")
    with toggle, st.container(horizontal=True, horizontal_alignment="right"):
        view = st.segmented_control("Tampilan", ["Ringkas", "Rinci"], default="Ringkas", required=True,
                                    key="dist_view", label_visibility="collapsed")
    if view == "Ringkas":
        _small_multiples(df, selected)
        return
    var = st.selectbox("Variabel", selected, key="dist_var")
    stats = describe_numeric(df, [var]).loc[var]
    if stats["n"] == 0:
        st.info(f"Kolom **{var}** tidak memiliki nilai valid.")
        return

    st.plotly_chart(distribution_figure(df[var], var), width="stretch")
    skew = stats["skewness"]
    if pd.isna(skew):
        shape = "skewness tidak terdefinisi (n < 3 atau tanpa variasi)"
    else:
        direction = "ekor kanan lebih panjang" if skew > 0 else "ekor kiri lebih panjang" if skew < 0 else "simetris"
        shape = f"skewness = {skew:.3f} ({direction})"
    st.caption(
        f"n = {int(stats['n']):,} · mean = {fmt_num(stats['mean'])} · median = {fmt_num(stats['median'])} · "
        f"{shape} · {int(stats['n_outlier'])} outlier menurut pagar Tukey. "
        "Lebar bin mengikuti aturan 'auto' NumPy (Sturges / Freedman–Diaconis). Bentuk histogram ikut "
        "berubah bila lebar bin diubah, jadi bacalah bersama box plot dan statistik di atas."
    )


def _small_multiples(df: pd.DataFrame, selected: list[str]) -> None:
    shown = selected[:MAX_SMALL_MULTIPLES]
    stats = describe_numeric(df, shown)
    cols = st.columns(2, gap="medium")
    for i, var in enumerate(shown):
        row = stats.loc[var]
        with cols[i % 2], st.container(border=True, key=f"panel-hist-{i}"):
            missing = f" · {int(row['n_hilang'])} hilang" if row["n_hilang"] else ""
            st.html(
                '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap">'
                f'<span class="dl-mono" style="font-size:13.5px;font-weight:500">{ui.esc(var)}</span>'
                f'<span class="dl-mono" style="font-size:12px;color:var(--dl-muted)">mean {fmt_num(row["mean"])} · '
                f'SD {fmt_num(row["sd"])}{missing}</span></div>'
            )
            if row["n"] == 0:
                st.caption("Tidak ada nilai valid.")
            else:
                st.plotly_chart(mini_histogram(df[var], var), width="stretch", key=f"mini_hist_{var}")
    note = "Garis putus-putus hijau = rata-rata. Pilih **Rinci** untuk box plot, median, dan outlier."
    if len(selected) > MAX_SMALL_MULTIPLES:
        note += f" Menampilkan {MAX_SMALL_MULTIPLES} dari {len(selected)} variabel terpilih."
    st.caption(note)


def _correlation_section(df: pd.DataFrame, selected: list[str]) -> None:
    st.subheader("Korelasi")
    if len(selected) < 2:
        st.info("Pilih minimal dua variabel untuk melihat korelasi.")
        return

    method_label = st.radio("Metode", list(METHODS), horizontal=True, key="corr_method")
    result = correlation_matrix(df, selected, METHODS[method_label])

    if result.constant:
        st.warning(
            f"Korelasi tidak terdefinisi untuk kolom tanpa variasi: {', '.join(result.constant)}."
        )
    if result.pairwise_n_varies:
        st.info(
            "Nilai hilang ditangani dengan *pairwise deletion*: setiap sel dihitung dari baris yang lengkap "
            "untuk pasangan tersebut, sehingga n bisa berbeda antarsel (lihat n saat kursor diarahkan ke sel)."
        )

    st.plotly_chart(correlation_heatmap(result), width="stretch")

    pairs = strongest_pairs(result)
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.markdown("**Pasangan dengan |r| terbesar**")
        st.dataframe(
            pairs,
            hide_index=True,
            width="stretch",
            column_config={
                "variabel_1": "Variabel 1",
                "variabel_2": "Variabel 2",
                "r": _NUM("r", format="%.3f"),
                "n": _NUM("n"),
            },
        )
        with st.expander("Matriks korelasi lengkap (tabel)"):
            st.dataframe(result.r.round(3), width="stretch")
    with right:
        _scatter(df, selected, result, pairs)

    with st.expander("Cara membaca korelasi", icon=":material/menu_book:"):
        st.markdown(
            """
- **Pearson r** mengukur kekuatan hubungan *linear* dan peka terhadap outlier. **Spearman ρ** adalah
  korelasi Pearson atas peringkat data. Ia mengukur hubungan *monoton* dan lebih tahan terhadap outlier.
- **r ≈ 0 tidak berarti tidak ada hubungan.** Hubungan nonlinear (misalnya berbentuk U) bisa memiliki
  r mendekati nol. Selalu periksa diagram pencar.
- **Korelasi bukan kausalitas.** Dua variabel bisa berkorelasi karena variabel ketiga (perancu),
  kebetulan, atau arah sebab yang terbalik.
- **r bukan slope.** r tidak bersatuan dan tidak menyatakan seberapa besar Y berubah per unit X.
- **Mengapa tidak ada p-value?** Menguji banyak pasangan sekaligus memperbesar peluang temuan
  "signifikan" yang palsu (masalah perbandingan berganda). Tab ini dimaksudkan untuk eksplorasi, bukan
  untuk pengujian hipotesis.
"""
        )


def _scatter(df: pd.DataFrame, selected: list[str], result, pairs: pd.DataFrame) -> None:
    st.markdown("**Diagram pencar**")
    default_x, default_y = (pairs.iloc[0]["variabel_1"], pairs.iloc[0]["variabel_2"]) if len(pairs) else selected[:2]
    c1, c2 = st.columns(2)
    x = c1.selectbox("Sumbu X", selected, index=selected.index(default_x))
    y = c2.selectbox("Sumbu Y", selected, index=selected.index(default_y))
    if x == y:
        st.info("Pilih dua variabel yang berbeda.")
        return
    st.plotly_chart(scatter_figure(df, x, y), width="stretch")
    r = result.r.loc[x, y]
    label = "r" if result.method == "pearson" else "ρ"
    st.caption(
        f"{label} ({result.method.capitalize()}) = {fmt_num(r, 3) if pd.notna(r) else 'tidak terdefinisi'}, "
        f"n = {int(result.n.loc[x, y]):,} pasangan lengkap."
    )
