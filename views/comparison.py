"""Model comparison (tab in the Regresi step): two specifications, in-sample fit vs. cross-validated error."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.comparison import BASELINE, ComparisonResult, compare_models
from datalens.data import numeric_columns
from datalens.formatting import fmt_num
from datalens.plots import cv_error_figure
from datalens.regression import ModelSpecError, RegressionResult
from views.components import p_label


def render(df: pd.DataFrame) -> None:
    numeric = numeric_columns(df)
    if len(numeric) < 3:
        st.warning("Perbandingan model membutuhkan minimal tiga kolom numerik.")
        return

    # Prefill from the model on the Regresi tab: A = its first predictor, B = all of them.
    current: RegressionResult | None = st.session_state.get("result")
    target_default = current.target if current else numeric[-1]
    b_default = list(current.predictors) if current and len(current.predictors) >= 2 else []
    a_default = b_default[:1]

    with st.form("comparison_spec"):
        target = st.selectbox("Variabel dependen (Y)", numeric, index=numeric.index(target_default))
        c1, c2 = st.columns(2)
        pred_a = c1.multiselect("Model A: prediktor", numeric, default=a_default,
                                placeholder="mis. satu prediktor (regresi sederhana)")
        pred_b = c2.multiselect("Model B: prediktor", numeric, default=b_default,
                                placeholder="mis. beberapa prediktor (regresi berganda)")
        c3, c4, c5 = st.columns(3)
        n_splits = c3.selectbox("Jumlah fold (k)", [5, 10], index=0)
        n_repeats = c4.selectbox("Pengulangan", [1, 5, 10], index=2)
        seed = c5.number_input("Seed acak", min_value=0, value=42, step=1)
        submitted = st.form_submit_button("Bandingkan model", type="primary")

    if submitted:
        try:
            with st.spinner("Menjalankan validasi silang…"):
                st.session_state.comparison = compare_models(
                    df, target, pred_a, pred_b, n_splits=n_splits, n_repeats=n_repeats, seed=int(seed)
                )
        except ModelSpecError as exc:
            st.session_state.comparison = None
            st.error(str(exc), icon=":material/error:")

    result: ComparisonResult | None = st.session_state.get("comparison")
    if result is None:
        if not submitted:
            st.caption("Tentukan dua model lalu klik **Bandingkan model**.")
        return
    render_result(result)


def render_result(r: ComparisonResult) -> None:
    a, b = r.names
    st.caption(
        f"Semua model dievaluasi pada {r.n_obs:,} baris yang sama (baris lengkap untuk gabungan variabel "
        f"kedua model; {r.n_dropped} baris dikeluarkan)."
    )
    for w in r.warnings:
        st.warning(w)

    st.subheader("Kecocokan dalam sampel")
    st.dataframe(
        r.in_sample,
        width="stretch",
        column_config={
            "formula": "Spesifikasi",
            "k": st.column_config.NumberColumn("k", help="Jumlah prediktor."),
            "r_squared": st.column_config.NumberColumn("R²", format="%.4f"),
            "adj_r_squared": st.column_config.NumberColumn("Adj. R²", format="%.4f"),
            "aic": st.column_config.NumberColumn("AIC", format="%.2f"),
            "bic": st.column_config.NumberColumn("BIC", format="%.2f"),
            "rmse_in_sample": st.column_config.NumberColumn("RMSE (in-sample)", format="%.4g"),
        },
    )
    if r.nested_f:
        f = r.nested_f
        st.markdown(
            f"**Uji F parsial (model bersarang).** Model {f.full} menambah {f.df_num} prediktor ke model "
            f"{f.restricted}: F({f.df_num}, {f.df_denom}) = {fmt_num(f.f_statistic)}, p {p_label(f.p_value)}. "
            "H0: koefisien prediktor tambahan semuanya nol. Uji ini mengukur bukti statistik dalam sampel, "
            "bukan seberapa besar perbaikan prediksinya."
        )
    else:
        st.caption("Uji F parsial tidak tersedia: prediktor kedua model tidak bersarang (bukan subset satu sama lain).")

    st.subheader(f"Validasi silang ({r.n_splits}-fold × {r.n_repeats} pengulangan)")
    s = r.cv_summary
    cv_table = pd.DataFrame(
        {
            "RMSE (rata-rata ± SD)": [f"{fmt_num(s.loc[m, 'rmse_mean'])} ± {fmt_num(s.loc[m, 'rmse_std'], 3)}" for m in s.index],
            "MAE (rata-rata ± SD)": [f"{fmt_num(s.loc[m, 'mae_mean'])} ± {fmt_num(s.loc[m, 'mae_std'], 3)}" for m in s.index],
        },
        index=pd.Index(s.index, name="Model"),
    )
    st.dataframe(cv_table, width="stretch")
    st.plotly_chart(cv_error_figure(r), width="stretch")

    diff = r.paired_rmse_diff
    better_b = (diff < 0).mean()
    lower = b if s.loc[b, "rmse_mean"] < s.loc[a, "rmse_mean"] else a
    st.markdown(
        f"Rata-rata RMSE validasi silang lebih rendah pada **model {lower}**. Selisih per split "
        f"RMSE(B) − RMSE(A) = {fmt_num(diff.mean())} (SD {fmt_num(diff.std(), 3)}). RMSE model B lebih rendah "
        f"daripada A pada {better_b:.0%} dari {len(diff)} split. Bandingkan juga keduanya dengan baseline: model yang "
        f"tidak jauh lebih baik dari **{BASELINE}** hampir tidak memiliki daya prediksi."
    )

    with st.expander("Mengapa kecocokan dalam sampel tidak cukup?", icon=":material/menu_book:"):
        st.markdown(
            f"""
- **R² dalam sampel selalu naik** (atau tetap) saat prediktor ditambahkan, bahkan prediktor berupa noise
  murni, karena model dievaluasi pada data yang sama dengan data yang dipakai untuk estimasi. Ini
  disebut *overfitting*.
- **Adjusted R², AIC, dan BIC** memberi penalti untuk kompleksitas (BIC lebih keras). Nilai AIC/BIC
  yang lebih kecil lebih baik, dan hanya bisa dibandingkan antarmodel pada data yang sama, seperti di sini.
- **Validasi silang** mengukur galat pada data yang *tidak* dipakai untuk estimasi, sehingga lebih
  dekat dengan kinerja pada data baru dari populasi yang sama. Ini tetap bukan jaminan untuk data dari
  waktu atau populasi berbeda.
- **Tanpa kebocoran data.** Setiap model di-*fit* ulang di dalam tiap fold latih melalui pipeline
  scikit-learn. Satu-satunya praproses (menghapus baris tidak lengkap) bersifat per baris dan tidak
  memakai informasi dari baris lain.
- **SD antar-fold bukan galat baku.** Fold berbagi data latih sehingga tidak independen. Perbedaan kecil
  antar-model sebaiknya tidak ditafsirkan berlebihan.
- **Split identik** dipakai untuk semua model (seed {r.seed}), sehingga selisih per split adalah
  perbandingan berpasangan.
- **Prediksi baik ≠ kausal.** Model dengan galat prediksi terendah belum tentu menggambarkan
  hubungan sebab-akibat.
"""
        )
