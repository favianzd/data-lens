"""Welcome / empty state: upload a CSV or load the synthetic sample."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from datalens.data import DataLoadError, load_csv
from datalens.sample_data import TARGET, TRUE_COEFFICIENTS, make_study_dataset
from views import ui
from views.state import SAMPLE_KEY, go, set_dataset

SEPARATORS = {"Otomatis": None, "Koma (,)": ",", "Titik koma (;)": ";", "Tab": "\t", "Pipa (|)": "|"}
DECIMALS = {"Titik (1.5)": ".", "Koma (1,5)": ","}

HERO = """
<section class="dl-hero">
  <span class="dl-eyebrow">Laboratorium analisis statistik interaktif</span>
  <h1>Ubah file CSV menjadi model regresi yang dapat dipertanggungjawabkan.</h1>
  <p>DataLens memprofilkan data Anda, mengestimasi model regresi linear (OLS), dan memeriksa asumsinya
  dalam empat langkah terpandu. Setiap angka di layar dihitung dari file Anda.</p>
</section>
"""

UPLOAD_HEAD = """
<div style="display:flex;flex-direction:column;align-items:center;gap:10px;text-align:center;padding:14px 8px 4px">
  <span style="width:52px;height:52px;border-radius:50%;background:var(--dl-tint);
    display:flex;align-items:center;justify-content:center">{icon}</span>
  <span style="font-size:20px;font-weight:600;letter-spacing:-.01em">Seret file CSV ke sini untuk memulai</span>
  <span style="font-size:14px;color:var(--dl-ink2)">Pemisah kolom dan encoding (UTF-8 atau Latin-1) dideteksi
    otomatis. Maksimum 50 MB.</span>
</div>
"""

STEPS_HTML = """
<h2 style="font-family:var(--dl-serif);font-weight:400;font-size:30px;margin:8px 0 18px;padding:0">
  Yang terjadi setelah Anda mengunggah</h2>
<ol class="dl-steps">
  <li><span class="n">01</span><b>Baca &amp; validasi</b><span>Encoding dan pemisah kolom dideteksi. File kosong,
    file yang hanya berisi header, dan nama kolom ganda ditolak dengan pesan yang jelas.</span></li>
  <li><span class="n">02</span><b>Profil data</b><span>Jumlah baris dan kolom, tipe data, nilai hilang, dan
    baris duplikat dirangkum sebelum Anda memodelkan apa pun.</span></li>
  <li><span class="n">03</span><b>Estimasi model</b><span>Pilih variabel dependen numerik dan prediktor. OLS
    diestimasi pada baris lengkap dan dilaporkan beserta ketidakpastiannya.</span></li>
  <li><span class="n">04</span><b>Diagnostik &amp; laporan</b><span>Plot residual, Q–Q, dan VIF menunjukkan di
    mana asumsi mungkin tidak terpenuhi. Temuan diekspor sebagai laporan.</span></li>
</ol>
"""

# Illustrative mini-figures for the capability cards (decorative, not data).
FIG_0 = """<svg viewBox="0 0 300 110" width="300" height="110"><path d="M4 100 H296" stroke="#CFCCC1"></path>
    <g fill="#2D4A6E"><rect x="8" y="90" width="22" height="10"></rect><rect x="34" y="78" width="22" height="22"></rect>
    <rect x="60" y="62" width="22" height="38"></rect><rect x="86" y="40" width="22" height="60"></rect>
    <rect x="112" y="20" width="22" height="80"></rect><rect x="138" y="8" width="22" height="92"></rect>
    <rect x="164" y="16" width="22" height="84"></rect><rect x="190" y="36" width="22" height="64"></rect>
    <rect x="216" y="58" width="22" height="42"></rect><rect x="242" y="76" width="22" height="24"></rect>
    <rect x="268" y="88" width="22" height="12"></rect></g>
    <path d="M150 2 V100" stroke="#0B7A57" stroke-width="1.5" stroke-dasharray="4 3"></path></svg>"""
FIG_1 = """<svg viewBox="0 0 300 110" width="300" height="110"><path d="M120 4 V104" stroke="#CFCCC1"
    stroke-dasharray="3 3"></path><g font-family="JetBrains Mono, monospace" font-size="10.5" fill="#3A4659">
    <text x="0" y="28">prediktor_1</text><text x="0" y="58">prediktor_2</text><text x="0" y="88">prediktor_3</text></g>
    <g stroke="#0F1D33" stroke-width="2" stroke-linecap="round"><path d="M230 24 H269"></path>
    <path d="M177 54 H214"></path><path d="M134 84 H169"></path></g><g fill="#0B7A57"><circle cx="250" cy="24" r="5">
    </circle><circle cx="195" cy="54" r="5"></circle><circle cx="151" cy="84" r="5"></circle></g></svg>"""
FIG_2 = """<svg viewBox="0 0 300 110" width="300" height="110"><path d="M4 55 H296" stroke="#CFCCC1"
    stroke-dasharray="3 3"></path><path d="M14 48h.01M24 62h.01M33 37h.01M41 54h.01M50 43h.01M58 70h.01M66 34h.01
    M75 51h.01M83 58h.01M92 40h.01M100 47h.01M109 73h.01M126 53h.01M134 45h.01M143 64h.01M151 38h.01M160 50h.01
    M168 58h.01M177 35h.01M185 46h.01M194 67h.01M202 41h.01M211 52h.01M228 59h.01M236 44h.01M245 54h.01M253 38h.01
    M262 62h.01M270 47h.01M279 33h.01M287 53h.01" stroke="#2D4A6E" stroke-width="6" stroke-linecap="round"
    opacity=".8"></path><path d="M8 53 C80 55 140 57 200 55 S270 53 292 54" stroke="#0B7A57" stroke-width="2"
    fill="none"></path><path d="M219 22h.01" stroke="#B45A1E" stroke-width="8" stroke-linecap="round"></path></svg>"""

CAPABILITIES_TEMPLATE = """
<h2 style="font-family:var(--dl-serif);font-weight:400;font-size:30px;margin:8px 0 18px;padding:0">
  Tiga hal yang dikerjakan DataLens</h2>
<div class="dl-caps">
  <article><div class="fig">{fig0}</div>
    <div class="body"><span class="dl-eyebrow" style="color:var(--dl-emerald-dark)">Eksplorasi</span>
    <h3>Kenali setiap variabel sebelum mempercayainya</h3>
    <ul><li>Ringkasan nilai hilang dan duplikat</li><li>Statistik deskriptif per kolom</li>
    <li>Distribusi dan peta korelasi</li></ul></div></article>
  <article><div class="fig">{fig1}</div>
    <div class="body"><span class="dl-eyebrow" style="color:var(--dl-emerald-dark)">Model</span>
    <h3>Estimasi OLS dengan variabel pilihan Anda</h3>
    <ul><li>Koefisien, galat baku, CI, dan p-value</li><li>R², adjusted R², dan uji F</li>
    <li>Perbandingan dua model dengan validasi silang</li></ul></div></article>
  <article><div class="fig">{fig2}</div>
    <div class="body"><span class="dl-eyebrow" style="color:var(--dl-emerald-dark)">Diagnostik</span>
    <h3>Periksa apakah asumsi model terpenuhi</h3>
    <ul><li>Residual vs prediksi dan scale-location</li><li>Normal Q–Q dan uji formal</li>
    <li>VIF dan observasi berpengaruh</li></ul></div></article>
</div>
"""
CAPABILITIES = CAPABILITIES_TEMPLATE.format(
    fig0='<span class="dl-fig-0" role="img" aria-label="Contoh histogram dengan garis rata-rata"></span>',
    fig1='<span class="dl-fig-1" role="img" aria-label="Contoh plot koefisien dengan interval kepercayaan"></span>',
    fig2='<span class="dl-fig-2" role="img" aria-label="Contoh plot residual terhadap nilai prediksi"></span>',
)
FIGURE_CSS = "<style>" + "".join(
    ui.svg_rule(f"dl-fig-{i}", fig, "100%", "110px") for i, fig in enumerate((FIG_0, FIG_1, FIG_2))
) + "</style>"


@st.cache_data(show_spinner="Membaca file…")
def _load(raw: bytes, sep: str | None, decimal: str) -> pd.DataFrame:
    return load_csv(raw, sep=sep, decimal=decimal)


def _load_sample() -> None:
    set_dataset(make_study_dataset(), "contoh_sintetis.csv", SAMPLE_KEY)


def _sample_schema() -> str:
    rows = [("program_studi", "kategori")] + [(c, "numerik") for c in TRUE_COEFFICIENTS] + [(TARGET, "dependen")]
    return "".join(
        f'<div style="display:flex;justify-content:space-between;padding:7px 12px;'
        f'border-bottom:1px solid var(--dl-line)"><span class="dl-mono">{ui.esc(c)}</span>'
        f'<span style="color:{"var(--dl-emerald-dark)" if t == "dependen" else "var(--dl-muted)"}">{t}</span></div>'
        for c, t in rows
    )


def render(df: pd.DataFrame | None) -> None:
    ui.breadcrumb(["Ruang kerja", "Analisis baru"])
    st.html(HERO)

    if df is not None:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.info(
                f"Dataset aktif: **{st.session_state.dataset_name}**. Memuat dataset baru akan menghapus model, "
                "diagnostik, dan laporan yang ada.",
                icon=":material/info:",
            )
            st.button("Kembali ke Data", icon=":material/arrow_back:", on_click=go, args=("data",))

    left, right = st.columns([2, 1], gap="medium")
    with left, st.container(border=True, key="upload-zone"):
        st.html(UPLOAD_HEAD.format(icon=ui.ICON_UPLOAD))
        uploaded = st.file_uploader("Unggah dataset CSV", type=["csv"], label_visibility="collapsed")
        with st.expander("Opsi format CSV", icon=":material/tune:"):
            c1, c2 = st.columns(2)
            sep_label = c1.selectbox("Pemisah kolom", list(SEPARATORS))
            dec_label = c2.selectbox("Pemisah desimal", list(DECIMALS))
        if uploaded is not None:
            raw = uploaded.getvalue()
            sep, dec = SEPARATORS[sep_label], DECIMALS[dec_label]
            key = hashlib.sha256(raw).hexdigest() + repr((sep, dec))
            try:
                set_dataset(_load(raw, sep, dec), uploaded.name, key)
            except DataLoadError as exc:
                st.error(f"**Tidak dapat membaca {uploaded.name}.** {exc}", icon=":material/error:")
            else:
                st.rerun()

    with right, st.container(border=True, key="sample-zone", height="stretch"):
        st.html(
            '<div style="display:flex;justify-content:space-between;align-items:center"><span class="dl-eyebrow">'
            'Belum punya data?</span><span class="dl-chip dl-chip-amber">Sintetis</span></div>'
            '<h2 style="font-family:var(--dl-serif);font-weight:400;font-size:25px;line-height:1.15;margin:10px 0 8px;'
            'padding:0">Coba contoh data kebiasaan belajar</h2>'
            '<p style="margin:0 0 12px;font-size:14px;line-height:1.6;color:var(--dl-ink2)">200 mahasiswa simulasi '
            "dengan koefisien populasi yang diketahui, sehingga Anda dapat memeriksa apakah estimasi mendekati nilai "
            "sebenarnya.</p>"
            '<div style="background:#fff;border:1px solid var(--dl-border);border-radius:6px;font-size:12.5px">'
            f"{_sample_schema()}</div>"
        )
        st.button("Gunakan contoh data sintetis", key="load_sample", icon=":material/dataset:", width="stretch",
                  on_click=_load_sample)
        st.caption("Data dibangkitkan secara acak (lihat `datalens/sample_data.py`), jadi bukan data nyata.")

    st.space("medium")
    st.html(STEPS_HTML)
    st.space("medium")
    st.html(FIGURE_CSS + CAPABILITIES)
    st.space("small")
    st.caption("File hanya dibaca ke memori untuk sesi ini · Dibangun dengan Python, pandas, statsmodels, dan Streamlit")
