"""Data overview: size, types, missingness, duplicates, column profile and a preview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.data import profile
from views import ui

PREVIEW_ROWS = 100
PREVIEW_MODES = ["Awal", "Acak", "Hanya baris tak lengkap"]


def render(df: pd.DataFrame) -> None:
    prof = profile(df)
    n_cells = prof.n_rows * prof.n_cols
    n_other = prof.n_cols - prof.n_numeric
    ui.stat_strip(
        [
            ui.Stat("Baris", f"{prof.n_rows:,}", "observasi"),
            ui.Stat("Kolom", f"{prof.n_cols}", f"{prof.n_numeric} numerik · {n_other} non-numerik"),
            ui.Stat("Sel kosong", f"{prof.n_missing_cells:,}",
                    f"{prof.n_missing_cells / n_cells:.1%} dari {n_cells:,} sel" if n_cells else ""),
            ui.Stat("Baris tak lengkap", f"{prof.n_rows_with_missing:,}",
                    f"{prof.n_rows_with_missing / prof.n_rows:.1%} dari semua baris",
                    value_color="var(--dl-amber)" if prof.n_rows_with_missing else None),
            ui.Stat("Baris duplikat", f"{prof.n_duplicate_rows:,}",
                    "Tidak ditemukan" if not prof.n_duplicate_rows else "Periksa sebelum memodelkan",
                    tone="ok" if not prof.n_duplicate_rows else "review"),
        ]
    )

    if prof.n_rows_with_missing:
        cols = prof.columns.index[prof.columns["n_hilang"] > 0]
        st.warning(
            f"**{prof.n_rows_with_missing:,} baris memiliki nilai hilang** pada {', '.join(f'`{c}`' for c in cols)}. "
            "Regresi hanya memakai baris yang lengkap pada variabel model (complete-case analysis).",
            icon=":material/warning:",
        )
    if prof.n_duplicate_rows:
        st.warning(
            f"Terdapat {prof.n_duplicate_rows} baris duplikat. DataLens tidak menghapusnya otomatis: "
            "duplikat bisa berupa kesalahan entri atau observasi sah yang kebetulan identik.",
            icon=":material/content_copy:",
        )

    left, right = st.columns([1, 1.5], gap="medium")
    with left, st.container(border=True, key="panel-profile"):
        ui.panel_title("Profil kolom", f"{prof.n_cols} kolom")
        st.dataframe(
            prof.columns,
            width="stretch",
            column_config={
                "_index": "Kolom",
                "tipe": "Tipe",
                "n_hilang": st.column_config.NumberColumn("Hilang"),
                "persen_hilang": st.column_config.ProgressColumn(
                    "% hilang", format="%.1f%%", min_value=0, max_value=100
                ),
                "n_unik": st.column_config.NumberColumn("Unik"),
            },
        )
    with right, st.container(border=True, key="panel-preview"):
        mode = st.segmented_control("Pratinjau data", PREVIEW_MODES, default="Awal", required=True,
                                    key="preview_mode")
        if mode == "Acak":
            shown = df.sample(min(PREVIEW_ROWS, len(df)), random_state=0).sort_index()
        elif mode == "Hanya baris tak lengkap":
            shown = df[df.isna().any(axis=1)].head(PREVIEW_ROWS)
        else:
            shown = df.head(PREVIEW_ROWS)
        st.dataframe(shown, width="stretch", height=260)
        st.caption(f"Menampilkan {len(shown)} dari {len(df):,} baris.")
