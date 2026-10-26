"""Step 01 'Data': dataset overview followed by exploration."""

from __future__ import annotations

import pandas as pd

from views import explore, overview
from views.components import render_safely, step_header


def render(df: pd.DataFrame) -> None:
    step_header(
        "data",
        "Penjelajah Data",
        "Periksa struktur, kualitas, dan bentuk data sebelum memodelkan. Tidak ada yang diubah pada data Anda di sini.",
        cta=("Lanjut ke Regresi", "regresi"),
    )
    render_safely(overview.render, df)
    render_safely(explore.render, df)
