"""DataLens Streamlit entry point: sidebar (workflow navigation) and step routing.

All statistics come from datalens.*; views.* only render them.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens import __version__
from datalens.comparison import formula
from datalens.formatting import fmt_num
from datalens.regression import RegressionResult
from views import data_page, diagnostics, regression, report, ui, welcome
from views.components import render_safely
from views.state import STEPS, WELCOME, current_step, go, is_synthetic, lock_reason

st.set_page_config(page_title="DataLens", page_icon=":material/query_stats:", layout="wide")

VIEWS = {
    WELCOME: welcome.render,
    "data": data_page.render,
    "regresi": regression.render,
    "diagnostik": diagnostics.render,
    "laporan": report.render,
}
NAV_ICONS = {
    "current": ":material/radio_button_checked:",
    "complete": ":material/check_circle:",
    "available": ":material/radio_button_unchecked:",
    "locked": ":material/lock:",
}


def _nav_state(step: str, current: str) -> str:
    if step == current:
        return "current"
    if lock_reason(step):
        return "locked"
    done = {
        "data": True,
        "regresi": st.session_state.get("result") is not None,
        "laporan": st.session_state.get("report") is not None,
    }
    return "complete" if done.get(step) else "available"


def sidebar(df: pd.DataFrame | None, current: str) -> None:
    with st.sidebar:
        st.html(f'<div class="dl-brand">{ui.LOGO}<div><b>DataLens</b><span>Lab statistik</span></div></div>')

        st.html('<span class="dl-eyebrow">Dataset</span>')
        if df is None:
            st.html(
                '<div class="dl-card-dashed"><b style="font-size:13.5px">Belum ada dataset</b>'
                '<div style="font-size:12.5px;color:var(--dl-muted);margin-top:2px">Unggah CSV atau gunakan '
                "contoh data untuk membuka alur kerja.</div></div>"
            )
        else:
            chip = '<span class="dl-chip dl-chip-amber">Sintetis</span>' if is_synthetic() else ""
            st.html(
                f'<div class="dl-card"><div style="font-size:13.5px;font-weight:600;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap">{ui.esc(st.session_state.dataset_name)}</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px">'
                f'<span class="dl-mono" style="font-size:11.5px;color:var(--dl-muted)">{len(df):,} baris · '
                f"{df.shape[1]} kolom</span>{chip}</div></div>"
            )
            st.button("Ganti dataset", key="replace_dataset", type="tertiary", icon=":material/swap_horiz:",
                      on_click=go, args=(WELCOME,))

        st.html('<span class="dl-eyebrow">Alur kerja</span>')
        for s in STEPS:
            state = _nav_state(s.key, current)
            with st.container(key=f"nav-{state}-{s.key}"):
                st.button(
                    f"{s.number} · {s.title}",
                    key=f"nav_{s.key}",
                    icon=NAV_ICONS[state],
                    disabled=state == "locked",
                    help=lock_reason(s.key) or s.subtitle,
                    width="stretch",
                    on_click=go,
                    args=(s.key,),
                )

        result: RegressionResult | None = st.session_state.get("result")
        if result is not None:
            st.html(
                '<div style="border-top:1px solid var(--dl-border);padding-top:12px;margin-top:4px;display:flex;'
                'flex-direction:column;gap:4px"><span class="dl-eyebrow">Model saat ini</span>'
                f'<span class="dl-mono" style="font-size:12px;line-height:1.5">'
                f"{ui.esc(formula(result.target, result.predictors))}</span>"
                f'<span style="font-size:12.5px;color:var(--dl-emerald-dark)">R² {fmt_num(result.r_squared, 3)}'
                f" · n = {result.n_obs:,}</span></div>"
            )
        st.caption(f"DataLens v{__version__} · Python · Streamlit")


def main() -> None:
    ui.inject_css()
    df: pd.DataFrame | None = st.session_state.get("df")
    step = current_step()
    sidebar(df, step)
    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash, icon=":material/check_circle:")
    render_safely(VIEWS[step], df)


main()
