"""Small rendering helpers shared across views."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from datalens.formatting import fmt_pvalue
from views import ui
from views.state import STEP_BY_KEY, go, is_synthetic


def render_safely(render: Callable[[pd.DataFrame], None], df: pd.DataFrame) -> None:
    """Last-resort error boundary so a bug in one section cannot take down the others.

    Streamlit's own control flow (st.rerun, st.stop) raises BaseException
    subclasses, which this deliberately does not catch.
    """
    try:
        render(df)
    except Exception as exc:  # noqa: BLE001 - boundary: show, don't crash
        st.error(
            "Terjadi kesalahan tak terduga di bagian ini. Bagian lain tetap dapat digunakan. "
            "Coba periksa data atau pilihan variabel.",
            icon=":material/error:",
        )
        with st.expander("Detail teknis"):
            st.exception(exc)


def step_header(step: str, title: str, description: str, cta: tuple[str, str] | None = None) -> None:
    """Breadcrumb, eyebrow, page title and the screen's one primary action.

    ``cta`` is (button label, step to open).
    """
    s = STEP_BY_KEY[step]
    name = st.session_state.get("dataset_name", "dataset")
    tag = "Data sintetis" if is_synthetic() else ""
    ui.breadcrumb(["Ruang kerja", name, s.title], tag)
    left, right = st.columns([3, 1.15], vertical_alignment="bottom", gap="medium")
    with left:
        ui.page_header(f"Langkah {s.number} dari 04 · {s.subtitle}", title, description)
    if cta:
        label, target = cta
        with right, st.container(horizontal=True, horizontal_alignment="right"):
            st.button(label, key=f"cta_{step}", type="primary", icon=":material/arrow_forward:",
                      on_click=go, args=(target,))


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """Bordered metric cards that wrap onto new lines instead of truncating."""
    with st.container(horizontal=True, gap="small"):
        for label, value, help_text in items:
            st.metric(label, value, help=help_text, border=True, width="content")


def p_label(p: float) -> str:
    """'< 0.001' or '= 0.042', for inline 'p …' text."""
    s = fmt_pvalue(p)
    return s if s.startswith("<") else f"= {s}"
