"""Workflow steps and the session-state rules that connect them.

Session keys: df, dataset_name, dataset_key, result, comparison, report,
fit_error, step, flash. Loading a different dataset clears everything that was
derived from the previous one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

WELCOME = "beranda"
SAMPLE_KEY = "sample"
REGRESSION_WIDGET_KEYS = ("reg_target", "reg_predictors", "reg_conf")


@dataclass(frozen=True)
class Step:
    key: str
    number: str
    title: str
    subtitle: str


STEPS = (
    Step("data", "01", "Data", "Eksplorasi & profil data"),
    Step("regresi", "02", "Regresi", "Spesifikasi & estimasi OLS"),
    Step("diagnostik", "03", "Diagnostik", "Periksa asumsi model"),
    Step("laporan", "04", "Laporan", "Ringkas & ekspor"),
)
STEP_BY_KEY = {s.key: s for s in STEPS}


def go(step: str) -> None:
    st.session_state.step = step


def lock_reason(step: str) -> str | None:
    """Why a step cannot be opened yet, or None when it can."""
    if step == WELCOME:
        return None
    if st.session_state.get("df") is None:
        return "Muat dataset terlebih dahulu."
    if step in ("diagnostik", "laporan") and st.session_state.get("result") is None:
        return "Estimasi model di langkah Regresi terlebih dahulu."
    return None


def current_step() -> str:
    step = st.session_state.get("step", WELCOME)
    if st.session_state.get("df") is None:
        return WELCOME
    if step != WELCOME and (step not in STEP_BY_KEY or lock_reason(step)):
        return "data"
    return step


def set_dataset(df: pd.DataFrame, name: str, key: str) -> None:
    """Make ``df`` the active dataset and open the Data step."""
    if st.session_state.get("dataset_key") != key:
        for k in REGRESSION_WIDGET_KEYS:
            st.session_state.pop(k, None)
        st.session_state.update(
            df=df, dataset_name=name, dataset_key=key, result=None, comparison=None, report=None, fit_error=None
        )
        st.session_state.flash = f"Dataset dimuat: {len(df):,} baris × {df.shape[1]} kolom."
    st.session_state.step = "data"


def is_synthetic() -> bool:
    return st.session_state.get("dataset_key") == SAMPLE_KEY
