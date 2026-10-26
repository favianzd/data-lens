"""End-to-end smoke test: run app.py headlessly and walk through every workflow step.

Catches what unit tests cannot: import errors, widget wiring, session-state
handling and exceptions in any view (each view is wrapped in an error
boundary, so failures surface as st.error elements, not as exceptions).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from datalens.sample_data import TARGET, TRUE_COEFFICIENTS

APP = Path(__file__).resolve().parents[1] / "app.py"
BOUNDARY_TEXT = "kesalahan tak terduga"
PREDICTORS = list(TRUE_COEFFICIENTS)


def _by_label(elements, label):
    matches = [e for e in elements if e.label == label]
    assert matches, f"widget {label!r} not found"
    return matches[0]


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    errors = [e.value for e in at.error]
    assert not any(BOUNDARY_TEXT in str(e) for e in errors), errors


def _open(at: AppTest, step: str) -> None:
    at.sidebar.button(key=f"nav_{step}").click().run()
    _assert_clean(at)


def _load_sample(at: AppTest) -> None:
    _by_label(at.button, "Gunakan contoh data sintetis").click().run()
    _assert_clean(at)


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at


def test_empty_state_renders_with_later_steps_locked(app):
    _assert_clean(app)
    assert _by_label(app.button, "Gunakan contoh data sintetis")
    for step in ("data", "regresi", "diagnostik", "laporan"):
        assert app.sidebar.button(key=f"nav_{step}").disabled


def test_full_workflow_through_every_step(app):
    _load_sample(app)
    assert app.session_state["dataset_name"] == "contoh_sintetis.csv"
    assert app.session_state["step"] == "data"
    assert any(h.value == "Statistik deskriptif" for h in app.subheader)
    assert app.sidebar.button(key="nav_diagnostik").disabled  # no model yet

    # Regresi: fit the 3-predictor model.
    _open(app, "regresi")
    _by_label(app.multiselect, "Prediktor (X)").set_value(PREDICTORS)
    _by_label(app.button, "Estimasi model").click().run()
    _assert_clean(app)
    result = app.session_state["result"]
    assert result.target == TARGET and list(result.predictors) == PREDICTORS
    assert any(h.value == "Tabel koefisien" for h in app.subheader)
    assert not app.warning or not any("Spesifikasi berubah" in w.value for w in app.warning)

    # Changing the specification marks the results stale instead of hiding them.
    _by_label(app.multiselect, "Prediktor (X)").set_value(PREDICTORS[:1]).run()
    _assert_clean(app)
    assert any("Spesifikasi berubah" in w.value for w in app.warning)
    assert list(app.session_state["result"].predictors) == PREDICTORS
    _by_label(app.multiselect, "Prediktor (X)").set_value(PREDICTORS).run()

    # Perbandingan (tab in the Regresi step): prefilled from the fitted model (A = first predictor, B = all).
    assert _by_label(app.multiselect, "Model A: prediktor").value == PREDICTORS[:1]
    _by_label(app.selectbox, "Pengulangan").set_value(1)
    _by_label(app.button, "Bandingkan model").click().run()
    _assert_clean(app)
    assert app.session_state["comparison"].models[1].predictors == tuple(PREDICTORS)

    # Diagnostik follows the fitted model.
    _open(app, "diagnostik")
    assert any(h.value == "Diagnostik visual" for h in app.subheader)

    # Returning to Regresi keeps the fitted specification on screen.
    _open(app, "regresi")
    assert _by_label(app.multiselect, "Prediktor (X)").value == PREDICTORS

    # Laporan: build the report without figures (keeps the test fast).
    _open(app, "laporan")
    _by_label(app.checkbox, "Sertakan grafik interaktif").uncheck()
    _by_label(app.button, "Buat laporan").click().run()
    _assert_clean(app)
    html = app.session_state["report"]["html"]
    assert html.startswith("<!doctype html>") and "4. Perbandingan model" in html


def test_invalid_specification_shows_message_not_crash(app):
    _load_sample(app)
    _open(app, "regresi")
    _by_label(app.button, "Estimasi model").click().run()  # no predictors selected
    _assert_clean(app)
    assert any("minimal satu prediktor" in e.value for e in app.error)
    assert app.session_state["result"] is None


def test_replacing_dataset_returns_to_welcome(app):
    _load_sample(app)
    app.sidebar.button(key="replace_dataset").click().run()
    _assert_clean(app)
    assert app.session_state["step"] == "beranda"
    assert any("Dataset aktif" in i.value for i in app.info)
