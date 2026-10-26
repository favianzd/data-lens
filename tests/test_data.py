import numpy as np
import pandas as pd
import pytest

from datalens.data import DataLoadError, load_csv, numeric_columns, profile


def test_load_comma_csv():
    df = load_csv(b"a,b\n1,2\n3,4\n")
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_load_semicolon_with_decimal_comma():
    df = load_csv(b"a;b\n1,5;2\n3,25;4\n", sep=";", decimal=",")
    assert df["a"].tolist() == [1.5, 3.25]


def test_semicolon_separator_is_sniffed():
    df = load_csv(b"a;b\n1;2\n3;4\n")
    assert list(df.columns) == ["a", "b"]


def test_latin1_fallback():
    df = load_csv("kota,nilai\nBogor\xe9,1\n".encode("latin-1"))
    assert df.shape == (1, 2)


def test_strips_column_name_whitespace():
    assert list(load_csv(b" a , b\n1,2\n").columns) == ["a", "b"]


@pytest.mark.parametrize("raw", [b"", b"   \n", b"a,b\n"])
def test_empty_or_header_only_rejected(raw):
    with pytest.raises(DataLoadError):
        load_csv(raw)


def test_numeric_columns_excludes_text_and_bool():
    df = pd.DataFrame({"i": [1, 2], "f": [1.0, np.nan], "s": ["a", "b"], "b": [True, False]})
    assert numeric_columns(df) == ["i", "f"]


def test_profile_counts():
    df = pd.DataFrame({"a": [1, 1, np.nan, 4], "b": ["x", "x", "y", None]})
    p = profile(df)
    assert (p.n_rows, p.n_cols, p.n_numeric) == (4, 2, 1)
    assert p.n_duplicate_rows == 1
    assert p.n_rows_with_missing == 2
    assert p.n_missing_cells == 2
    assert p.columns.loc["a", "n_hilang"] == 1
    assert p.columns.loc["a", "persen_hilang"] == pytest.approx(25.0)
