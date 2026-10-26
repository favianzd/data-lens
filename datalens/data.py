"""Loading and profiling of uploaded CSV data.

This module is UI-agnostic: it never imports Streamlit, so it can be tested
directly and reused by the report generator.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# Tried in order. utf-8-sig also handles the BOM that Excel writes.
ENCODINGS = ("utf-8-sig", "latin-1")


class DataLoadError(ValueError):
    """Raised when an uploaded file cannot be parsed into a usable table."""


def load_csv(raw: bytes, sep: str | None = None, decimal: str = ".") -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame.

    ``sep=None`` lets pandas sniff the delimiter, which covers the common
    ``,`` and ``;`` (Excel with Indonesian locale) cases.
    """
    if not raw or not raw.strip():
        raise DataLoadError("File kosong: tidak ada data yang dapat dibaca.")

    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                sep=sep,
                decimal=decimal,
                encoding=encoding,
                engine="python",
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError("File tidak memiliki kolom atau baris data.") from exc
        except (pd.errors.ParserError, ValueError) as exc:
            raise DataLoadError(f"Format CSV tidak valid: {exc}") from exc
    else:
        raise DataLoadError(f"Encoding file tidak dikenali: {last_error}")

    columns = [str(c).strip() for c in df.columns]
    if len(set(columns)) != len(columns):
        raise DataLoadError("Terdapat nama kolom duplikat setelah spasi di tepi nama dihapus.")
    df.columns = columns

    if df.shape[0] == 0:
        raise DataLoadError("File hanya berisi header, tanpa baris data.")
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Columns usable as numeric model variables (booleans excluded)."""
    return [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]


@dataclass(frozen=True)
class DatasetProfile:
    n_rows: int
    n_cols: int
    n_numeric: int
    n_duplicate_rows: int
    n_rows_with_missing: int
    n_missing_cells: int
    columns: pd.DataFrame


def profile(df: pd.DataFrame) -> DatasetProfile:
    """Structural summary of a dataset: size, types, missingness, duplicates."""
    n_missing = df.isna().sum()
    columns = pd.DataFrame(
        {
            "tipe": df.dtypes.astype(str),
            "n_hilang": n_missing,
            "persen_hilang": (n_missing / len(df) * 100) if len(df) else 0.0,
            "n_unik": df.nunique(dropna=True),
        }
    )
    columns.index.name = "kolom"
    return DatasetProfile(
        n_rows=len(df),
        n_cols=df.shape[1],
        n_numeric=len(numeric_columns(df)),
        n_duplicate_rows=int(df.duplicated().sum()),
        n_rows_with_missing=int(df.isna().any(axis=1).sum()),
        n_missing_cells=int(n_missing.sum()),
        columns=columns,
    )
