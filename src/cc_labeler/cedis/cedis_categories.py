"""
cedis_categories.py
--------------------
Lookup utilities for CEDIS code → category mapping.

Category rules (from data/cedis_codes.csv):
  - ENT sub-domains (Ears / Mouth/Throat/Neck / Nose) are merged into "ENT".
  - All other clinical domains map to their domain name.
  - Every General and Minor code maps to its own complaint text, so that
    Fever, Unknown, Follow-up/Return Visit, etc. each remain a distinct category.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

_CEDIS_CSV = Path(__file__).resolve().parents[3] / "data" / "cedis_codes.csv"


@lru_cache(maxsize=1)
def _load_lookup() -> dict[str, str]:
    """Return {str(code): category} mapping, loaded once and cached."""
    df = pd.read_csv(_CEDIS_CSV, dtype=str)
    if "category" not in df.columns:
        raise ValueError(
            f"'category' column not found in {_CEDIS_CSV}. "
            "Ensure data/cedis_codes.csv includes the category column."
        )
    df["code"] = df["code"].str.strip().str.lstrip("0")
    return dict(zip(df["code"], df["category"]))


def get_category(code: str | int | float | None) -> str | None:
    """
    Return the category for a CEDIS code, or None if not found.

    Accepts codes as strings (with or without leading zeros), ints, or floats.
    Examples:
        get_category("251")   → "Gastrointestinal"
        get_category("001")   → "Cardiovascular"
        get_category(1)       → "Cardiovascular"
        get_category("999")   → "Unknown"
        get_category("888")   → "Follow-up/Return Visit"
        get_category("052")   → "ENT"
    """
    if code is None:
        return None
    # Normalise to a string without leading zeros or trailing .0
    normalized = str(code).strip().rstrip("0").rstrip(".") if "." in str(code) else str(code).strip()
    # Strip leading zeros for lookup
    key = normalized.lstrip("0") or "0"
    return _load_lookup().get(key)


def apply_category_column(
    df: pd.DataFrame,
    code_col: str = "cedis_code",
    out_col: str = "cedis_category",
) -> pd.DataFrame:
    """
    Add a cedis_category column to df by looking up each value in code_col.

    Inserts the new column immediately after code_col if it exists.
    Rows with no matching code receive an empty string.

    Parameters
    ----------
    df : pd.DataFrame
    code_col : column containing CEDIS codes (default "cedis_code")
    out_col  : name of the new category column (default "cedis_category")

    Returns
    -------
    A copy of df with the new column added.
    """
    if code_col not in df.columns:
        raise KeyError(f"Column '{code_col}' not found in DataFrame.")

    df = df.copy()
    df[out_col] = df[code_col].map(lambda c: get_category(c) or "")

    # Reorder: insert out_col right after code_col
    cols = list(df.columns)
    if out_col in cols:
        cols.remove(out_col)
    insert_at = cols.index(code_col) + 1
    cols.insert(insert_at, out_col)
    return df[cols]


def list_categories() -> list[str]:
    """Return all unique categories in sorted order."""
    return sorted(set(_load_lookup().values()))
