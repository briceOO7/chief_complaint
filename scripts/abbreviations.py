"""
Local abbreviation expander for CEDIS chief-complaint pre-processing.

Each dataset can ship its own abbreviations CSV under data/abbreviations/.
The CSV must have columns:
  abbreviation   — the short form
  expansion      — the long form to substitute
  notes          — optional free-text notes (ignored at runtime)
  case_sensitive — optional; set to "true" to match exact case only
                   (default: case-insensitive, whole-word match)

Usage:
    from abbreviations import load_abbreviations, expand_abbreviations

    abbrevs = load_abbreviations("data/abbreviations/medevac_abbreviations.csv")
    clean   = expand_abbreviations("pt here for FV and RPNV", abbrevs)
    # → "pt here for follow-up visit and routine pre-natal visit"
"""

import re
from pathlib import Path

import pandas as pd


def load_abbreviations(path: str | Path) -> dict[str, tuple[str, bool]]:
    """
    Load an abbreviations CSV and return a dict mapping
    upper-cased abbreviation → (expansion, case_sensitive).
    Rows with blank abbreviations or expansions are skipped.
    """
    df = pd.read_csv(Path(path), dtype=str)
    df = df.dropna(subset=["abbreviation", "expansion"])
    df = df[df["abbreviation"].str.strip() != ""]
    df = df[df["expansion"].str.strip() != ""]
    result = {}
    for _, row in df.iterrows():
        abbr = row["abbreviation"].strip()
        exp  = row["expansion"].strip()
        cs   = str(row.get("case_sensitive", "")).strip().lower() == "true"
        result[abbr.upper()] = (exp, cs, abbr)   # store original casing too
    return result


def expand_abbreviations(text: str, abbrevs: dict[str, tuple[str, bool, str]]) -> str:
    """
    Replace each abbreviation in *text* with its expansion.
    Matching is whole-word for all entries.
    Entries with case_sensitive=true match exact case only;
    all others match case-insensitively.
    Longer abbreviations are applied before shorter ones to avoid
    partial matches (e.g. "RPNV" before "RPN").
    """
    if not abbrevs or not text:
        return text

    for key in sorted(abbrevs, key=len, reverse=True):
        expansion, case_sensitive, original = abbrevs[key]
        pattern = rf"(?<!\w){re.escape(original)}(?!\w)"
        flags   = 0 if case_sensitive else re.IGNORECASE
        text    = re.sub(pattern, expansion, text, flags=flags)

    return text


def expand_series(series: pd.Series, abbrevs: dict[str, str]) -> pd.Series:
    """Apply expand_abbreviations to every element of a pandas Series."""
    if not abbrevs:
        return series
    return series.fillna("").apply(lambda t: expand_abbreviations(str(t), abbrevs))
