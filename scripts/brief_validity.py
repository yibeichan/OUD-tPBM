"""Flag BRIEF-A administrations that fail a validity scale.

Every BRIEF-A T-score export carries three validity flags per
administration: inconsistency, infrequency, and negativity. An
administration is valid only when all three read acceptable. The real
t-score files' infrequency column header carries a trailing space
("Infrequency_acceptable "), so column names are stripped before lookup.
"""
import pandas as pd

VALIDITY_FLAGS = ("inconsistency_acceptable", "Infrequency_acceptable", "negativity_acceptable")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.upper() == "TRUE"


def valid_administration_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask, True where every validity flag is acceptable.

    Raises KeyError naming any flag column absent from ``df`` (after
    stripping whitespace from the column names).
    """
    column_by_stripped_name = {}
    for col in df.columns:
        column_by_stripped_name.setdefault(str(col).strip(), col)

    mask = pd.Series(True, index=df.index)
    for flag in VALIDITY_FLAGS:
        if flag not in column_by_stripped_name:
            raise KeyError(flag)
        mask &= _as_bool(df[column_by_stripped_name[flag]])
    return mask
