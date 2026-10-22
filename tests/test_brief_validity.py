from pathlib import Path

import pandas as pd
import pytest

from scripts.brief_validity import valid_administration_mask


def test_padded_header_and_string_flags():
    df = pd.DataFrame({
        "inconsistency_acceptable": [True, True, "TRUE"],
        "Infrequency_acceptable ": [True, "False", "true"],   # trailing space, as in the real file
        "negativity_acceptable": ["True", True, False],
    })
    assert valid_administration_mask(df).tolist() == [True, False, False]


def test_missing_flag_column_raises():
    df = pd.DataFrame({"inconsistency_acceptable": [True], "negativity_acceptable": [True]})
    with pytest.raises(KeyError, match="Infrequency_acceptable"):
        valid_administration_mask(df)


def test_real_tscore_files_validity(reference_tree, pinned):
    tscore_dir = reference_tree / "01_assessments" / "t-scores"
    if not tscore_dir.is_dir():
        pytest.skip(f"{tscore_dir} does not exist")

    expected_invalid = pinned("brief_invalid_administrations")
    for timepoint, invalid_subjects in expected_invalid.items():
        df = pd.read_csv(tscore_dir / f"executive_function_{timepoint}_subscale_t-scores.csv")
        df = df[df["n_items"] > 0]
        mask = valid_administration_mask(df)
        assert set(df.loc[~mask, "subject"]) == set(invalid_subjects), timepoint
