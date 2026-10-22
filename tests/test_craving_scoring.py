"""The craving (BSCS) total sums only the six 0-4 ratings.

The sheet's ten rows also hold two substance-type codes and two occurrence
counts, which must stay out of the total. Every fixture is synthetic and
under tmp_path; subject codes from S90 up are outside the cohort's range.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assessments" / "04_analyze_craving.py"

QUESTIONS = [
    "Primary Substance Type",
    "INTENSITY of craving in past 24 hours",
    "FREQUENCY of craving in past 24 hours",
    "LENGTH of time spent craving in past 24 hours",
    "NUMBER of times craving occurred in past 24 hours",
    "Second Substance Type (if applicable)",
    "INTENSITY of second substance craving",
    "FREQUENCY of second substance craving",
    "LENGTH of time spent craving second substance",
    "NUMBER of times craving second substance occurred",
]


@pytest.fixture
def craving(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    spec = importlib.util.spec_from_file_location("analyze_craving", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(subject, values):
    """One subject-timepoint as 00_load_assessment_data.py serialises it."""
    return [{"Item_Number": i + 1, "Question": q, subject: v}
            for i, (q, v) in enumerate(zip(QUESTIONS, values))]


def test_total_sums_only_the_ratings(craving):
    # Summing every row would give 24.
    scores = craving.calculate_craving_scores(
        {"baseline": _records("S91", [3, 2, 1, 0, 4, 6, 0, 1, 2, 5])}, "baseline")

    assert scores["total_score"] == 6
    assert scores["n_items"] == 6
    assert scores["separate"] == {
        "primary_substance_type": 3, "second_substance_type": 6,
        "craving_occurrences_primary": 4, "craving_occurrences_second": 5}


def test_blank_second_substance_block_is_counted_in_n_items(craving):
    # No second substance: summing every row would give 10.
    nan = np.nan
    scores = craving.calculate_craving_scores(
        {"baseline": _records("S92", [3, 0, 2, 1, 4, 0, nan, nan, nan, nan])}, "baseline")

    assert scores["total_score"] == 3
    assert scores["n_items"] == 3
    assert np.isnan(scores["separate"]["craving_occurrences_second"])


def test_unrecognised_item_raises(craving):
    records = _records("S91", [3, 2, 1, 0, 4, 6, 0, 1, 2, 5])
    records[3]["Question"] = "DURATION of craving"
    with pytest.raises(ValueError, match="Unrecognised craving item"):
        craving.calculate_craving_scores({"baseline": records}, "baseline")


def test_bscs_total_sums_the_three_primary_ratings(craving):
    # Second-substance ratings (0, 1, 2 here) are excluded: 2 + 1 + 0 = 3.
    scores = craving.calculate_craving_scores(
        {"baseline": _records("S91", [3, 2, 1, 0, 4, 6, 0, 1, 2, 5])}, "baseline")

    assert scores["bscs_total"] == 3
    assert scores["bscs_n_items"] == 3


def test_bscs_total_excludes_second_substance_values(craving):
    scores = craving.calculate_craving_scores(
        {"baseline": _records("S93", [2, 1, 3, 2, 6, 4, 3, 3, 3, 3])}, "baseline")

    assert scores["bscs_total"] == 6
    assert scores["total_score"] == 15


def test_blank_second_substance_does_not_affect_bscs_total(craving):
    nan = np.nan
    scores = craving.calculate_craving_scores(
        {"baseline": _records("S92", [3, 0, 2, 1, 4, 0, nan, nan, nan, nan])}, "baseline")

    assert scores["bscs_total"] == 3
    assert scores["bscs_n_items"] == 3


def test_bscs_total_is_nan_when_a_primary_rating_is_missing(craving):
    nan = np.nan
    scores = craving.calculate_craving_scores(
        {"baseline": _records("S92", [3, 2, nan, 1, 2, 0, 1, 2, 3, 4])}, "baseline")

    assert np.isnan(scores["bscs_total"])
    assert scores["bscs_n_items"] == 2
    assert scores["total_score"] == 2 + 1 + 6  # six-rating total unchanged


def test_analyze_all_subjects_has_bscs_columns_and_summary(craving):
    data = {"S91": {"baseline": _records("S91", [3, 2, 1, 0, 4, 6, 0, 1, 2, 5])},
            "S92": {"baseline": _records("S92", [3, 2, np.nan, 1, 2, 0, 0, 0, 0, 0])}}
    df = craving.analyze_all_subjects(data)

    assert df.set_index("subject")["bscs_total"].iloc[0] == 3
    assert np.isnan(df.set_index("subject")["bscs_total"].iloc[1])
    assert list(df["bscs_n_items"]) == [3, 2]
    summary = craving.generate_summary_statistics(df)
    assert summary["baseline"]["bscs_total"]["n"] == 1
    assert summary["baseline"]["bscs_total"]["mean"] == 3
    assert summary["baseline"]["total_score"]["mean"] == pytest.approx(
        df["total_score"].mean())
