"""The identifier contract: one format, anchored parsing, numeric ordering."""
import pytest

from subject_ids import (
    extract_session_id,
    extract_subject_id,
    is_subject_id,
    sort_subjects,
    standardize_subject_id,
    subject_sort_key,
)


@pytest.mark.parametrize("value", ["S1", "S5", "S12"])
def test_accepts_canonical_codes(value):
    assert is_subject_id(value)


@pytest.mark.parametrize("value", ["S01", "S0", "RAW_01", "S", "s1", "S1_000000a", ""])
def test_rejects_non_canonical_codes(value):
    assert not is_subject_id(value)


def test_sorts_numerically_not_lexicographically():
    # sorted() alone gives S1, S10, S11, S12, S2, S5 -- the reason this exists.
    assert sort_subjects(["S10", "S2", "S1", "S12", "S5", "S11"]) == [
        "S1", "S2", "S5", "S10", "S11", "S12",
    ]


def test_sort_key_is_the_integer():
    assert subject_sort_key("S7") == 7
    assert subject_sort_key("S12") == 12


@pytest.mark.parametrize(
    "value,expected",
    [("S1", "S1"), ("s1", "S1"), ("S01", "S1"), ("S-1", "S1"), (" S1 ", "S1")],
)
def test_standardize_normalizes_padding_and_case(value, expected):
    assert standardize_subject_id(value) == expected


def test_standardize_rejects_rather_than_falling_back():
    # The previous implementation returned its input unchanged on no match,
    # which let a malformed identifier travel silently into a filename.
    with pytest.raises(ValueError, match="not a subject identifier"):
        standardize_subject_id("not-an-id")


@pytest.mark.parametrize(
    "filename,subject,session",
    [
        ("NeuroLume_S1_000000a_5.snirf", "S1", "000000a"),
        ("NeuroLume_S12_0000014_task_events.tsv", "S12", "0000014"),
        ("S1_000000a_glm_contrasts.csv", "S1", "000000a"),
        ("S5_0000009_results.json", "S5", "0000009"),
    ],
)
def test_extracts_both_fields_from_every_filename_family(filename, subject, session):
    assert extract_subject_id(filename) == subject
    assert extract_session_id(filename) == session


@pytest.mark.parametrize(
    "filename,session",
    [
        ("mapping_stroop_01_000000a.json", "000000a"),
        ("mapping_wcst_12_0000014.json", "0000014"),
        ("roi_mapping_stroop_01_000000a.json", "000000a"),
        ("roi_mapping_wcst_12_0000014.json", "0000014"),
        # Post-rename, step2a emits an S-code in the subject position.
        ("mapping_stroop_S1_000000a.json", "000000a"),
    ],
)
def test_extracts_session_from_mapping_filenames(filename, session):
    assert extract_session_id(filename) == session


@pytest.mark.parametrize(
    "filename",
    ["channel_roi_mapping.json", "glm_to_parcellation_mapping.json"],
)
def test_aggregate_mappings_carry_no_session(filename):
    """Tree-wide mappings are not session-scoped and must not parse as one."""
    with pytest.raises(ValueError, match="no session identifier"):
        extract_session_id(filename)


def test_parsing_is_anchored_not_positional():
    # An extra leading component must not shift which token is read as the
    # subject; the old split('_') offsets did exactly that.
    with pytest.raises(ValueError):
        extract_subject_id("extra_NeuroLume_S1_000000a_5.snirf")


def test_extract_raises_on_unmatched_filename():
    with pytest.raises(ValueError, match="no subject identifier"):
        extract_subject_id("session_index.json")
    with pytest.raises(ValueError, match="no session identifier"):
        extract_session_id("session_index.json")
