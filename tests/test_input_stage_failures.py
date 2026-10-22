"""Behaviour and assessment inputs raise instead of being skipped.

Covers the behaviour scripts, the assessment loader and subject-column
selection. Every fixture is under tmp_path.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from subject_ids import subject_column  # noqa: E402


def _load(relpath, tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    path = _SCRIPTS / relpath
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# E13
# --------------------------------------------------------------------------

def test_subject_column_is_selected_by_name():
    assert subject_column(["Item_Number", "Question", "S7"]) == "S7"
    assert subject_column(["S7", "Item_Number", "Question"]) == "S7"


@pytest.mark.parametrize("columns", [["Item_Number", "Question"], ["Item_Number", "S1", "S2"]])
def test_subject_column_requires_exactly_one(columns):
    with pytest.raises(ValueError, match="exactly one subject column"):
        subject_column(columns)


@pytest.fixture
def loader(tmp_path, monkeypatch):
    return _load("assessments/00_load_assessment_data.py", tmp_path, monkeypatch)


def test_known_responses_still_convert(loader):
    assert loader.convert_response_to_numeric("N") == 1
    # A double response scores the lower code.
    assert loader.convert_response_to_numeric("N S") == 1
    assert np.isnan(loader.convert_response_to_numeric("-"))
    assert loader.convert_response_to_numeric("3", scale_value="0-3") == 3.0


def test_unconvertible_response_raises(loader):
    with pytest.raises(ValueError, match="'D'"):
        loader.convert_response_to_numeric("D", scale_value="0-3")


def test_missing_assessment_file_raises(loader, tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ASSESSMENT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="phq9_baseline.csv"):
        loader.process_assessment_data("phq9", ["phq9_baseline.csv"])


def test_sheet_without_subject_columns_raises(loader, tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ASSESSMENT_DIR", tmp_path)
    pd.DataFrame({"Item_Number": [1], "Question": ["q"], "P01": ["N"]}).to_csv(
        tmp_path / "phq9_baseline.csv", index=False)
    with pytest.raises(ValueError, match="no canonical subject columns"):
        loader.process_assessment_data("phq9", ["phq9_baseline.csv"])


def test_unreadable_sheet_raises(loader, tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ASSESSMENT_DIR", tmp_path)
    (tmp_path / "phq9_baseline.csv").write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        loader.process_assessment_data("phq9", ["phq9_baseline.csv"])


# --------------------------------------------------------------------------
# E12
# --------------------------------------------------------------------------

@pytest.mark.parametrize("script,task", [
    ("behav/01_stroop_behav.py", "stroop"), ("behav/01_wcst_behav.py", "wcst"),
])
def test_missing_event_file_raises(tmp_path, monkeypatch, script, task):
    module = _load(script, tmp_path, monkeypatch)
    (tmp_path / "base" / "data" / task).mkdir(parents=True)
    pd.DataFrame([("f.snirf", "S1", "aaaaaaa", 1)],
                 columns=["filename", "subj_id", "task_id", "task_order"]).to_csv(
        tmp_path / "base" / "data" / f"{task}_task_info.csv", index=False)
    with pytest.raises(FileNotFoundError, match="S1"):
        module.main(task)


@pytest.mark.parametrize("script,task", [
    ("behav/01_stroop_behav.py", "stroop"), ("behav/01_wcst_behav.py", "wcst"),
])
def test_failing_session_raises(tmp_path, monkeypatch, script, task):
    module = _load(script, tmp_path, monkeypatch)
    (tmp_path / "base" / "data" / task).mkdir(parents=True)
    pd.DataFrame([("f.snirf", "S1", "aaaaaaa", 1)],
                 columns=["filename", "subj_id", "task_id", "task_order"]).to_csv(
        tmp_path / "base" / "data" / f"{task}_task_info.csv", index=False)
    (tmp_path / "base" / "data" / task / "NeuroLume_S1_aaaaaaa_task_events.tsv").write_text(
        "no_event_column\n1\n")
    with pytest.raises(KeyError):
        module.main(task)
