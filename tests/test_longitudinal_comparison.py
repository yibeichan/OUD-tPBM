"""Tests for the assessment longitudinal-comparison tables.

``get_subjects_with_all_timepoints`` decides the row order of every table and
figure ``06_longitudinal_comparison.py`` emits. It sorted correctly while
subjects were zero-padded, because lexicographic and numeric order agree on
``01``…``12``; canonical codes are unpadded, so ``sorted()`` places ``S10``
before ``S2`` and the cohort comes out mis-ordered. ``subject_ids`` exists to
carry that ordering, and this pins the call site to it.

The module is loaded by path: ``scripts/assessments/`` is not a package, and
adding an ``__init__.py`` would change what gets packaged. Unlike the
longitudinal loaders, this script calls ``load_dotenv()`` at import, exits when
``OUTPUT_DIR`` is unset, and creates its score directories as an import side
effect. ``load_dotenv()`` does not override a variable already set in the
environment, so pointing ``OUTPUT_DIR`` at ``tmp_path`` before the import is
what keeps these tests off the configured output root.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "assessments"
    / "06_longitudinal_comparison.py"
)

# Spans the boundary where lexicographic and numeric order diverge: S9 sorts
# after S10 numerically but before it as text. Listed in the expected order.
COHORT = ["S1", "S2", "S3", "S9", "S10", "S11", "S12"]


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    """The script module, imported with its output root under ``tmp_path``."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("longitudinal_comparison", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(subjects):
    return pd.DataFrame({"subject": list(subjects), "GEC_t": range(len(subjects))})


def test_common_subjects_come_back_in_numeric_order(comparison):
    data = {"baseline": _frame(COHORT), "post": _frame(COHORT)}

    result = comparison.get_subjects_with_all_timepoints(data, ["baseline", "post"])

    assert result == COHORT


def test_non_canonical_subject_is_reported_readably(comparison):
    # A subject column holding values that are not S-codes reaches the sort.
    # That has to read as an instruction, not a traceback.
    # Exactly one non-canonical value, so the reported element is
    # deterministic: the intersection is a set, and which member the sort key
    # rejects first otherwise varies with the process hash seed.
    legacy = ["S1", "P01"]
    data = {"baseline": _frame(legacy), "post": _frame(legacy)}

    with pytest.raises(SystemExit) as excinfo:
        comparison.get_subjects_with_all_timepoints(data, ["baseline", "post"])

    message = str(excinfo.value)
    assert "P01" in message
    assert "S1" in message


def test_invalid_brief_administrations_are_dropped(comparison, tmp_path):
    """An administration that fails a validity scale stays out of the table."""
    t_dir = tmp_path / "01_assessments" / "t-scores"
    t_dir.mkdir(parents=True)
    pd.DataFrame({
        "subject": ["S91", "S92"], "total_score": [130, 75], "n_items": [75, 75],
        "GEC_t": [59, 35],
        "inconsistency_acceptable": [True, True],
        "Infrequency_acceptable ": [True, False],  # the real header's trailing space
        "negativity_acceptable": [True, True],
    }).to_csv(t_dir / "executive_function_post_subscale_t-scores.csv", index=False)

    df = comparison.load_t_score_data("post")

    assert df["subject"].tolist() == ["S91"]
    assert comparison.INVALID_ADMINISTRATIONS["post"] == ["S92"]
