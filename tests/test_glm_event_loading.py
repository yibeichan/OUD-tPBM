"""The first-level GLM readers must survive padded event tables.

Some event tables carry whitespace-padded column names and padded values. A
bare read_csv leaves a column literally named "       event        ", so
events_df['event'] raises KeyError and takes the whole stage down. The
readers strip both, as the behavioural scripts do.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"
sys.path.insert(0, str(_BRAIN.parent))

PADDED = (
    "timestamp          \t event           \t duration  \t block_type\n"
    " 11.85\t start_experiment\t 423.44    \t nan       \n"
    " 11.86\t start_block     \t  31.73    \t cong_only \n"
    " 12.90\t start_trial     \t   1.00    \t cong_only \n"
)


def _is_mne_family(name):
    """True for the modules the isolation fixture evicts: mne and mne_nirs.

    Deliberately narrow. Evicting every module a by-path load pulled in also
    evicts scipy, whose C extensions cannot be re-imported once dropped from
    sys.modules (``scipy.optimize._highspy._core`` re-imports as a non-package
    and every later ``import scipy.stats`` dies), which breaks any test module
    that loads step1_glm_* after this one.

    The step1_glm_* modules themselves need no entry here: module_from_spec +
    exec_module never registers them in sys.modules, so each load is already
    fresh and there is nothing to evict.
    """
    return name.split(".")[0] in {"mne", "mne_nirs"}


@pytest.fixture(scope="module", autouse=True)
def _isolate_sys_modules():
    """Evict the mne and mne_nirs modules that loading step1_glm_* by path adds.

    Both readers import mne/mne_nirs/nilearn. Loaded via importlib into the
    shared pytest process, those would stay in sys.modules for the rest of
    the run; evicting them keeps later test modules independent of this
    one. Module-scoped, so it wraps both parametrized reader cases.
    """
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if _is_mne_family(name):
            sys.modules.pop(name, None)


@pytest.fixture(params=["step1_glm_wcst.py", "step1_glm_stroop.py"])
def reader(request):
    """The module under test, loaded by path (scripts/brain is not a package)."""
    path = _BRAIN / request.param
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_padded_event_table_yields_usable_event_column(tmp_path, reader):
    events_file = tmp_path / "NeuroLume_S1_0000000_task_events.tsv"
    events_file.write_text(PADDED)

    df = reader.read_event_table(str(events_file))

    assert "event" in df.columns, "column names must be stripped"
    assert list(df["event"]) == ["start_experiment", "start_block", "start_trial"]
    # The filter every caller performs must now select a row.
    assert len(df[df["event"] == "start_trial"]) == 1


# A padded table's numeric columns are the second half of the problem. When a
# cell is blank, read_csv cannot type the column, so it arrives as object and
# every ``== 1`` against it compares a string to an int -- always False, never
# an error. Stripping the values alone leaves the column object dtype, so the
# trial-classification loop silently classifies nothing.
PADDED_NUMERIC = (
    "timestamp          \t event           \t response_outcome \t trial_type\n"
    " 12.90\t start_trial     \t 1.0              \t colour    \n"
    " 14.10\t start_trial     \t 0.0              \t shape     \n"
    " 15.30\t start_trial     \t                  \t number    \n"
)

# Not every padded column is numeric. A column that only *sometimes* parses as
# a number must stay text: blanket ``errors="coerce"`` would turn it into all
# NaN just as silently as the bug above, in the other direction.
PADDED_MIXED = (
    "timestamp          \t event           \t rule_change \n"
    " 12.90\t start_trial     \t yes         \n"
    " 14.10\t start_trial     \t 1           \n"
    " 15.30\t start_trial     \t no          \n"
)


def test_padded_numeric_column_becomes_numeric_with_nan_for_blank(tmp_path, reader):
    events_file = tmp_path / "NeuroLume_S1_0000000_task_events.tsv"
    events_file.write_text(PADDED_NUMERIC)

    df = reader.read_event_table(str(events_file))

    outcome = df["response_outcome"]
    assert pd.api.types.is_float_dtype(outcome), (
        f"padded numeric column stayed {outcome.dtype}; "
        "row['response_outcome'] == 1 would never match"
    )
    assert outcome.iloc[0] == 1
    assert outcome.iloc[1] == 0
    assert pd.isna(outcome.iloc[2]), "an all-whitespace cell must read as missing"
    # The comparison the trial-classification loop actually performs.
    assert (outcome == 1).sum() == 1


def test_textual_column_stays_text_and_stripped(tmp_path, reader):
    events_file = tmp_path / "NeuroLume_S1_0000000_task_events.tsv"
    events_file.write_text(PADDED_NUMERIC)

    df = reader.read_event_table(str(events_file))

    assert df["event"].dtype == object
    assert list(df["event"]) == ["start_trial"] * 3
    assert list(df["trial_type"]) == ["colour", "shape", "number"]


def test_partly_numeric_column_is_not_coerced_to_nan(tmp_path, reader):
    events_file = tmp_path / "NeuroLume_S1_0000000_task_events.tsv"
    events_file.write_text(PADDED_MIXED)

    df = reader.read_event_table(str(events_file))

    rule_change = df["rule_change"]
    assert rule_change.dtype == object, "a column that is not wholly numeric must stay text"
    assert list(rule_change) == ["yes", "1", "no"]


def test_real_event_tables_classify_trial_outcomes(reader, study_base_dir, pinned):
    """On the real WCST event tables, ``response_outcome`` reads as numeric.

    For each session listed under ``wcst_event_outcome_counts`` in the local
    pinned file, the ``start_trial`` rows split into correct (1), incorrect
    (0) and blank outcomes in the pinned numbers, so ``== 1`` finds the
    correct trials even when the table is padded.
    """
    from subject_ids import task_events_filename

    info = pd.read_csv(study_base_dir / "data" / "wcst_task_info.csv")
    for expected in pinned("wcst_event_outcome_counts"):
        row = info[(info.subj_id == expected["subject"])
                   & (info.task_order == expected["task_order"])]
        assert len(row) == 1, expected
        events_file = (study_base_dir / "data" / "wcst"
                       / task_events_filename(expected["subject"], row.task_id.iloc[0]))
        df = reader.read_event_table(str(events_file))

        outcome = df["response_outcome"]
        assert pd.api.types.is_float_dtype(outcome), f"dtype is {outcome.dtype}"
        trials = df[df["event"] == "start_trial"]
        assert len(trials) == expected["start_trial"]
        assert (trials["response_outcome"] == 1).sum() == expected["correct"]
        assert (trials["response_outcome"] == 0).sum() == expected["incorrect"]
        assert trials["response_outcome"].isna().sum() == expected["blank"]
