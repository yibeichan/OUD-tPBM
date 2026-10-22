"""Per-trial WCST durations must be read, not derived and not defaulted.

Every WCST event table records a per-trial ``duration``, so the modulator
feeding ``duration_effect_correct`` reads it. The gap to the next trial's
onset is not a duration: it includes the inter-trial interval and, at a
block's final trial, the inter-block pause.

The fixtures keep a SPARSE, non-contiguous index, the shape that filtering
start_trial rows out of the full table produces, because the caller looks
durations up by POSITION. A fixture with a reset index would hide an
implementation that indexed by label.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_WCST = Path(__file__).resolve().parents[1] / "scripts" / "brain" / "step1_glm_wcst.py"


def _is_mne_family(name):
    """True for the modules the isolation fixture evicts: mne and mne_nirs.

    Deliberately narrow, matching tests/test_glm_event_loading.py: evicting
    every module a by-path load pulled in also evicts scipy, whose C
    extensions cannot be re-imported once dropped from sys.modules.

    step1_glm_wcst itself needs no entry: module_from_spec + exec_module
    never registers it in sys.modules, so each load is already fresh.
    """
    return name.split(".")[0] in {"mne", "mne_nirs"}


@pytest.fixture(scope="module", autouse=True)
def _isolate_sys_modules():
    """Evict the mne and mne_nirs modules that loading step1_glm_wcst by path adds.

    The module imports mne/mne_nirs/nilearn. Loaded via importlib into the
    shared pytest process, those would stay in sys.modules for the rest of
    the run; evicting them keeps later test modules independent of this
    one. Module-scoped, so it wraps every case here.
    """
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if _is_mne_family(name):
            sys.modules.pop(name, None)


@pytest.fixture
def wcst():
    spec = importlib.util.spec_from_file_location("step1_glm_wcst", _WCST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_durations_come_from_the_recorded_column(wcst):
    # Sparse, non-contiguous index: the shape the real start_trial filter produces.
    frame = pd.DataFrame(
        {"timestamp": [10.0, 14.0, 20.5], "duration": [1.5, 2.25, 0.75]},
        index=[3, 7, 11],
    )

    assert wcst.trial_durations_from_table(frame) == pytest.approx([1.5, 2.25, 0.75])


def test_a_missing_duration_raises_rather_than_defaulting(wcst):
    frame = pd.DataFrame(
        {"timestamp": [10.0, 14.0], "duration": [1.5, float("nan")]}, index=[3, 7]
    )

    with pytest.raises(ValueError, match="missing or non-positive"):
        wcst.trial_durations_from_table(frame)


def test_a_non_positive_duration_raises(wcst):
    frame = pd.DataFrame(
        {"timestamp": [10.0, 14.0], "duration": [1.5, 0.0]}, index=[3, 7]
    )

    with pytest.raises(ValueError, match="missing or non-positive"):
        wcst.trial_durations_from_table(frame)


def test_one_duration_per_row_in_row_order(wcst):
    """The length-and-order contract the consumer silently depends on.

    ``process_events`` attaches durations by POSITION
    (``event_df_filtered.index.get_loc(...)``). A short, long or reordered
    return would attach the wrong duration to every trial from the mismatch
    onward and raise nothing, so assert the contract directly rather than
    inferring it from the values.
    """
    n = 9
    frame = pd.DataFrame(
        {
            "timestamp": [10.0 + 3 * i for i in range(n)],
            "duration": [0.5 + 0.25 * i for i in range(n)],
        },
        # Sparse and deliberately NOT ascending: position, not label, decides.
        index=[40, 3, 17, 8, 91, 2, 55, 23, 6],
    )

    durations = wcst.trial_durations_from_table(frame)

    assert len(durations) == len(frame)
    assert durations == pytest.approx(list(frame["duration"]))


def test_an_empty_frame_yields_no_durations(wcst):
    """len(durations) == len(frame) must hold unconditionally, including at zero."""
    frame = pd.DataFrame({"timestamp": [], "duration": []})

    assert wcst.trial_durations_from_table(frame) == []


class _StubRaw:
    """Only ``info['sfreq']`` is read from raw while events are processed."""

    def __init__(self, sfreq):
        self.info = {"sfreq": sfreq}


def _events_frame():
    """A minimal WCST table: interleaved non-trial rows keep the index sparse."""
    rows = []
    for i in range(4):
        rows.append({"event": "fixation", "timestamp": 10.0 + 4 * i})
        rows.append(
            {
                "event": "start_trial",
                "timestamp": 11.0 + 4 * i,
                "duration": 1.5,
                "block_type": "experimental",
                "rule_change": "yes" if i == 0 else "no",
                "trial_type": "color",
                "response_outcome": 1,
            }
        )
    return pd.DataFrame(rows)


def test_an_unmatched_duration_raises_instead_of_substituting_the_mean(wcst):
    """The alignment step must fail loudly, not invent a plausible duration.

    Both NaN branches in ``process_events`` are unreachable today: durations
    come back one per filtered row, and every event sample is recomputed from
    that same frame. This drives one anyway, by having the duration source
    return a short list, because the branch's old behaviour -- substituting
    the mean -- is the failure mode this change exists to remove, and a
    future refactor could make it live again.
    """
    analysis = wcst.WCSTfNIRSAnalysis(task_name="wcst")
    analysis.raw = _StubRaw(sfreq=10.0)
    analysis.events_df = _events_frame()

    # One duration short: the final trial's position falls off the end.
    wcst.trial_durations_from_table = lambda event_df: [1.5] * (len(event_df) - 1)

    with pytest.raises(ValueError, match="could not be matched to a recorded duration"):
        analysis.process_events()
