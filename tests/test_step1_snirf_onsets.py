"""First-level trial onsets come from the SNIRF stims, not the event table.

The event table's ``timestamp`` runs a constant offset ahead of the SNIRF data
axis (0.8-16.8 s across the 36 sessions). These tests build a synthetic
recording whose ``StartTrial`` annotations sit a known offset before the
table's timestamps, and check that each condition regressor starts at the
SNIRF onset. No study data is read.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"
sys.path.insert(0, str(_BRAIN))

SFREQ = 4.0
N_TIMES = 800  # 200 s
OFFSET = 11.0  # event table minus SNIRF, seconds (Stroop median)


def _is_mne_family(name):
    # Same narrow eviction as tests/test_wcst_trial_durations.py.
    return name.split(".")[0] in {"mne", "mne_nirs"}


@pytest.fixture(scope="module", autouse=True)
def _isolate_sys_modules():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if _is_mne_family(name):
            sys.modules.pop(name, None)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _BRAIN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stroop():
    return _load("step1_glm_stroop")


@pytest.fixture(scope="module")
def wcst():
    return _load("step1_glm_wcst")


@pytest.fixture(scope="module")
def onsets_mod():
    return _load("trial_onsets")


def _raw_with_stims(snirf_onsets, n_times=N_TIMES, extra=()):
    """A two-channel fNIRS RawArray carrying SNIRF-style StartTrial annotations."""
    import mne

    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SFREQ, ["hbo", "hbr"])
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.normal(size=(2, n_times)) * 1e-6, info, verbose="error")
    onsets = list(snirf_onsets) + [o for o, _ in extra]
    descriptions = ["StartTrial"] * len(snirf_onsets) + [d for _, d in extra]
    raw.set_annotations(mne.Annotations(onsets, [1.0] * len(onsets), descriptions))
    return raw


# Trial onsets on the SNIRF clock; the table is OFFSET seconds later.
STROOP_SNIRF = [10.3, 20.6, 30.9, 41.2, 51.5, 61.8]
STROOP_ROWS = [("congruent", "cong_only"), ("congruent", "cong_only"),
               ("congruent", "mixed"), ("mixed", "mixed"),
               ("congruent", "mixed"), ("mixed", "mixed")]


def _stroop_table(offset=OFFSET):
    rows = [{"event": "start_block", "timestamp": STROOP_SNIRF[0] + offset - 1,
             "trial_type": np.nan, "block_type": "cong_only", "duration": 30.0}]
    for t, (trial_type, block_type) in zip(STROOP_SNIRF, STROOP_ROWS):
        rows.append({"event": "start_trial", "timestamp": t + offset,
                     "trial_type": trial_type, "block_type": block_type, "duration": 1.0})
    return pd.DataFrame(rows)


def test_stroop_events_sit_at_snirf_onsets(stroop):
    analysis = stroop.StroopfNIRSAnalysis("stroop")
    analysis.raw = _raw_with_stims(STROOP_SNIRF, extra=[(5.0, "StartExperiment")])
    analysis.events_df = _stroop_table()

    events, mapping = analysis.process_events()

    expected = [int(t * SFREQ) for t in STROOP_SNIRF]
    assert events[:, 0].tolist() == expected
    assert events[:, 2].tolist() == [1, 1, 2, 3, 2, 3]
    assert analysis.onset_offset == pytest.approx(OFFSET)


def test_stroop_regressors_start_at_snirf_onsets(stroop):
    """The design matrix's condition columns switch on at the SNIRF onsets."""
    from mne_nirs.experimental_design import make_first_level_design_matrix

    analysis = stroop.StroopfNIRSAnalysis("stroop")
    analysis.raw = _raw_with_stims(STROOP_SNIRF)
    analysis.events_df = _stroop_table()
    events, mapping = analysis.process_events()
    analysis.create_annotations(events, mapping)

    # hrf_model=None leaves the boxcars unconvolved, so each column's first
    # non-zero sample is its first trial's onset.
    design = make_first_level_design_matrix(
        analysis.raw, hrf_model=None, drift_model=None, stim_dur=1.0)
    first_on = {c: int(np.flatnonzero(design[c].to_numpy() > 0)[0])
                for c in ("congruent_pure", "congruent_mixed", "incongruent")}
    assert first_on == {
        "congruent_pure": int(STROOP_SNIRF[0] * SFREQ),
        "congruent_mixed": int(STROOP_SNIRF[2] * SFREQ),
        "incongruent": int(STROOP_SNIRF[3] * SFREQ),
    }


WCST_SNIRF = [4.1, 9.3, 14.5, 20.0, 25.2, 30.4, 35.6, 40.8, 46.0]
WCST_ROWS = [  # block_type, trial_type, rule_change, outcome, duration
    ("control", np.nan, np.nan, 1, 2.0),
    ("experimental", "color", "yes", 0, 1.5),
    ("experimental", "color", "no", 1, 1.6),
    ("experimental", "color", "no", 1, 1.7),
    ("experimental", "color", "no", 1, 1.8),
    ("experimental", "color", "no", 1, 1.9),
    ("experimental", "color", "no", 0, 2.1),
    ("experimental", "shape", "yes", 0, 2.2),
    ("experimental", "shape", "no", 1, 2.3),
]


def _wcst_table(offset=OFFSET):
    rows = []
    for t, (block, trial_type, change, outcome, dur) in zip(WCST_SNIRF, WCST_ROWS):
        rows.append({"event": "fixation", "timestamp": t + offset - 0.5})
        rows.append({"event": "start_trial", "timestamp": t + offset, "block_type": block,
                     "trial_type": trial_type if block == "experimental" else "color",
                     "rule_change": change, "response_outcome": outcome, "duration": dur})
    return pd.DataFrame(rows)


def test_wcst_events_and_durations_sit_at_snirf_onsets(wcst):
    analysis = wcst.WCSTfNIRSAnalysis("wcst")
    analysis.raw = _raw_with_stims(WCST_SNIRF)
    analysis.events_df = _wcst_table()

    events, mapping, durations = analysis.process_events()

    # The control trial is not modelled; the eight experimental ones are.
    assert events[:, 0].tolist() == [int(t * SFREQ) for t in WCST_SNIRF[1:]]
    assert [mapping[c] for c in events[:, 2]] == [
        "change", "post_change", "post_change", "post_change",
        "regular_correct", "regular_incorrect", "change", "post_change"]
    assert durations == pytest.approx([r[4] for r in WCST_ROWS[1:]])
    assert analysis.onset_offset == pytest.approx(OFFSET)


@pytest.mark.parametrize("module_name,cls_name", [
    ("stroop", "StroopfNIRSAnalysis"), ("wcst", "WCSTfNIRSAnalysis"),
])
def test_count_mismatch_raises(request, module_name, cls_name):
    module = request.getfixturevalue(module_name)
    analysis = getattr(module, cls_name)(module_name)
    snirf = STROOP_SNIRF if module_name == "stroop" else WCST_SNIRF
    analysis.raw = _raw_with_stims(snirf[:-1])
    analysis.events_df = _stroop_table() if module_name == "stroop" else _wcst_table()
    with pytest.raises(ValueError, match="StartTrial stims"):
        analysis.process_events()


def test_non_constant_offset_raises(onsets_mod):
    table = np.array([10.0, 20.0, 30.0])
    snirf = np.array([1.0, 11.0, 22.0])  # offset 9, 9, 8: one second of drift
    with pytest.raises(ValueError, match="not constant"):
        onsets_mod.match_trial_onsets(table, snirf, SFREQ)


def test_offset_within_one_sample_is_accepted(onsets_mod):
    table = np.array([10.0, 20.0, 30.0])
    snirf = table - 9.0 + np.array([0.0, 0.2, -0.04])  # spread 0.24 s < 0.25 s
    onsets, offset = onsets_mod.match_trial_onsets(table, snirf, SFREQ)
    assert onsets.tolist() == snirf.tolist()
    assert offset == pytest.approx(9.0, abs=0.25)


def test_onset_outside_recording_raises(onsets_mod):
    """On the SNIRF clock every trial must lie inside the recording."""
    table = np.array([10.0, 20.0, 230.0])
    snirf = table - 5.0  # last onset at 225 s in a 200 s recording
    with pytest.raises(ValueError, match="outside the recording"):
        onsets_mod.match_trial_onsets(table, snirf, SFREQ, n_times=N_TIMES)
