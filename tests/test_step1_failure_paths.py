"""First-level GLM failure paths stop the session instead of reporting SUCCESS.

Each test drives one step1 method with fakes in place of the SNIRF
recording and the nilearn design matrix, so no study data is read. A WCST
session without regular-incorrect trials must keep producing its contrasts.
"""
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"
sys.path.insert(0, str(_BRAIN))

N_TIMES = 400
SFREQ = 4.0


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


class FakeRaw:
    def __init__(self, n_times=N_TIMES, sfreq=SFREQ):
        self.info = {"sfreq": sfreq, "meas_date": None}
        self.n_times = n_times
        self.times = np.arange(n_times) / sfreq
        self.ch_names = ["S1_D1 hbo", "S1_D1 hbr"]


class FakeShort:
    """What get_short_channels returns: .ch_names and .copy().pick().get_data()."""

    def __init__(self, hbo=2, hbr=2, seed=0):
        self._n = {"hbo": hbo, "hbr": hbr}
        self._rng = np.random.default_rng(seed)
        self.ch_names = [f"s{i} hbo" for i in range(hbo)] + [f"s{i} hbr" for i in range(hbr)]
        self._pick = None

    def copy(self):
        return self

    def pick(self, picks):
        self._pick = picks
        return self

    def get_data(self):
        return self._rng.normal(size=(self._n[self._pick], N_TIMES))


def _design(columns, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(N_TIMES, len(columns))), columns=columns)


STROOP_COLUMNS = ["congruent_mixed", "congruent_pure", "incongruent", "drift_1", "constant"]


def _stroop_analysis(stroop, monkeypatch, columns=STROOP_COLUMNS, short=None):
    monkeypatch.setattr(stroop, "make_first_level_design_matrix", lambda *a, **k: _design(columns))
    monkeypatch.setattr(stroop, "get_short_channels", lambda raw: short if short is not None else FakeShort())
    analysis = stroop.StroopfNIRSAnalysis("stroop")
    analysis.raw = FakeRaw()
    return analysis


# --------------------------------------------------------------------------
# Short-channel regression failure must stop the session
# --------------------------------------------------------------------------

def test_stroop_short_channel_failure_raises(stroop, monkeypatch):
    analysis = _stroop_analysis(stroop, monkeypatch)

    def boom(raw):
        raise RuntimeError("no montage")

    monkeypatch.setattr(stroop, "get_short_channels", boom)
    with pytest.raises(RuntimeError, match="no montage"):
        analysis.create_design_matrix(stim_dur=1.0)


@pytest.mark.parametrize("hbo,hbr,match", [(0, 0, "No short channels"), (2, 0, "HBR")])
def test_stroop_missing_short_regressor_raises(stroop, monkeypatch, hbo, hbr, match):
    analysis = _stroop_analysis(stroop, monkeypatch, short=FakeShort(hbo=hbo, hbr=hbr))
    with pytest.raises(ValueError, match=match):
        analysis.create_design_matrix(stim_dur=1.0)


def test_stroop_design_gets_both_short_regressors(stroop, monkeypatch):
    analysis = _stroop_analysis(stroop, monkeypatch)
    analysis.create_design_matrix(stim_dur=1.0)
    assert {"ShortHbO", "ShortHbR"} <= set(analysis.design_matrix.columns)
    assert set(analysis.contrasts) == {"stroop_effect", "mixing_cost", "incongruent_vs_pure"}


# --------------------------------------------------------------------------
# Rank is checked on the fitted matrix, and deficiency stops the session
# --------------------------------------------------------------------------

def test_stroop_rank_deficiency_from_short_regressors_raises(stroop, monkeypatch):
    analysis = _stroop_analysis(stroop, monkeypatch)
    design = _design(STROOP_COLUMNS)
    monkeypatch.setattr(stroop, "make_first_level_design_matrix", lambda *a, **k: design.copy())

    class CollinearShort(FakeShort):
        def get_data(self):
            # Mean of the picked channels equals the design's drift column.
            return np.tile(design["drift_1"].to_numpy(), (self._n[self._pick], 1))

    monkeypatch.setattr(stroop, "get_short_channels", lambda raw: CollinearShort())
    with pytest.raises(ValueError, match="rank deficient"):
        analysis.create_design_matrix(stim_dur=1.0)


# --------------------------------------------------------------------------
# A missing condition or a failed contrast must not yield SUCCESS
# --------------------------------------------------------------------------

def test_stroop_missing_condition_raises(stroop, monkeypatch):
    analysis = _stroop_analysis(
        stroop, monkeypatch, columns=["congruent_mixed", "incongruent", "drift_1", "constant"])
    with pytest.raises(ValueError, match="congruent_pure"):
        analysis.create_design_matrix(stim_dur=1.0)


class FailingEstimates:
    def compute_contrast(self, vector, contrast_type="t"):
        raise RuntimeError("singular contrast")


@pytest.mark.parametrize("module_name,cls_name", [
    ("stroop", "StroopfNIRSAnalysis"), ("wcst", "WCSTfNIRSAnalysis"),
])
def test_failed_contrast_raises(request, monkeypatch, module_name, cls_name):
    module = request.getfixturevalue(module_name)
    monkeypatch.setattr(module, "run_glm", lambda raw, dm, noise_model: FailingEstimates())
    analysis = getattr(module, cls_name)(module_name)
    analysis.raw = FakeRaw()
    analysis.design_matrix = _design(["a", "b"])
    analysis.contrasts = {"c": np.array([1.0, -1.0])}
    with pytest.raises(RuntimeError, match="singular contrast"):
        analysis.run_glm_analysis()


@pytest.mark.parametrize("module_name,cls_name", [
    ("stroop", "StroopfNIRSAnalysis"), ("wcst", "WCSTfNIRSAnalysis"),
])
def test_save_error_is_not_swallowed(request, tmp_path, module_name, cls_name):
    module = request.getfixturevalue(module_name)
    analysis = getattr(module, cls_name)(module_name)

    class Broken:
        def items(self):
            raise AttributeError("contrast_maps has no items")

    analysis.contrast_maps = Broken()
    with pytest.raises(AttributeError):
        analysis.save_glm_results(str(tmp_path / "S1_aaaaaaa_glm.pkl"))


@pytest.mark.parametrize("module_name,cls_name", [
    ("stroop", "StroopfNIRSAnalysis"), ("wcst", "WCSTfNIRSAnalysis"),
])
def test_saving_no_contrasts_raises(request, tmp_path, module_name, cls_name):
    module = request.getfixturevalue(module_name)
    analysis = getattr(module, cls_name)(module_name)
    analysis.contrast_maps = {}
    with pytest.raises(ValueError, match="No contrast"):
        analysis.save_glm_results(str(tmp_path / "S1_aaaaaaa_glm.pkl"))


# --------------------------------------------------------------------------
# WCST design: missing columns and degenerate designs
# --------------------------------------------------------------------------

WCST_MAPPING = {11: "regular_correct", 12: "regular_incorrect", 21: "change", 31: "post_change"}


def _wcst_analysis(wcst, monkeypatch, event_codes, short=None):
    base_columns = sorted({WCST_MAPPING[c] for c in event_codes}) + ["drift_1", "constant"]
    monkeypatch.setattr(wcst, "make_first_level_design_matrix", lambda *a, **k: _design(base_columns))
    monkeypatch.setattr(wcst, "get_short_channels", lambda raw: short if short is not None else FakeShort())
    analysis = wcst.WCSTfNIRSAnalysis("wcst")
    analysis.raw = FakeRaw()
    samples = np.arange(len(event_codes)) * 20 + 10
    analysis.processed_events = np.array([[s, 0, c] for s, c in zip(samples, event_codes)])
    analysis.condition_mapping = WCST_MAPPING
    rng = np.random.default_rng(3)
    analysis.durations = list(1.0 + rng.uniform(size=len(event_codes)))
    return analysis


ALL_CODES = [21, 31, 31, 11, 11, 12, 21, 31, 11, 12, 11, 11]


def test_wcst_full_design_builds_four_contrasts(wcst, monkeypatch):
    analysis = _wcst_analysis(wcst, monkeypatch, ALL_CODES)
    analysis.create_design_matrix()
    assert set(analysis.contrasts) == {
        "correct_vs_incorrect", "change_vs_regular", "learning", "duration_effect_correct"}
    assert {"ShortHbO", "ShortHbR"} <= set(analysis.design_matrix.columns)


def test_wcst_without_regular_incorrect_still_builds_and_warns(wcst, monkeypatch, caplog):
    # A session with no regular_incorrect trial: the contrasts are still
    # written (and masked downstream), with a warning.
    codes = [c for c in ALL_CODES if c != 12]
    analysis = _wcst_analysis(wcst, monkeypatch, codes)
    with caplog.at_level(logging.WARNING, logger=wcst.logger.name):
        analysis.create_design_matrix()
    assert set(analysis.contrasts) == {
        "correct_vs_incorrect", "change_vs_regular", "learning", "duration_effect_correct"}
    assert any("No regular_incorrect trials" in r.getMessage() for r in caplog.records)


def test_wcst_missing_other_condition_raises(wcst, monkeypatch):
    codes = [c for c in ALL_CODES if c != 11]  # no regular_correct
    analysis = _wcst_analysis(wcst, monkeypatch, codes)
    with pytest.raises(ValueError, match="regular_correct"):
        analysis.create_design_matrix()


def test_wcst_short_channel_failure_raises(wcst, monkeypatch):
    analysis = _wcst_analysis(wcst, monkeypatch, ALL_CODES)

    def boom(raw):
        raise RuntimeError("no montage")

    monkeypatch.setattr(wcst, "get_short_channels", boom)
    with pytest.raises(RuntimeError, match="no montage"):
        analysis.create_design_matrix()


def test_wcst_missing_short_hbo_raises(wcst, monkeypatch):
    analysis = _wcst_analysis(wcst, monkeypatch, ALL_CODES, short=FakeShort(hbo=0, hbr=2))
    with pytest.raises(ValueError, match="HBO"):
        analysis.create_design_matrix()


def test_wcst_rank_deficiency_raises(wcst, monkeypatch):
    analysis = _wcst_analysis(wcst, monkeypatch, ALL_CODES)
    base = _design(["change", "post_change", "regular_correct", "regular_incorrect",
                    "drift_1", "constant"])
    base["constant"] = base["drift_1"]  # duplicate column
    monkeypatch.setattr(wcst, "make_first_level_design_matrix", lambda *a, **k: base.copy())
    with pytest.raises(ValueError, match="rank deficient"):
        analysis.create_design_matrix()


# --------------------------------------------------------------------------
# A WCST trial past the end of the recording stops the session
# --------------------------------------------------------------------------
# Onsets come from the SNIRF StartTrial stims, and the export is
# Cropped to the experiment with about 1 s to spare at each end, so every
# Trial lies inside the recording. One that does not means the stims and the
# Data disagree; it raises rather than being dropped with a warning.

def test_wcst_event_past_end_raises(wcst):
    import mne

    analysis = wcst.WCSTfNIRSAnalysis("wcst")
    analysis.raw = FakeRaw(n_times=40, sfreq=1.0)  # 40 s recording
    snirf_onsets = [2.0, 6.0, 10.0, 14.0, 18.0, 45.0]  # the last starts after the recording
    analysis.raw.annotations = mne.Annotations(
        snirf_onsets, [1.5] * len(snirf_onsets), ["StartTrial"] * len(snirf_onsets))
    rows = []
    for ts, outcome, change in zip(snirf_onsets, [1, 1, 1, 1, 0, 1],
                                   ["yes", "no", "no", "no", "no", "no"]):
        rows.append({"event": "start_trial", "timestamp": ts + 3.0, "block_type": "experimental",
                     "trial_type": "color", "rule_change": change, "response_outcome": outcome,
                     "duration": 1.5})
    analysis.events_df = pd.DataFrame(rows, index=range(0, 2 * len(rows), 2))

    with pytest.raises(ValueError, match="outside the recording"):
        analysis.process_events()
