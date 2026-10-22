"""Flat or non-finite channels are dropped before the GLM and recorded.

A constant channel used to be written as effect 0, se 0, p 0.5, which later
stages read as a measured zero, and its singular AR(1) fit pushed the whole
session to OLS. These tests run step1's own load/fit path on a synthetic
recording with one dead source-detector pair. No study data is read.
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
N_TIMES = 1200  # 300 s
GOOD = ["S1_D1", "S2_D2", "S3_D3"]
DEAD = "S9_D9"


def _is_mne_family(name):
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
def quality():
    return _load("channel_quality")


SNIRF_ONSETS = list(np.arange(10.0, 280.0, 7.5))
TRIALS = [("congruent", "cong_only"), ("congruent", "mixed"), ("mixed", "mixed")]


def _raw(dead=DEAD, nonfinite=None):
    import mne

    names, types = [], []
    for pair in GOOD + [dead]:
        names += [f"{pair} hbo", f"{pair} hbr"]
        types += ["hbo", "hbr"]
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(names), N_TIMES)) * 1e-6
    data[-2:] = 0.0  # the dead pair is flat in both chromophores
    if nonfinite is not None:
        data[names.index(nonfinite), 100] = np.nan
    raw = mne.io.RawArray(data, mne.create_info(names, SFREQ, types), verbose="error")
    raw.set_annotations(mne.Annotations(SNIRF_ONSETS, [1.0] * len(SNIRF_ONSETS),
                                        ["StartTrial"] * len(SNIRF_ONSETS)))
    return raw


def _stroop_table(offset=8.0):
    rows = []
    for i, t in enumerate(SNIRF_ONSETS):
        trial_type, block_type = TRIALS[i % 3]
        rows.append({"event": "start_trial", "timestamp": t + offset,
                     "trial_type": trial_type, "block_type": block_type, "duration": 1.0})
    return pd.DataFrame(rows)


class _Short:
    """Stands in for get_short_channels: two non-flat short channels."""

    ch_names = ["s0 hbo", "s0 hbr"]

    def __init__(self):
        self._rng = np.random.default_rng(5)

    def copy(self):
        return self

    def pick(self, picks):
        return self

    def get_data(self):
        return self._rng.normal(size=(1, N_TIMES)) * 1e-6


def test_unusable_channels_finds_flat_pair_and_nonfinite(quality):
    raw = _raw(nonfinite="S2_D2 hbr")
    assert quality.unusable_channels(raw) == [
        "S2_D2 hbo", "S2_D2 hbr", f"{DEAD} hbo", f"{DEAD} hbr"]


def test_stroop_session_drops_flat_pair_fits_ar1_and_records_it(stroop, monkeypatch, tmp_path):
    monkeypatch.setattr(stroop.mne.io, "read_raw_snirf", lambda *a, **k: _raw())
    monkeypatch.setattr(stroop, "rename_raw_to_canonical", lambda raw, path: 0)
    monkeypatch.setattr(stroop, "read_event_table", lambda path: _stroop_table())
    monkeypatch.setattr(stroop, "get_short_channels", lambda raw: _Short())

    analysis = stroop.StroopfNIRSAnalysis("stroop")
    analysis.load_data("unused.snirf", "unused.tsv")
    events, mapping = analysis.process_events()
    analysis.create_annotations(events, mapping)
    analysis.create_design_matrix(stim_dur=1.0)
    analysis.run_glm_analysis()

    # The dead pair is absent from every contrast: missing, not zero.
    for contrast in analysis.contrast_maps.values():
        assert set(contrast) == {f"{p} {c}" for p in GOOD for c in ("hbo", "hbr")}
    # One dead pair no longer pushes the whole session to OLS.
    assert analysis.used_noise_model == "ar1"

    record_path = tmp_path / "S1_aaaaaaa_glm_session.json"
    analysis.save_session_record("S1", "aaaaaaa", str(record_path))
    record = pd.read_json(record_path, typ="series")
    assert record["dropped_channels"] == [f"{DEAD} hbo", f"{DEAD} hbr"]
    assert record["noise_model"] == "ar1"
    assert record["n_channels_fit"] == 2 * len(GOOD)
    assert record["onset_offset_s"] == pytest.approx(8.0)


def test_wcst_load_drops_flat_pair(wcst, monkeypatch):
    monkeypatch.setattr(wcst.mne.io, "read_raw_snirf", lambda *a, **k: _raw())
    monkeypatch.setattr(wcst, "rename_raw_to_canonical", lambda raw, path: 0)
    monkeypatch.setattr(wcst, "read_event_table", lambda path: pd.DataFrame())

    analysis = wcst.WCSTfNIRSAnalysis("wcst")
    analysis.load_data("unused.snirf", "unused.tsv")

    assert analysis.dropped_channels == [f"{DEAD} hbo", f"{DEAD} hbr"]
    assert DEAD not in {name.split(" ")[0] for name in analysis.raw.ch_names}
