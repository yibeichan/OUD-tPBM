"""The WCST duration modulators use an HRF sampled at the data interval.

nilearn's ``glover_hrf(t_r, ...)`` takes the sampling interval in seconds.
step1 passed the sampling rate (3.76 Hz), which gave a 9-sample kernel whose
convolved peak sat 0.5 s after onset. With ``1 / sfreq`` the peak lands about
5 s after onset, like the condition regressors. No study data is read.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"
sys.path.insert(0, str(_BRAIN))

SFREQ = 3.7593984962406015  # the Kernel Flow rate of every session
N_TIMES = 400


def _is_mne_family(name):
    return name.split(".")[0] in {"mne", "mne_nirs"}


@pytest.fixture(scope="module", autouse=True)
def _isolate_sys_modules():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if _is_mne_family(name):
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def wcst():
    spec = importlib.util.spec_from_file_location("step1_glm_wcst", _BRAIN / "step1_glm_wcst.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRaw:
    def __init__(self):
        self.info = {"sfreq": SFREQ, "meas_date": None}
        self.n_times = N_TIMES
        self.times = np.arange(N_TIMES) / SFREQ
        self.ch_names = ["S1_D1 hbo", "S1_D1 hbr"]


class FakeShort:
    ch_names = ["s0 hbo", "s0 hbr"]

    def __init__(self):
        self._rng = np.random.default_rng(0)

    def copy(self):
        return self

    def pick(self, picks):
        return self

    def get_data(self):
        return self._rng.normal(size=(1, N_TIMES))


@pytest.mark.parametrize("hrf_model", ["glover", "spm"])
def test_kernel_peaks_near_five_seconds(wcst, hrf_model):
    kernel = wcst.duration_hrf_kernel(hrf_model, SFREQ)
    peak_s = np.argmax(kernel) / SFREQ
    assert 4.0 <= peak_s <= 6.0
    # About 32 s of kernel, not 9 samples.
    assert len(kernel) / SFREQ == pytest.approx(32.0, abs=1.0)


def test_unknown_hrf_model_raises(wcst):
    with pytest.raises(ValueError, match="not supported"):
        wcst.duration_hrf_kernel("fir", SFREQ)


def test_modulator_column_peaks_about_five_seconds_after_onset(wcst, monkeypatch):
    mapping = {11: "regular_correct", 12: "regular_incorrect", 21: "change", 31: "post_change"}
    codes = [21, 31, 11, 12, 21, 31, 11, 12]
    samples = np.arange(len(codes)) * 45 + 10  # 12 s apart, so responses do not overlap
    rng = np.random.default_rng(1)
    base = pd.DataFrame(rng.normal(size=(N_TIMES, 6)),
                        columns=["change", "post_change", "regular_correct",
                                 "regular_incorrect", "drift_1", "constant"])
    monkeypatch.setattr(wcst, "make_first_level_design_matrix", lambda *a, **k: base.copy())
    monkeypatch.setattr(wcst, "get_short_channels", lambda raw: FakeShort())

    analysis = wcst.WCSTfNIRSAnalysis("wcst")
    analysis.raw = FakeRaw()
    analysis.processed_events = np.array([[s, 0, c] for s, c in zip(samples, codes)])
    analysis.condition_mapping = mapping
    # The first regular_correct trial is long, the second short: after
    # mean-centring their modulator weights have opposite signs.
    analysis.durations = [1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 0.5, 1.0]
    analysis.create_design_matrix()

    column = analysis.design_matrix["regular_correct_duration"].to_numpy()
    first = samples[2]
    window = column[first:first + 40]  # about 10.6 s after the first regular_correct onset
    peak_s = np.argmax(window) / SFREQ
    assert 4.0 <= peak_s <= 6.0
