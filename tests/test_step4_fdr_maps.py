"""The FDR topomaps paint only what survives FDR; ROI maps say what they paint.

An "FDR-corrected" map must not be the uncorrected map under another title.
No study data is read.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "brain" / "step4_glm_topoplots.py"


@pytest.fixture(scope="module", autouse=True)
def _isolate_sys_modules():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.split(".")[0] in {"mne", "mne_nirs"}:
            sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def step4(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("step4")
    mp = pytest.MonkeyPatch()
    mp.setenv("BASE_DIR", str(tmp))
    mp.setenv("OUTPUT_DIR", str(tmp))
    spec = importlib.util.spec_from_file_location("step4_glm_topoplots", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mp.undo()
    return module


STATS = pd.DataFrame({
    "channel": ["S1_D1", "S2_D2", "S3_D3", "S4_D4"],
    "t_statistic": [3.0, -5.0, 1.0, np.nan],
    "significant_fdr": [True, False, False, False],
})


def test_uncorrected_map_paints_every_value(step4):
    values = step4.display_values(STATS, "t_statistic", fdr_corrected=False)
    np.testing.assert_array_equal(values[:3], [3.0, -5.0, 1.0])
    assert np.isnan(values[3])


def test_fdr_map_paints_only_survivors(step4):
    values = step4.display_values(STATS, "t_statistic", fdr_corrected=True)
    np.testing.assert_array_equal(values[:3], [3.0, 0.0, 0.0])
    assert np.isnan(values[3])  # missing stays missing, not painted as 0


def test_fdr_map_with_no_survivors_is_blank(step4):
    stats = STATS.assign(significant_fdr=False)
    values = step4.display_values(stats, "t_statistic", fdr_corrected=True)
    assert (values[:3] == 0).all()


def test_fdr_map_needs_the_significance_column(step4):
    with pytest.raises(KeyError, match="significant_fdr"):
        step4.display_values(STATS.drop(columns="significant_fdr"), "t_statistic", True)


def test_roi_map_labels_its_value_as_descriptive(step4):
    source = _SCRIPT.read_text()
    assert 'cbar.set_label("mean first-level t (descriptive)"' in source


@pytest.mark.parametrize("fdr", [False, True])
def test_colour_limit_comes_from_painted_channels_only(step4, monkeypatch, tmp_path, fdr):
    """Review m1: a row with no channel on the montage must not set the scale.

    Before B11 the limit was max |t| over the painted channels; B11 took it
    over every stats row, so an unpainted row could move the uncorrected
    map's scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    captured = {}

    def fake_topomap(values, info, axes=None, vlim=None, **kwargs):
        captured["vlim"] = vlim
        return cm.ScalarMappable(), None

    monkeypatch.setattr(step4.mne, "pick_info", lambda info, idx: info)
    monkeypatch.setattr(step4.mne, "EvokedArray", lambda data, info, tmin=0: type("E", (), {"info": info})())
    monkeypatch.setattr(step4.mne.viz, "plot_topomap", fake_topomap)

    channels = [f"S{i}_D{i}" for i in range(12)]
    stats = pd.DataFrame({
        "channel": channels + ["S99_D99"],              # last one is not on the montage
        "t_statistic": [1.0 + 0.1 * i for i in range(12)] + [9.0],
        "significant_fdr": [True] + [False] * 12,
        "n_subjects": 11,
    })
    ch_mapping = {ch: i for i, ch in enumerate(channels)}
    info = {"ch_names": channels}

    step4.create_channel_topoplot(stats, info, ch_mapping, "stroop_effect", tmp_path,
                                  fdr_corrected=fdr)

    assert captured["vlim"] == pytest.approx((-2.1, 2.1))
