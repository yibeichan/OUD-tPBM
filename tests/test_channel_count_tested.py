"""results.json's n_channels_tested counts channels actually tested.

Channels in the target ROIs that are skipped for having fewer than five
participants are not counted. Synthetic data; the correlation is stubbed.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_MODULES = ("roi_utils", "unified_ef_model_common", "unified_ef_model_roi",
            "unified_ef_model_channel", "utils.statistical_methods", "utils", "seeding")


@pytest.fixture(scope="module")
def channel(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("import_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_REPO / "scripts" / "brain"))
        mp.syspath_prepend(str(_REPO / "scripts" / "hypo_test"))
        yield importlib.import_module("unified_ef_model_channel")
    for name in _MODULES:
        sys.modules.pop(name, None)


def test_skipped_channels_are_not_counted_as_tested(channel, monkeypatch):
    subjects = [f"S{i}" for i in range(1, 7)]
    rows = [{"subject": s, "channel": "S1_D1", "network_roi": "Cont_PFCl", "activation": 0.1 * i}
            for i, s in enumerate(subjects)]
    # Only three participants have the second channel: skipped (< 5).
    rows += [{"subject": s, "channel": "S2_D2", "network_roi": "Cont_PFCl", "activation": 0.2}
             for s in subjects[:3]]
    monkeypatch.setattr(channel, "load_channel_data", lambda task, contrast, cache=None: pd.DataFrame(rows))
    monkeypatch.setattr(channel, "test_correlation_fast",
                        lambda x, y, rng=None, **k: {"r": 0.1, "p": 0.5, "n": len(x), "status": "completed"})
    merged = pd.DataFrame({"subject": subjects, "Inhibit_t": np.arange(6.0),
                           "interference": np.arange(6.0)[::-1]})
    hypo = {"name": "H1", "task": "stroop", "contrast": "stroop_effect",
            "target_rois": ["Cont_PFCl"], "brief_measure": "Inhibit_t",
            "behavioral_measure": "interference"}

    result = channel.test_hypothesis_channel_level("H1", hypo, {"merged": merged}, {}, seed=1)

    assert result["n_channels_tested"] == 1
    assert result["n_candidate_channels"] == 2
