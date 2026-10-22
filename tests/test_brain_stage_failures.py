"""Brain stages 2a, 2b and 3c fail loudly instead of running on a short session set.

A failed session in step2a/2b must fail the stage, and step3c must refuse a
combined ROI file with several sessions of one subject. Every fixture is under tmp_path; session ids are synthetic.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"
sys.path.insert(0, str(BRAIN))


def _load(name, tmp_path, monkeypatch):
    # These modules exit at import without both variables; load_dotenv does
    # not override, so a local .env cannot redirect them.
    monkeypatch.setenv("BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    spec = importlib.util.spec_from_file_location(name, BRAIN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setattr(module, "OUTPUT_DIR", str(tmp_path / "out"))
    return module


class FakeMapper:
    def __init__(self, snirf_path):
        if "bbbbbbb" in snirf_path:
            raise OSError("unreadable probe")

    def get_stats(self):
        return {"n_sources": 1, "n_detectors": 1, "n_measurement_pairs": 1}

    def glm_to_snirf_labels(self, src, det):
        return f"S{src}", f"D{det}"

    def glm_to_parcellation_labels(self, src, det):
        return f"S{src}", f"D{det}"

    def create_glm_to_parcellation_lookup(self):
        return {"S1_D1": 3}


def test_step2a_exits_nonzero_when_a_session_fails(tmp_path, monkeypatch):
    step2a = _load("step2a_glm_to_parcellation", tmp_path, monkeypatch)
    stroop_dir = tmp_path / "base" / "data" / "stroop"
    stroop_dir.mkdir(parents=True)
    good = stroop_dir / "NeuroLume_S1_aaaaaaa_1.snirf"
    good.write_bytes(b"")
    (stroop_dir / "NeuroLume_S2_bbbbbbb_1.snirf").write_bytes(b"")
    monkeypatch.setattr(step2a, "find_representative_snirf", lambda: good)
    monkeypatch.setattr(step2a, "ChannelMapper", FakeMapper)
    monkeypatch.setattr(sys, "argv", ["step2a", "--per-session"])

    with pytest.raises(SystemExit) as exc:
        step2a.main()

    assert exc.value.code not in (0, None)
    # The sessions that did map are still written for inspection.
    index = json.loads((tmp_path / "out" / "04_channel_mappings" / "session_index.json").read_text())
    assert [s["session_id"] for s in index["sessions"]] == ["aaaaaaa"]


def test_step2b_raises_when_a_session_fails(tmp_path, monkeypatch):
    step2b = _load("step2b_parcellation_to_roi", tmp_path, monkeypatch)
    mapping_dir = tmp_path / "out" / "04_channel_mappings"
    mapping_dir.mkdir(parents=True)
    (mapping_dir / "mapping_stroop_S1_aaaaaaa.json").write_text("{not json")

    with pytest.raises(RuntimeError, match="mapping_stroop_S1_aaaaaaa.json"):
        step2b.process_per_session_mappings(mapping_dir, np.zeros((2, 3)), {})


def test_step3c_rejects_several_sessions_per_subject(tmp_path, monkeypatch):
    step3c = _load("step3c_roi_group_analysis", tmp_path, monkeypatch)
    path = tmp_path / "out" / "06_roi_activation" / "stroop" / "all_subjects_roi_activation_network_roi.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "subject": ["S1", "S1", "S2"],
        "session_id": ["aaaaaaa", "ccccccc", "bbbbbbb"],
        "contrast": ["stroop_effect"] * 3,
        "network_roi": ["Cont_PFCl"] * 3,
        "activation": [0.1, 0.2, 0.3],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"more than one session for \['S1'\]"):
        step3c.load_roi_activation_data("stroop")

    pd.read_csv(path).iloc[[0, 2]].to_csv(path, index=False)
    assert sorted(step3c.load_roi_activation_data("stroop")["subject"]) == ["S1", "S2"]
