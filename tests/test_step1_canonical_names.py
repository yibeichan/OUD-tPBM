"""step1 must rename channels to canonical names right after loading."""
import importlib.util
import sys
from pathlib import Path

import pytest

BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"
sys.path.insert(0, str(BRAIN))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BRAIN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("module_name,cls_name", [
    ("step1_glm_stroop", "StroopfNIRSAnalysis"),
    ("step1_glm_wcst", "WCSTfNIRSAnalysis"),
])
def test_load_data_renames_to_canonical(monkeypatch, tmp_path, module_name, cls_name):
    mod = _load(module_name)

    class FakeRaw:
        ch_names = ["S31_D61 hbo", "S31_D61 hbr"]
        def load_data(self):
            return self

        def get_data(self):
            # Two signal-carrying channels: load_data's flat-channel screen
            # finds nothing to drop.
            return [[0.0, 1.0], [1.0, 0.0]]

    calls = []
    monkeypatch.setattr(mod.mne.io, "read_raw_snirf", lambda *a, **k: FakeRaw())
    monkeypatch.setattr(mod, "read_event_table", lambda path: None)
    monkeypatch.setattr(mod, "rename_raw_to_canonical",
                        lambda raw, path: calls.append(path) or 2)
    analysis = getattr(mod, cls_name)("task")
    analysis.load_data("rec.snirf", "events.tsv")
    assert calls == ["rec.snirf"]
