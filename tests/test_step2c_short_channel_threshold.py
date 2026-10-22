"""step2c's short-channel cutoff must equal the one step1 applies via MNE-NIRS."""
import importlib.util
import inspect
import sys
from pathlib import Path

from mne_nirs.channels import get_short_channels

BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"


def test_short_channel_threshold_matches_mne_nirs_default(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(BRAIN))
    spec = importlib.util.spec_from_file_location("step2c_thr", BRAIN / "step2c_session_info.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("step1_glm_wcst", "channel_names"):
        sys.modules.pop(name, None)
    default = inspect.signature(get_short_channels).parameters["max_dist"].default
    assert module.SHORT_CHANNEL_MAX_DIST_M == default
