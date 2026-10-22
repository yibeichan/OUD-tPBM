"""step2c_session_info.py must refuse to write into any run tree that the
PROTECTED_TREE_MARKERS setting (scripts/analysis_variant.py) names.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analysis_variant import PROTECTED_TREE_MARKERS_VAR, protected_tree_markers  # noqa: E402

BRAIN = Path(__file__).resolve().parents[1] / "scripts" / "brain"
MARKERS = ("archive/frozen", "reference_run")


@pytest.fixture(scope="module")
def step2c(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("import_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(BRAIN))
        spec = importlib.util.spec_from_file_location(
            "step2c_session_info", BRAIN / "step2c_session_info.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    for name in ("step2c_session_info", "step1_glm_wcst", "channel_names"):
        sys.modules.pop(name, None)


@pytest.fixture
def markers(monkeypatch):
    monkeypatch.setenv(PROTECTED_TREE_MARKERS_VAR, " , ".join(MARKERS))
    return MARKERS


@pytest.mark.parametrize("marker", MARKERS)
def test_refuses_configured_markers(step2c, markers, marker, tmp_path):
    path = str(tmp_path / marker / "output")
    with pytest.raises(RuntimeError, match=marker):
        step2c.refuse_unsafe_output_dir(path)


def test_allows_a_safe_output_dir(step2c, markers, tmp_path):
    step2c.refuse_unsafe_output_dir(str(tmp_path / "main"))


def test_markers_are_parsed_from_a_comma_separated_list(markers):
    assert protected_tree_markers() == MARKERS


def test_nothing_is_protected_when_unset(step2c, monkeypatch, tmp_path):
    # Set but empty: a local .env cannot fill it in, since it does not override.
    monkeypatch.setenv(PROTECTED_TREE_MARKERS_VAR, "")
    assert protected_tree_markers() == ()
    step2c.refuse_unsafe_output_dir(str(tmp_path / MARKERS[0] / "output"))


def test_shell_guard_exits_nonzero_on_protected_tree(markers, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "analysis_variant.py"
    for path, code in ((tmp_path / MARKERS[1], 1), (tmp_path / "run_ok", 0)):
        result = subprocess.run(
            [sys.executable, str(script), "--check-output-dir", str(path)],
            capture_output=True, text=True,
        )
        assert result.returncode == code, result.stderr
