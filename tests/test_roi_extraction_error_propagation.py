"""Per-session errors at the ROI-extraction stage must not be swallowed.

``step3b_roi_extraction.py``'s ``main()`` must not catch an exception from
``process_single_subject``, log it, and exit 0 when a session failed. Every
exception propagates, and so does a session that yields no ROI rows. The
stage reads baseline sessions only unless --all-sessions is given.

``roi_utils``/``step3b_roi_extraction`` bind ``OUTPUT_DIR``/``BASE_DIR`` at
import, so the module is imported once against a throwaway directory and
``get_glm_with_roi_mappings``/``process_single_subject`` are monkeypatched
per test -- no real GLM or ROI-mapping files are read.

Session ids here are synthetic, not real session hashes.
"""
import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_BRAIN = _REPO / "scripts" / "brain"


@pytest.fixture(scope="module")
def step3b(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("import_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_BRAIN))
        module = importlib.import_module("step3b_roi_extraction")
        yield module
    for name in ("step3b_roi_extraction", "roi_utils"):
        sys.modules.pop(name, None)


@pytest.fixture
def out_dir(tmp_path, monkeypatch, step3b):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    monkeypatch.setattr(step3b, "OUTPUT_DIR", str(out))
    return out


def _one_session(step3b, monkeypatch):
    monkeypatch.setattr(
        step3b, "get_glm_with_roi_mappings",
        lambda task, baseline_only=False: [
            ("glm.csv", "roi_mapping.json", "aaaaaaa", "S1")
        ],
    )


def _run(step3b, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["step3b_roi_extraction.py", "--task", "stroop"])
    step3b.main()


@pytest.mark.parametrize("exc_name,message", [
    ("VariantError", "bad variant"),
    ("FileNotFoundError", "missing file"),
    ("KeyError", "'aaaaaaa'"),
])
def test_named_exceptions_propagate(step3b, out_dir, monkeypatch, exc_name, message):
    # VariantError must be resolved off the module under test: step3b_roi_extraction
    # imports it as a top-level module (via roi_utils' sys.path trick), which is a
    # distinct class object from `scripts.analysis_variant.VariantError` -- see
    # implementer-common.md's "no shared module-level state" note.
    exc_type = {
        "VariantError": step3b.VariantError,
        "FileNotFoundError": FileNotFoundError,
        "KeyError": KeyError,
    }[exc_name]
    _one_session(step3b, monkeypatch)

    def raiser(*args, **kwargs):
        raise exc_type(message)

    monkeypatch.setattr(step3b, "process_single_subject", raiser)

    with pytest.raises(exc_type):
        _run(step3b, monkeypatch)


def test_any_other_exception_propagates(step3b, out_dir, monkeypatch, caplog):
    _one_session(step3b, monkeypatch)

    def raiser(*args, **kwargs):
        raise ValueError("some other failure")

    monkeypatch.setattr(step3b, "process_single_subject", raiser)

    with caplog.at_level("ERROR", logger=step3b.logger.name):
        with pytest.raises(ValueError, match="some other failure"):
            _run(step3b, monkeypatch)

    assert any("some other failure" in r.getMessage() for r in caplog.records)


def test_a_session_with_no_roi_rows_raises(step3b, out_dir, monkeypatch):
    import pandas as pd

    _one_session(step3b, monkeypatch)
    monkeypatch.setattr(step3b, "process_single_subject",
                        lambda *a, **k: (pd.DataFrame(), {}))
    with pytest.raises(ValueError, match="No ROI results for S1"):
        _run(step3b, monkeypatch)


@pytest.mark.parametrize("flags,expected", [
    ([], True), (["--baseline-only"], True), (["--all-sessions"], False),
])
def test_baseline_only_is_the_default(step3b, out_dir, monkeypatch, flags, expected):
    seen = []

    def pairs(task, baseline_only=False):
        seen.append(baseline_only)
        return []

    monkeypatch.setattr(step3b, "get_glm_with_roi_mappings", pairs)
    monkeypatch.setattr(sys, "argv", ["step3b_roi_extraction.py", "--task", "stroop", *flags])
    with pytest.raises(SystemExit):  # no sessions
        step3b.main()
    assert seen == [expected]
