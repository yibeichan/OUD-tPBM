"""Per-session errors in the longitudinal all-timepoints ROI extraction must
not be swallowed.

``extract_roi_alltimepoints.py``'s ``extract()`` must not catch an
exception from ``process_single_subject``, log it, skip the session and
exit 0 when the session failed for a reason that should stop the run: a
bad/missing ``analysis_variant.json`` (``VariantError``), a missing input
file (``FileNotFoundError``), or a session absent from a lookup table
(``KeyError``). Those three propagate at once; any other failing session,
or one with no ROI rows, is logged and then fails the run before any file
is written.

Loaded by path, like ``test_longitudinal_followup_as_post.py``:
``scripts/longitudinal/`` is not a package. ``BASE_DIR``/``OUTPUT_DIR`` are
set to a throwaway ``tmp_path`` before exec, and the modules it pulls in
via ``sys.path`` tricks (``step3b_roi_extraction``, ``roi_utils``, ...) are
dropped from ``sys.modules`` first so each test starts clean.

Session ids here are synthetic, not real session hashes.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "longitudinal" / "extract_roi_alltimepoints.py"
)

_RELOAD_MODULES = (
    "step3b_roi_extraction", "roi_utils", "analysis_variant",
    "session_exclusions", "session_info", "subject_ids",
)


@pytest.fixture
def module(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    for name in _RELOAD_MODULES:
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("extract_roi_alltimepoints_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _one_session(module, monkeypatch):
    monkeypatch.setattr(
        module, "get_glm_with_roi_mappings",
        lambda task, baseline_only=False: [
            ("glm.csv", "roi_mapping.json", "aaaaaaa", "S1")
        ],
    )


@pytest.mark.parametrize("exc_name,message", [
    ("VariantError", "bad variant"),
    ("FileNotFoundError", "missing file"),
    ("KeyError", "'aaaaaaa'"),
])
def test_named_exceptions_propagate(module, monkeypatch, exc_name, message):
    # VariantError is resolved off the loaded module: extract_roi_alltimepoints
    # imports it as a top-level module (via a sys.path trick), a distinct class
    # object from `scripts.analysis_variant.VariantError` -- see
    # implementer-common.md's "no shared module-level state" note.
    exc_type = {
        "VariantError": module.VariantError,
        "FileNotFoundError": FileNotFoundError,
        "KeyError": KeyError,
    }[exc_name]
    _one_session(module, monkeypatch)

    def raiser(*args, **kwargs):
        raise exc_type(message)

    monkeypatch.setattr(module, "process_single_subject", raiser)

    with pytest.raises(exc_type):
        module.extract("stroop")


def test_other_exception_fails_the_run_after_logging(module, monkeypatch, caplog, tmp_path):
    _one_session(module, monkeypatch)

    def raiser(*args, **kwargs):
        raise ValueError("some other failure")

    monkeypatch.setattr(module, "process_single_subject", raiser)

    with caplog.at_level("ERROR", logger=module.log.name):
        with pytest.raises(RuntimeError, match="S1/aaaaaaa"):
            module.extract("stroop")

    assert any("some other failure" in r.getMessage() for r in caplog.records)
    assert not list(tmp_path.rglob("all_subjects_roi_activation_*.csv"))


def test_a_session_with_no_rows_fails_the_run(module, monkeypatch):
    import pandas as pd

    _one_session(module, monkeypatch)
    monkeypatch.setattr(module, "process_single_subject", lambda *a, **k: (pd.DataFrame(), {}))
    with pytest.raises(RuntimeError, match="S1/aaaaaaa"):
        module.extract("stroop")


def test_no_sessions_fails_the_run(module, monkeypatch):
    monkeypatch.setattr(module, "get_glm_with_roi_mappings", lambda task, baseline_only=False: [])
    with pytest.raises(RuntimeError, match="No ROI activations"):
        module.extract("stroop")
