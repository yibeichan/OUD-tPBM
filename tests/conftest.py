"""Shared fixtures that gate tests on the study data.

The raw data and the archived run trees are not part of the repository.
Tests that need them take one of these fixtures and are skipped, with the
variable named in the skip reason, when it is unset or points nowhere.

Expected per-participant values for the study-data checks are not part of
the repository either. They live in ``tests/local_pinned_values.json``
(untracked; ``tests/local_pinned_values.json.example`` shows its layout),
and a test that needs them is skipped when the file is absent.
"""
import json
import os
from pathlib import Path

import pytest


def _env_path(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        pytest.skip(f"{var} not set; study data not available")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{var}={value} does not exist")
    return path


@pytest.fixture(scope="session")
def study_base_dir() -> Path:
    """Parent of data/ (raw task_info, assessments, SNIRF)."""
    return _env_path("OUD_TPBM_BASE_DIR")


@pytest.fixture(scope="session")
def reference_tree() -> Path:
    """An archived run tree to check against, e.g. <archive>/main."""
    return _env_path("OUD_TPBM_REFERENCE_TREE")


PINNED_VALUES_FILE = Path(__file__).resolve().parent / "local_pinned_values.json"


@pytest.fixture(scope="session")
def pinned_values() -> dict:
    """Expected per-participant values, read from the untracked local file."""
    if not PINNED_VALUES_FILE.is_file():
        pytest.skip(f"{PINNED_VALUES_FILE.name} not present; pinned study values not available")
    return json.loads(PINNED_VALUES_FILE.read_text())


@pytest.fixture(scope="session")
def pinned(pinned_values):
    """``pinned(key)`` returns one entry of the local file; the test skips if the key is absent."""
    def get(key: str):
        if key not in pinned_values:
            pytest.skip(f"{key!r} not in {PINNED_VALUES_FILE.name}")
        return pinned_values[key]
    return get
