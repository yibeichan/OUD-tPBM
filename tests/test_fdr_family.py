"""The confirmatory FDR family is the eight H1-H4 primary tests.

The H5 clause keyed on a '_direct' suffix that the code overwrites before
storing, so it never fired and the family has always been 8 -- which is what
the manuscript documents. Repairing the clause would move it to 9 and change
every published p_FDR, so the dead branch is removed and the size pinned.

``unified_ef_model_roi`` calls ``load_dotenv()`` at import and binds
``OUTPUT_DIR`` at module level, and ``utils.statistical_methods`` exits the
process when ``BASE_DIR``/``OUTPUT_DIR`` are unset -- and ``load_dotenv()``
walks up to the repo's own ``.env`` if they are, silently binding the import
to whatever OUTPUT_DIR is configured there. So BASE_DIR/OUTPUT_DIR are pinned
to a throwaway tmp path before the import, same as tests/test_seeded_inference.py.
"""
import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HYPO_DIR = _REPO / "scripts" / "hypo_test"


@pytest.fixture(scope="module")
def unified_ef_model_roi(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("output_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BASE_DIR", str(tmp))
        mp.setenv("OUTPUT_DIR", str(tmp))
        mp.syspath_prepend(str(_HYPO_DIR))
        yield importlib.import_module("unified_ef_model_roi")
    for name in ("unified_ef_model_roi", "unified_ef_model_common",
                 "utils.statistical_methods", "utils", "seeding"):
        sys.modules.pop(name, None)


def test_the_eight_primary_tests_are_admitted(unified_ef_model_roi):
    is_primary_test = unified_ef_model_roi.is_primary_test
    admitted = [
        (h, f"{h}_{kind}")
        for h in ("H1", "H2", "H3", "H4")
        for kind in ("self_brain", "brain_behav")
    ]
    assert all(is_primary_test(h, name) for h, name in admitted)
    assert len(admitted) == unified_ef_model_roi.PRIMARY_FAMILY_SIZE


def test_h5_is_not_in_the_family_under_either_name(unified_ef_model_roi):
    is_primary_test = unified_ef_model_roi.is_primary_test
    assert not is_primary_test("H5", "H5_global_performance_z")
    # The name the dead clause keyed on. Still excluded, so a rename cannot
    # silently grow the family to nine.
    assert not is_primary_test("H5", "H5_direct")


def test_self_report_behavioral_tests_are_excluded(unified_ef_model_roi):
    assert not unified_ef_model_roi.is_primary_test("H1", "H1_self_behav")
