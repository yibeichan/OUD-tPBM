"""figure_inputs refuses a non-main run tree and names every missing input."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "figures"))
from _common import figure_inputs  # noqa: E402

from scripts.analysis_variant import PROTECTED_TREE_MARKERS_VAR, VARIANT_FILENAME  # noqa: E402

MARKERS = ("archive/frozen", "runs/reference")

MAIN = {
    "name": "main", "channels": "long", "exclude_low_accuracy": True,
    "h2_metric": "perseverative_error_rate", "exclude_invalid_brief": True,
    "second_timepoint": "most_complete",
    "mask_wcst_without_regular_incorrect": True, "brain_direction": "efficiency",
}
REQUIRED = [
    "10_longitudinal/behavior_long.csv",
    "10_longitudinal/consistency_summary.csv",
    "01_assessments/scores/phq9_scores.csv",
]


def _tree(tmp_path, payload, present=REQUIRED):
    (tmp_path / VARIANT_FILENAME).write_text(json.dumps(payload))
    for rel in present:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x\n")
    return tmp_path


def test_non_main_variant_is_refused(tmp_path):
    tree = _tree(tmp_path, dict(MAIN, name="supp_count"))
    with pytest.raises(SystemExit, match="supp_count"):
        figure_inputs(tree, REQUIRED)


def test_missing_inputs_are_all_named(tmp_path):
    tree = _tree(tmp_path, MAIN, present=REQUIRED[:1])
    with pytest.raises(SystemExit) as exc:
        figure_inputs(tree, REQUIRED)
    message = str(exc.value)
    assert REQUIRED[1] in message
    assert REQUIRED[2] in message
    assert REQUIRED[0] not in message


def test_main_tree_returns_figures_dir(tmp_path):
    tree = _tree(tmp_path, MAIN)
    assert figure_inputs(tree, REQUIRED) == tree / "figures"


def test_missing_variant_file_is_a_clean_exit(tmp_path):
    with pytest.raises(SystemExit, match=VARIANT_FILENAME):
        figure_inputs(tmp_path, REQUIRED)


@pytest.fixture
def markers(monkeypatch):
    monkeypatch.setenv(PROTECTED_TREE_MARKERS_VAR, ",".join(MARKERS))
    return MARKERS


@pytest.mark.parametrize("marker", MARKERS)
def test_protected_tree_is_refused(tmp_path, markers, marker):
    tree = tmp_path / marker / "main"
    tree.mkdir(parents=True)
    _tree(tree, MAIN)
    with pytest.raises(SystemExit, match=marker):
        figure_inputs(tree, REQUIRED)
    assert not (tree / "figures").exists()


@pytest.mark.parametrize("rel", ["output/runs/reference/main",
                                 "output/runs/reference/main/sub"])
def test_subtrees_of_a_protected_directory_are_refused(tmp_path, markers, rel):
    tree = tmp_path / rel
    tree.mkdir(parents=True)
    _tree(tree, MAIN)
    with pytest.raises(SystemExit, match="runs/reference"):
        figure_inputs(tree, REQUIRED)
    assert not (tree / "figures").exists()


def test_symlinked_scratch_tree_is_allowed(tmp_path, markers):
    tree = tmp_path / "scratch" / "check" / "tree"
    tree.mkdir(parents=True)
    _tree(tree, MAIN)
    assert figure_inputs(tree, REQUIRED) == tree / "figures"
