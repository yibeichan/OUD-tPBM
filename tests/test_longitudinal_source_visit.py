"""Written longitudinal files name the real visit behind "post".

The second timepoint is chosen per participant and relabelled order 2
("post"); when it is a follow-up visit, the written files must say so.
Synthetic frames only; subject codes from S90 up are outside the cohort's
range.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "longitudinal" / "longitudinal_exploratory.py")


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("longitudinal_exploratory", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T2 = pd.DataFrame({"subject_id": ["S91", "S93"], "t2_order": [2, 3],
                   "n_levels": [3, 3], "levels": ["brief+stroop+wcst"] * 2})


def test_follow_up_rows_are_named_as_follow_up(module):
    df = pd.DataFrame({"subject_id": ["S91", "S91", "S93", "S93"], "task_order": [1, 2, 1, 2],
                       "timepoint": ["baseline", "post", "baseline", "post"],
                       "metric": ["m"] * 4, "value": [1.0, 2.0, 3.0, 4.0]})
    out = module.with_source_visit(df, T2)
    assert out["source_task_order"].tolist() == [1, 2, 1, 3]
    assert out["source_timepoint"].tolist() == ["baseline", "post", "baseline", "followup"]
    # The analysis labels are untouched.
    assert out["timepoint"].tolist() == df["timepoint"].tolist()


def test_unselected_second_timepoint_raises(module):
    df = pd.DataFrame({"subject_id": ["S9"], "task_order": [2], "timepoint": ["post"]})
    with pytest.raises(ValueError, match="no selection"):
        module.with_source_visit(df, T2)


def test_existing_source_visit_is_kept(module):
    # A questionnaire taken from the post visit although the participant's
    # tasks use the follow-up keeps the source visit it was given.
    df = pd.DataFrame({"subject_id": ["S93", "S93"], "task_order": [1, 2],
                       "timepoint": ["baseline", "post"], "source_task_order": [1, 2]})
    out = module.with_source_visit(df, T2)
    assert out["source_task_order"].tolist() == [1, 2]
    assert out["source_timepoint"].tolist() == ["baseline", "post"]
    assert list(out.columns)[-2:] == ["source_task_order", "source_timepoint"]
