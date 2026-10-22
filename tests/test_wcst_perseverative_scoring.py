"""A rule change on a trial with no response still starts a new rule period.

The perseverative scorer must update its rule bookkeeping before it skips a
trial without a response. Otherwise, when a block's first trial is an
omission, that block and the next are scored against a stale "previous
rule". The synthetic case runs without study data; the baseline-count check
needs OUD_TPBM_BASE_DIR and the local pinned-values file.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "behav" / "01_wcst_behav.py"


@pytest.fixture(scope="module")
def behav():
    spec = importlib.util.spec_from_file_location("wcst_behav", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trial(n, rule, change, outcome, pressed, card=("blue", "star", 2.0)):
    return {
        "timestamp": 10.0 + 3 * n, "event": "start_trial", "block_type": "experimental",
        "trial": n, "trial_type": rule, "rule_change": "yes" if change else "no",
        "response_outcome": outcome, "response_pressed": pressed, "response_RT": 1.0,
        "response_correct_streak": 0, "stimulus_ambiguity_level": 1,
        "stimulus_card_color": card[0], "stimulus_card_shape": card[1],
        "stimulus_card_number": card[2],
    }


def _session():
    rows = [_trial(1, "color", True, 1, 9.0)]
    rows += [_trial(n, "color", False, 1, 9.0) for n in range(2, 6)]
    rows += [_trial(6, "shape", True, 1, 9.0)]
    rows += [_trial(n, "shape", False, 1, 9.0) for n in range(7, 10)]
    # The number block opens on an omission: no key, no outcome.
    rows += [_trial(10, "number", True, np.nan, np.nan)]
    # Key 8 is the green square card. Against a red square showing 2 it is
    # wrong under number, right under shape (the previous rule), wrong under
    # color (the rule before that).
    rows += [_trial(11, "number", False, 0, 8.0, card=("red", "square", 2.0))]
    rows += [_trial(n, "number", False, 1, 9.0) for n in range(12, 15)]
    return pd.DataFrame(rows)


def test_omitted_rule_change_trial_starts_the_new_rule_period(behav):
    metrics = behav.WCSTEventAnalyzer(_session()).get_performance_metrics()

    # The error repeats the shape rule, which was in force just before the
    # change: perseverative. Scored against color it would be non-perseverative.
    assert metrics["perseverative_errors"] == 1
    assert metrics["non_perseverative_errors"] == 0
    # The omission itself is not scored as an error.
    assert metrics["total_errors"] == 1


def test_baseline_counts_match_the_pinned_values(behav, study_base_dir, pinned):
    expected = pinned("wcst_baseline_perseverative_errors")
    task_info = pd.read_csv(study_base_dir / "data" / "wcst_task_info.csv")
    baseline = task_info[task_info["task_order"] == 1].set_index("subj_id")
    got = {}
    for subject in expected:
        task_id = baseline.loc[subject, "task_id"]
        events = pd.read_csv(
            study_base_dir / "data" / "wcst" / f"NeuroLume_{subject}_{task_id}_task_events.tsv",
            sep="\t")
        events.columns = events.columns.str.strip()
        events["event"] = events["event"].str.strip()
        got[subject] = behav.WCSTEventAnalyzer(events).get_performance_metrics()[
            "perseverative_errors"]
    assert got == expected
