"""A WCST rule change requires a different sorting rule.

The event tables flag every experimental block start with
``rule_change == yes``, and some block starts keep the sorting rule of the
previous experimental trial. Such a trial needs no set shift. Both stages
now derive rule changes from ``trial_type`` through one shared function:

* the GLM labels a same-rule block start as a continuation of the current
  rule period (post_change or regular), not as ``change``;
* behaviour does not open a new rule period there, so a perseverative error
  is scored against the last *different* rule (Heaton's previously correct
  principle).

No study data is read.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_BRAIN = _REPO / "scripts" / "brain"
_BEHAV = _REPO / "scripts" / "behav" / "01_wcst_behav.py"
sys.path.insert(0, str(_BRAIN))

SFREQ = 4.0
N_TIMES = 800
OFFSET = 5.0


def _is_mne_family(name):
    # Same narrow eviction as tests/test_wcst_trial_durations.py.
    return name.split(".")[0] in {"mne", "mne_nirs"}


@pytest.fixture(scope="module", autouse=True)
def _isolate_sys_modules():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if _is_mne_family(name):
            sys.modules.pop(name, None)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wcst():
    return _load(_BRAIN / "step1_glm_wcst.py", "step1_glm_wcst")


@pytest.fixture(scope="module")
def behav():
    return _load(_BEHAV, "wcst_behav")


# --------------------------------------------------------------------------
# Shared definition
# --------------------------------------------------------------------------

def test_shared_definition_marks_only_actual_switches():
    from wcst_rules import actual_rule_changes

    rules = pd.Series(["color", "color", "color", "shape", "shape", "color"],
                      index=[4, 9, 11, 20, 21, 30])
    got = actual_rule_changes(rules)
    # The first trial opens the first rule period; a repeated rule is never a change.
    assert got.tolist() == [True, False, False, True, False, True]
    assert got.index.tolist() == rules.index.tolist()


def test_shared_definition_refuses_a_missing_rule():
    from wcst_rules import actual_rule_changes

    with pytest.raises(ValueError, match="sorting rule"):
        actual_rule_changes(pd.Series(["color", np.nan, "shape"]))


# --------------------------------------------------------------------------
# GLM labelling
# --------------------------------------------------------------------------

def _raw_with_stims(snirf_onsets):
    import mne

    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SFREQ, ["hbo", "hbr"])
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.normal(size=(2, N_TIMES)) * 1e-6, info, verbose="error")
    raw.set_annotations(mne.Annotations(list(snirf_onsets), [1.0] * len(snirf_onsets),
                                        ["StartTrial"] * len(snirf_onsets)))
    return raw


def _glm_table(rows):
    """rows: (block_type, trial_type, rule_change, outcome); onsets 5 s apart."""
    snirf = [4.0 + 5.0 * i for i in range(len(rows))]
    table = []
    for t, (block, rule, change, outcome) in zip(snirf, rows):
        table.append({"event": "start_trial", "timestamp": t + OFFSET, "block_type": block,
                      "trial_type": rule, "rule_change": change,
                      "response_outcome": outcome, "duration": 1.5})
    return snirf, pd.DataFrame(table)


def _labels(wcst, rows):
    snirf, table = _glm_table(rows)
    analysis = wcst.WCSTfNIRSAnalysis("wcst")
    analysis.raw = _raw_with_stims(snirf)
    analysis.events_df = table
    events, mapping, _ = analysis.process_events()
    return [mapping[c] for c in events[:, 2]]


def test_glm_same_rule_block_start_continues_the_learned_period(wcst):
    rows = [
        ("experimental", "color", "yes", 1),
        ("experimental", "color", "no", 1),
        ("experimental", "color", "no", 1),
        ("experimental", "color", "no", 1),   # criterion (change trial not counted)
        ("experimental", "color", "no", 0),
        ("control", "color", np.nan, 1),
        ("experimental", "color", "yes", 1),  # block start, same rule
        ("experimental", "color", "no", 0),
        ("experimental", "shape", "yes", 0),  # actual switch
        ("experimental", "shape", "no", 1),
    ]
    assert _labels(wcst, rows) == [
        "change", "post_change", "post_change", "post_change", "regular_incorrect",
        # Previously: change, post_change. The rule is already learned.
        "regular_correct", "regular_incorrect",
        "change", "post_change"]


def test_glm_same_rule_block_start_keeps_the_learning_streak(wcst):
    # The criterion has not been met when the block ends; the streak carries
    # over the same-rule block start instead of restarting after a new change.
    rows = [
        ("experimental", "color", "yes", 0),
        ("experimental", "color", "no", 1),
        ("experimental", "color", "no", 1),
        ("experimental", "color", "yes", 1),  # block start, same rule: 3rd correct
        ("experimental", "color", "no", 1),
        ("experimental", "shape", "yes", 1),
    ]
    assert _labels(wcst, rows) == [
        "change", "post_change", "post_change", "post_change", "regular_correct", "change"]


def test_glm_omission_counts_as_incorrect(wcst):
    # Kept by decision: a trial with no response is incorrect. In the
    # learning phase it breaks the streak; after the criterion it is
    # regular_incorrect.
    rows = [
        ("experimental", "color", "yes", 1),
        ("experimental", "color", "no", 1),
        ("experimental", "color", "no", np.nan),
        ("experimental", "color", "no", 1),
        ("experimental", "color", "no", 1),
        ("experimental", "color", "no", 1),
        ("experimental", "color", "no", np.nan),
    ]
    assert _labels(wcst, rows) == [
        "change", "post_change", "post_change", "post_change", "post_change",
        "post_change", "regular_incorrect"]


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------

def _trial(n, rule, change, outcome, pressed=9.0, card=("blue", "star", 2.0),
           block="experimental"):
    return {
        "timestamp": 10.0 + 3 * n, "event": "start_trial", "block_type": block,
        "trial": n, "trial_type": rule, "rule_change": "yes" if change else "no",
        "response_outcome": outcome, "response_pressed": pressed, "response_RT": 1.0,
        "response_correct_streak": 0, "stimulus_ambiguity_level": 1,
        "stimulus_card_color": card[0], "stimulus_card_shape": card[1],
        "stimulus_card_number": card[2],
    }


def _session_with_same_rule_block_start():
    rows = [_trial(1, "color", True, 1)]
    rows += [_trial(n, "color", False, 1) for n in range(2, 6)]
    rows += [_trial(6, "shape", True, 1)]
    rows += [_trial(n, "shape", False, 1) for n in range(7, 11)]
    rows += [_trial(11, "color", False, 1, block="control")]
    # The next block starts flagged as a change but keeps the shape rule.
    rows += [_trial(12, "shape", True, 1)]
    # Key 7 is the red circle card. Against a red square showing 2 it is
    # right under color (the last different rule), wrong under shape.
    rows += [_trial(13, "shape", False, 0, pressed=7.0, card=("red", "square", 2.0))]
    rows += [_trial(n, "shape", False, 1) for n in range(14, 17)]
    rows += [_trial(17, "number", True, 1)]
    rows += [_trial(n, "number", False, 1) for n in range(18, 22)]
    return pd.DataFrame(rows)


def test_behaviour_scores_perseveration_against_the_last_different_rule(behav):
    metrics = behav.WCSTEventAnalyzer(
        _session_with_same_rule_block_start()).get_performance_metrics()
    # Previously the shape block start made shape the "previous rule", so
    # this error could not be perseverative.
    assert metrics["perseverative_errors"] == 1
    assert metrics["non_perseverative_errors"] == 0
    assert metrics["total_errors"] == 1


def test_behaviour_rule_periods_follow_actual_switches(behav):
    metrics = behav.WCSTEventAnalyzer(
        _session_with_same_rule_block_start()).get_performance_metrics()
    # color, shape, number: the same-rule block start does not open a period.
    assert metrics["total_rule_changes"] == 3
    assert metrics["rule_periods"] == 3
    assert metrics["categories_completed"] == 3
    assert metrics["prop_rule_periods_mastered"] == pytest.approx(1.0)


def test_behaviour_trial_table_flags_only_actual_switches(behav):
    analyzer = behav.WCSTEventAnalyzer(_session_with_same_rule_block_start())
    trials = analyzer.process_trials()
    flagged = trials.loc[trials["rule_change"] == 1, "trial_number"].tolist()
    assert flagged == [1, 6, 17]


# --------------------------------------------------------------------------
# Learning criterion: the change trial does
# not count toward the 3-consecutive-correct criterion, in either stage.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("correct,expected", [
    ([1, 1, 1, 1], 3),        # the correct change trial is not counted
    ([0, 1, 1, 1], 3),
    ([1, 1, 0, 1, 1, 1], 5),
    ([1, 1, 1], None),        # only two countable trials
    ([1, 1, 1, 0, 1, 1], None),
    ([], None),
])
def test_shared_learning_end_excludes_the_change_trial(correct, expected):
    from wcst_rules import learning_end_position

    assert learning_end_position([bool(c) for c in correct]) == expected


def test_shared_learning_end_treats_missing_outcome_as_incorrect():
    from wcst_rules import learning_end_position

    # No response breaks the streak.
    assert learning_end_position([True, True, np.nan, True, True]) is None
    assert learning_end_position([True, True, np.nan, True, True, True]) == 5


def _criterion_session():
    periods = [
        ("color", [1, 1, 1, 0, 1, 1]),              # mastered only if the change trial counts
        ("shape", [0, 1, 1, 1, 1, 0, 1]),
        ("number", [1, 1, 1, 1, 1, 1, 0, 1, 0]),
        ("color", [1, 0, 1, 1, 1, 1]),
    ]
    rows, n = [], 1
    for rule, outcomes in periods:
        for i, outcome in enumerate(outcomes):
            rows.append(_trial(n, rule, i == 0, outcome))
            n += 1
    return pd.DataFrame(rows)


def test_behaviour_criterion_excludes_the_change_trial(behav):
    metrics = behav.WCSTEventAnalyzer(_criterion_session()).get_performance_metrics()
    # Periods 2-4 reach the criterion; period 1 does not.
    assert metrics["rule_periods"] == 4
    assert metrics["categories_completed"] == 3
    assert metrics["prop_rule_periods_mastered"] == pytest.approx(0.75)
    # Trials from the period start through the criterion trial: 4, 4, 5.
    assert metrics["mean_trials_to_learn_rule"] == pytest.approx(13 / 3)
    assert metrics["trials_to_first_category"] == 4
    # Maintenance after the criterion: shape [1,0,1]; number [1,1,0 | 1,0];
    # color [1]. Early 5/7, late 1/2.
    assert metrics["early_rule_accuracy"] == pytest.approx(5 / 7)
    assert metrics["late_rule_accuracy"] == pytest.approx(1 / 2)
    assert metrics["wm_maintenance_decline"] == pytest.approx(5 / 7 - 1 / 2)
    assert metrics["wm_valid_periods"] == 3


# --------------------------------------------------------------------------
# Trial order: the GLM orders trials by timestamp and
# behaviour by trial number; both refuse a table where the two disagree.
# --------------------------------------------------------------------------

def test_shared_order_check_refuses_disagreeing_orders():
    from wcst_rules import check_trial_order

    ok = pd.DataFrame({"trial": [1, 2, 3], "timestamp": [1.0, 2.0, 3.0]}, index=[7, 3, 9])
    check_trial_order(ok)
    bad = pd.DataFrame({"trial": [1, 3, 2], "timestamp": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="order"):
        check_trial_order(bad)


def test_behaviour_refuses_disagreeing_orders(behav):
    events = _criterion_session()
    events.loc[3, "timestamp"], events.loc[4, "timestamp"] = \
        events.loc[4, "timestamp"], events.loc[3, "timestamp"]
    with pytest.raises(ValueError, match="order"):
        behav.WCSTEventAnalyzer(events).get_performance_metrics()


def test_glm_refuses_disagreeing_orders(wcst):
    rows = [("experimental", "color", "yes", 1)] + \
        [("experimental", "color", "no", 1)] * 4
    snirf, table = _glm_table(rows)
    table["trial"] = [1, 2, 4, 3, 5]
    analysis = wcst.WCSTfNIRSAnalysis("wcst")
    analysis.raw = _raw_with_stims(snirf)
    analysis.events_df = table
    with pytest.raises(ValueError, match="order"):
        analysis.process_events()
