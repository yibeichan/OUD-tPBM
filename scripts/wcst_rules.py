"""WCST rule changes, shared by the behavioural scoring and the GLM labelling.

A trial is a rule change only if its sorting rule (``trial_type``) differs
from the rule of the previous experimental trial. The event tables flag every
experimental block start with ``rule_change == yes``, including block starts
that keep the previous rule; such a trial needs no set shift, so it continues
the current rule period. Both stages call :func:`actual_rule_changes` so they
cannot disagree about where a rule period begins, and
:func:`learning_end_position` so they cannot disagree about where its learning
phase ends. :func:`check_trial_order` refuses a table whose timestamp order
(used by the GLM) and trial-number order (used by behaviour) differ.
"""
from typing import Optional, Sequence

import pandas as pd

LEARNING_CRITERION = 3


def actual_rule_changes(trial_types: pd.Series) -> pd.Series:
    """Return True for each experimental trial that starts a rule period.

    Parameters
    ----------
    trial_types : pd.Series
        The sorting rule of each *experimental* trial, in task order
        (control trials removed, so "previous" means the previous
        experimental trial even across an intervening control block).

    Returns
    -------
    pd.Series of bool, same index. The first trial is True: it opens the
    first rule period. Every other trial is True only where its rule differs
    from the previous trial's.

    Raises
    ------
    ValueError
        If any rule is missing. A trial with an unknown rule cannot be
        compared with its neighbour.
    """
    rules = trial_types.astype(object)
    missing = rules.isna() | rules.astype(str).str.strip().isin(["", "nan"])
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} experimental trial(s) have no sorting rule "
            "(trial_type); rule changes cannot be derived"
        )
    rules = rules.astype(str).str.strip()
    changed = rules.ne(rules.shift(1))
    return changed.astype(bool)


def learning_end_position(correct: Sequence, criterion: int = LEARNING_CRITERION) -> Optional[int]:
    """Return the position of the trial that meets the learning criterion.

    Parameters
    ----------
    correct : sequence
        Correctness of each trial in one rule period, in task order; position 0
        is the change trial that opens the period. A missing value (no
        response) counts as incorrect.
    criterion : int
        Consecutive correct trials required (3 in this study).

    Returns
    -------
    The 0-based position, within the period, of the trial completing
    ``criterion`` consecutive correct responses, or None if the period never
    reaches it. The change trial does not count toward the criterion: the
    streak starts on the trial after it, so the earliest possible position is
    ``criterion``. Trials up to and including this position are the learning
    phase; later trials are maintenance.
    """
    streak = 0
    for position, value in enumerate(correct):
        if position == 0:
            continue
        if not pd.isna(value) and bool(value):
            streak += 1
            if streak >= criterion:
                return position
        else:
            streak = 0
    return None


def check_trial_order(trials: pd.DataFrame) -> None:
    """Raise unless ordering ``trials`` by timestamp and by trial number agree.

    The GLM orders trials by ``timestamp`` and behaviour by ``trial``; rule
    changes and learning boundaries compare neighbours, so the two stages
    agree only if both keys give the same order. Checked when both columns
    are present (every study event table has both).
    """
    if not {"trial", "timestamp"} <= set(trials.columns):
        return
    by_time = trials.sort_values("timestamp", kind="stable").index
    by_trial = trials.sort_values("trial", kind="stable").index
    if not by_time.equals(by_trial):
        raise ValueError(
            "experimental trials are in a different order by timestamp than by "
            "trial number; rule periods would differ between stages"
        )
