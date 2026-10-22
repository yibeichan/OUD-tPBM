"""Pick each longitudinal participant's second timepoint.

A participant's second timepoint is post (task_order 2) or follow-up
(task_order 3), whichever has more usable measurement levels. The levels
are the BRIEF-A self-report, the Stroop session and the WCST session. On a
tie post wins. A participant with no usable level at either candidate is
left out.

The rule only counts what it is given: the caller removes excluded task
sessions and failed or empty BRIEF-A administrations before building the
order sets.
"""
from __future__ import annotations

from typing import Dict, Set

import pandas as pd
from natsort import natsorted

CANDIDATE_ORDERS = (2, 3)
T2_LABELS = {1: "baseline", 2: "post"}
SELECTION_COLUMNS = ["subject_id", "t2_order", "n_levels", "levels"]


def orders_by_subject(df: pd.DataFrame, order_col: str = "task_order") -> Dict[str, Set[int]]:
    """Return {subject_id: set of orders present in ``df``}."""
    if df.empty:
        return {}
    return {
        subject: {int(o) for o in orders}
        for subject, orders in df.groupby("subject_id")[order_col]
    }


def select_second_timepoints(
    task_orders: Dict[str, Dict[str, Set[int]]],
    brief_orders: Dict[str, Set[int]],
) -> pd.DataFrame:
    """Return one row per participant with a usable second timepoint.

    ``task_orders[task][subject]`` holds the task_orders with a usable
    session; ``brief_orders[subject]`` the task_orders with a usable BRIEF-A.
    """
    sources = {"brief": brief_orders, **task_orders}
    subjects = set(brief_orders)
    for per_subject in task_orders.values():
        subjects |= set(per_subject)

    rows = []
    for subject in natsorted(subjects):
        available = {level: set(orders.get(subject, set())) for level, orders in sources.items()}
        if not any(o > 1 for orders in available.values() for o in orders):
            continue
        best = None
        for candidate in CANDIDATE_ORDERS:
            levels = sorted(level for level, orders in available.items() if candidate in orders)
            # strict > keeps the earlier candidate (post) on a tie
            if best is None or len(levels) > len(best[1]):
                best = (candidate, levels)
        t2_order, levels = best
        if not levels:
            continue
        rows.append({
            "subject_id": subject,
            "t2_order": t2_order,
            "n_levels": len(levels),
            "levels": "+".join(levels),
        })
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)


def choose_second_timepoints(
    task_infos: Dict[str, pd.DataFrame], brief: pd.DataFrame
) -> pd.DataFrame:
    """Run ``select_second_timepoints`` on per-task session tables and BRIEF-A rows.

    ``task_infos`` maps each task to its sessions (subject_id, task_order)
    after exclusions; ``brief`` holds the usable BRIEF-A rows, one or more
    per administration (subject_id, task_order).
    """
    return select_second_timepoints(
        {task: orders_by_subject(info) for task, info in task_infos.items()},
        orders_by_subject(brief),
    )


def apply_second_timepoint(
    df: pd.DataFrame, t2: pd.DataFrame, order_col: str = "task_order"
) -> pd.DataFrame:
    """Keep baseline and the chosen second timepoint, relabelled as order 2.

    Rows of participants absent from ``t2`` are dropped, as are rows at any
    order other than 1 or the participant's ``t2_order``. ``timepoint`` is
    set from the new order.
    """
    chosen = dict(zip(t2["subject_id"], t2["t2_order"]))
    if df.empty:
        return df.copy()
    t2_order = df["subject_id"].map(chosen)
    keep = t2_order.notna() & ((df[order_col] == 1) | (df[order_col] == t2_order))
    out = df.loc[keep].copy()
    out.loc[out[order_col] != 1, order_col] = 2
    out["timepoint"] = out[order_col].map(T2_LABELS)
    return out


# Questionnaires that may come from a different post-baseline visit than the
# task sessions. BRIEF-A ("executive_function") is never one of them: it is a
# measurement level of the selection above and of the Convergence Index.
SUBSTITUTABLE_QUESTIONNAIRES = ("phq9", "gad7")


def apply_second_timepoint_questionnaires(
    df: pd.DataFrame,
    t2: pd.DataFrame,
    questionnaires=SUBSTITUTABLE_QUESTIONNAIRES,
    order_col: str = "task_order",
) -> pd.DataFrame:
    """``apply_second_timepoint`` for assessment rows, with questionnaire pairing.

    When a participant's selected second-timepoint
    visit has no administration of a questionnaire in ``questionnaires``
    (PHQ-9, GAD-7) but another post-baseline visit has one, that
    questionnaire's second-timepoint value comes from the post-baseline visit
    closest to the selected one (the earlier visit on a tie). Each
    questionnaire is paired on its own. BRIEF-A rows, and every other
    assessment, follow ``apply_second_timepoint`` unchanged, so BRIEF-A is
    never substituted. Descriptive use only: no questionnaire in
    ``questionnaires`` enters the selection, a hypothesis test or the
    Convergence Index.

    ``df`` needs ``subject_id``, ``assessment`` and ``order_col``. The output
    keeps baseline and the paired second-timepoint rows relabelled as order 2
    and adds ``source_task_order``, the visit each row was recorded at.
    """
    if df.empty:
        out = df.copy()
        out["source_task_order"] = pd.Series(dtype=int)
        return out
    chosen = dict(zip(t2["subject_id"], t2["t2_order"]))
    selected = df["subject_id"].map(chosen)

    source = selected.copy()
    is_q = df["assessment"].isin(questionnaires)
    present = (
        df[is_q & (df[order_col] > 1)]
        .groupby(["subject_id", "assessment"])[order_col]
        .agg(lambda s: sorted({int(o) for o in s}))
    )
    for (subject, assessment), orders in present.items():
        target = chosen.get(subject)
        if target is None or target in orders:
            continue
        # closest post-baseline visit; on a tie the earlier one
        nearest = min(orders, key=lambda o: (abs(o - target), o))
        rows = (df["subject_id"] == subject) & (df["assessment"] == assessment)
        source[rows] = nearest

    keep = selected.notna() & ((df[order_col] == 1) | (df[order_col] == source))
    out = df.loc[keep].copy()
    out["source_task_order"] = out[order_col].astype(int)
    out.loc[out[order_col] != 1, order_col] = 2
    out["timepoint"] = out[order_col].map(T2_LABELS)
    return out
