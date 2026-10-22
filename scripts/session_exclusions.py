"""Task sessions whose accuracy is at or below chance.

A session's accuracy metric is read from the behavioral results JSON that
``01_stroop_behav.py`` / ``01_wcst_behav.py`` wrote. A session at or below
chance accuracy is a candidate for exclusion from brain and behavior
analyses; whether it is actually dropped is gated by the run's
``analysis_variant.json`` (``exclude_low_accuracy``), read via
``load_variant``. ``write_exclusion_log`` always records the rule's result,
independent of that flag, so the excluded sessions are visible even on a
variant that keeps them in. A missing or non-finite accuracy is an error,
not a pass.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

from analysis_variant import load_variant

CHANCE_ACCURACY = {"stroop": 1 / 3, "wcst": 1 / 4}
ACCURACY_METRIC = {"stroop": "overall_accuracy", "wcst": "exp_accuracy"}


def _excluded_rows(output_dir, base_dir) -> list[dict]:
    """Return one row per session at or below chance accuracy, across tasks."""
    rows = []
    for task, metric in ACCURACY_METRIC.items():
        task_info = pd.read_csv(Path(base_dir) / "data" / f"{task}_task_info.csv")
        for row in task_info.itertuples(index=False):
            path = (
                Path(output_dir) / "02_behav" / task
                / f"{row.subj_id}_{row.task_id}_results.json"
            )
            if not path.is_file():
                raise FileNotFoundError(f"behavioral results not found: {path}")
            accuracy = json.loads(path.read_text())["metrics"][metric]
            if (isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
                    or not math.isfinite(accuracy)):
                raise ValueError(f"{path}: {metric} is {accuracy!r}, expected a finite number")
            if accuracy <= CHANCE_ACCURACY[task]:
                rows.append({
                    "task": task,
                    "subject": row.subj_id,
                    "session_id": row.task_id,
                    "task_order": int(row.task_order),
                    "accuracy_metric": metric,
                    "accuracy": accuracy,
                })
    return rows


def low_accuracy_sessions(output_dir, base_dir) -> dict[str, frozenset[str]]:
    """Return {task: frozenset of session_ids} at or below chance accuracy."""
    rows = _excluded_rows(output_dir, base_dir)
    return {
        task: frozenset(r["session_id"] for r in rows if r["task"] == task)
        for task in ACCURACY_METRIC
    }


def excluded_sessions(task, output_dir=None, base_dir=None) -> frozenset[str]:
    """Return the sessions to actually exclude for ``task``.

    Empty unless the run's variant sets ``exclude_low_accuracy``.
    """
    variant = load_variant(output_dir)
    if not variant.exclude_low_accuracy:
        return frozenset()
    if output_dir is None:
        output_dir = os.environ["OUTPUT_DIR"]
    if base_dir is None:
        base_dir = os.environ["BASE_DIR"]
    return low_accuracy_sessions(output_dir, base_dir)[task]


def write_exclusion_log(output_dir, base_dir) -> None:
    """Write the low-accuracy sessions to ``04b_session_info``, gate-independent."""
    rows = _excluded_rows(output_dir, base_dir)
    out_dir = Path(output_dir) / "04b_session_info"
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["task", "subject", "session_id", "task_order", "accuracy_metric", "accuracy"]
    pd.DataFrame(rows, columns=columns).to_csv(
        out_dir / "low_accuracy_sessions.csv", index=False
    )
