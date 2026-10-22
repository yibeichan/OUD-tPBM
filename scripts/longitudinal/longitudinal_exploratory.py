#!/usr/bin/env python3
"""Exploratory Longitudinal Case-Series Analysis.

This script performs comprehensive exploratory analysis of individual subject
trajectories across all available behavioral, neural, and assessment metrics.
It complements the hypothesis-driven unified_ef_longitudinal.py by providing:

- Coverage of ALL metrics (not just H1-H5)
- Individual trajectory visualizations
- Descriptive change statistics
- Pattern consistency analysis

Use this for:
- Exploratory data analysis
- Generating hypotheses for future studies
- Comprehensive individual subject characterization
- Supplementary materials

For hypothesis-driven analysis, see unified_ef_longitudinal.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subject_ids import sort_subjects, standardize_subject_id  # noqa: E402
from analysis_variant import load_variant, refuse_protected_tree  # noqa: E402
from brief_validity import valid_administration_mask  # noqa: E402
from session_exclusions import excluded_sessions  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
# roi_utils (imported where the GLM is read); appended so it shadows nothing
sys.path.append(str(Path(__file__).resolve().parent.parent / "brain"))
from timepoint_selection import (  # noqa: E402
    apply_second_timepoint,
    apply_second_timepoint_questionnaires,
    choose_second_timepoints,
)


# =============================================================================
# CONSTANTS
# =============================================================================

TIMEPOINT_LABELS = {1: "baseline", 2: "post", 3: "followup"}
TIMEPOINT_ORDER = {label: order for order, label in TIMEPOINT_LABELS.items()}

# Behavior metrics to track per task (ordered for plotting/reporting)
# EXPLORATORY: All available metrics, not just hypothesis-driven ones
BEHAVIOR_METRICS: Dict[str, Sequence[str]] = {
    "stroop": [
        # Core performance
        "overall_accuracy",
        "overall_rt",
        "overall_rt_cv",
        # Stroop-specific effects
        "mixing_cost_rt",
        "stroop_effect_rt",
        "mixing_cost_normalized",
        "stroop_effect_normalized",
        # Error monitoring
        "overall_error_rate",
        "conflict_adaptation_cost_rt",
    ],
    "wcst": [
        # Core performance
        "overall_accuracy",
        "exp_accuracy",
        "exp_correct_rt",
        # Perseveration (Heaton scoring)
        "perseverative_errors",
        "perseverative_error_rate",
        "non_perseverative_errors",
        "failures_to_maintain_set",
        # Learning efficiency
        "mean_trials_to_learn_rule",
        "set_shifting_efficiency",
        "categories_completed",
        "early_rule_accuracy",
        "late_rule_accuracy",
        # Working memory maintenance
        "wm_maintenance_decline",
        "exp_rt_cv",
    ],
}

# Direction encodes whether higher or lower values indicate improvement
# EXPLORATORY: Comprehensive coverage for all metrics
METRIC_DIRECTIONS: Dict[Tuple[str, str], str] = {
    ("stroop", "overall_accuracy"): "higher",
    ("stroop", "overall_rt"): "lower",
    ("stroop", "overall_rt_cv"): "lower",
    ("stroop", "mixing_cost_rt"): "lower",
    ("stroop", "stroop_effect_rt"): "lower",
    ("stroop", "mixing_cost_normalized"): "lower",
    ("stroop", "stroop_effect_normalized"): "lower",
    ("stroop", "overall_error_rate"): "lower",
    ("stroop", "conflict_adaptation_cost_rt"): "lower",
    # WCST metrics
    ("wcst", "overall_accuracy"): "higher",
    ("wcst", "exp_accuracy"): "higher",
    ("wcst", "exp_correct_rt"): "lower",
    ("wcst", "perseverative_errors"): "lower",
    ("wcst", "perseverative_error_rate"): "lower",
    ("wcst", "non_perseverative_errors"): "lower",
    ("wcst", "failures_to_maintain_set"): "lower",
    ("wcst", "mean_trials_to_learn_rule"): "lower",
    ("wcst", "set_shifting_efficiency"): "higher",
    ("wcst", "categories_completed"): "higher",
    ("wcst", "early_rule_accuracy"): "higher",
    ("wcst", "late_rule_accuracy"): "higher",
    ("wcst", "wm_maintenance_decline"): "lower",
    ("wcst", "exp_rt_cv"): "lower",
    # Assessment metrics (grouped by assessment type: phq9, gad7, executive_function)
    ("phq9", "phq9_total_score"): "lower",
    ("gad7", "gad7_total_score"): "lower",
    ("executive_function", "executive_function_inhibit_t"): "lower",  # T-score: lower = better
    ("executive_function", "executive_function_shift_t"): "lower",  # T-score: lower = better
    ("executive_function", "executive_function_working_memory_t"): "lower",  # T-score: lower = better
    ("executive_function", "executive_function_gec_t"): "lower",  # T-score: lower = better
    # Generic assessments key (fallback)
    ("assessments", "phq9_total_score"): "lower",
    ("assessments", "gad7_total_score"): "lower",
    ("assessments", "executive_function_inhibit_t"): "lower",
    ("assessments", "executive_function_shift_t"): "lower",
    ("assessments", "executive_function_working_memory_t"): "lower",
    ("assessments", "executive_function_gec_t"): "lower",
}

# No within-person effect size is computed: the long frames carry no
# per-session variance, so it would be NaN in every row.

# Assessment file mappings
ASSESSMENT_FILES: Dict[str, str] = {
    "phq9": "phq9_scores.csv",
    "gad7": "gad7_scores.csv",
    "executive_function": "executive_function_scores.csv",
}

# EXPLORATORY: Primary contrasts of interest for longitudinal tracking
BRAIN_CONTRAST_OF_INTEREST: Dict[str, str] = {
    "wcst": "learning",
    "stroop": "incongruent_vs_pure",  # Maximum contrast
}

# EXPLORATORY: All cognitive control ROIs for comprehensive tracking
# Using Schaefer network parcellation names
BRAIN_ROIS_OF_INTEREST: Dict[str, Sequence[str]] = {
    "stroop": [
        "Cont_PFCl",      # Left lateral PFC (cognitive control)
        "SalVentAttn_PFCl",  # Left ventral attention/insula
        "SalVentAttn_Med",   # Medial salience (ACC/vmPFC)
    ],
    "wcst": [
        "Cont_PFCl",      # Left lateral PFC
        "Cont_Par",       # Left parietal (set-shifting)
        "Default_PFC",    # Default mode PFC (learning)
    ],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def label_timepoint(order: int) -> str:
    """Map numeric task order to the reporting label."""
    return TIMEPOINT_LABELS.get(order, f"timepoint_{order}")


def with_source_visit(df: pd.DataFrame, t2: pd.DataFrame) -> pd.DataFrame:
    """Add the visit each row actually comes from.

    apply_second_timepoint relabels every participant's chosen second
    timepoint as order 2 ("post"), so a row from S3's or S5's follow-up reads
    as "post" in the written files. ``source_task_order`` and
    ``source_timepoint`` name the real visit; ``task_order``/``timepoint``
    stay the analysis labels.
    """
    if df.empty:
        return df
    chosen = dict(zip(t2["subject_id"], t2["t2_order"]))
    out = df.copy()
    second = out["subject_id"].map(chosen)
    if (out["task_order"] == 2).any() and second[out["task_order"] == 2].isna().any():
        raise ValueError("a second-timepoint row has no selection in t2")
    if "source_task_order" in out.columns:
        # Set upstream: a questionnaire paired from another post-baseline
        # visit keeps the visit it came from.
        # Moved to the end so the column order matches the other frames.
        out["source_task_order"] = out.pop("source_task_order").astype(int)
    else:
        out["source_task_order"] = np.where(out["task_order"] == 1, 1, second).astype(int)
    out["source_timepoint"] = out["source_task_order"].map(label_timepoint)
    return out


def order_from_label(label: str) -> int:
    """Map reported assessment labels to a numeric order."""
    return TIMEPOINT_ORDER.get(label.lower(), TIMEPOINT_ORDER.get(label, -1))


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def read_task_info(base_dir: Path, task: str) -> pd.DataFrame:
    """Read task info CSV and standardize column names.

    Every task_order is left as recorded; the second timepoint is chosen
    later by timepoint_selection.
    """
    path = base_dir / "data" / f"{task}_task_info.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing task info file: {path}")
    df = pd.read_csv(path)
    df.rename(columns={"subj_id": "subject_id"}, inplace=True)
    df["timepoint"] = df["task_order"].map(label_timepoint)
    return df


def load_behavior_metrics(
    output_dir: Path,
    task_info: pd.DataFrame,
    task: str,
    subjects: Sequence[str],
) -> pd.DataFrame:
    """Load behavioral metrics from task results JSON files.

    Returns DataFrame with columns:
        - subject_id
        - task_id
        - task_order
        - timepoint
        - task
        - source
        - All metric columns (overall_accuracy, rt, etc.)
    """
    records: List[Dict[str, object]] = []
    task_dir = output_dir / "02_behav" / task

    for row in task_info.itertuples():
        if row.subject_id not in subjects:
            continue

        result_path = task_dir / f"{row.subject_id}_{row.task_id}_results.json"
        # Every session left in task_info has been selected as usable, so a
        # missing results file is an error, not a gap.
        if not result_path.exists():
            raise FileNotFoundError(f"Behaviour results missing: {result_path}")

        with result_path.open("r") as fh:
            data = json.load(fh)

        metrics = data.get("metrics", {})
        record = {
            "subject_id": row.subject_id,
            "task_id": row.task_id,
            "task_order": int(row.task_order),
            "timepoint": label_timepoint(int(row.task_order)),
            "task": task,
            "source": str(result_path.relative_to(output_dir)),
        }

        for key, value in metrics.items():
            record[key] = value if value is not None else np.nan

        records.append(record)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values(["subject_id", "task", "task_order"], inplace=True)

    return df


def load_assessment_scores(
    output_dir: Path,
    subjects: Sequence[str],
    *,
    exclude_invalid_brief: bool,
) -> pd.DataFrame:
    """Load assessment scores including PHQ-9, GAD-7, and BRIEF-A T-scores.

    Every administration keeps its recorded order; the second timepoint is
    chosen later by timepoint_selection. ``exclude_invalid_brief`` drops BRIEF-A
    administrations that fail a validity scale (see brief_validity); the
    flags are read from the same t-score files the scores come from.
    PHQ-9 and GAD-7 rows are never filtered by it.

    Returns DataFrame with columns:
        - subject_id
        - assessment (phq9, gad7, executive_function)
        - metric
        - value
        - task_order
        - timepoint
    """
    rows = []

    # MODALITY 1: PHQ-9 and GAD-7 (Standard scores)
    for assessment, filename in [("phq9", "phq9_scores.csv"), ("gad7", "gad7_scores.csv")]:
        path = output_dir / "01_assessments" / "scores" / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["subject_id"] = df["subject"].apply(standardize_subject_id)
        df["timepoint_label"] = df["timepoint"].str.lower()
        df["task_order"] = df["timepoint_label"].map(order_from_label)
        df = df[df["subject_id"].isin(subjects)]

        for record in df.to_dict("records"):
            value = record.get("total_score")
            if value is None or pd.isna(value):
                continue

            rows.append({
                "subject_id": record["subject_id"],
                "assessment": assessment,
                "metric": f"{assessment}_total_score",
                "value": float(value),
                "task_order": int(record["task_order"]) if record.get("task_order") and record["task_order"] > 0 else np.nan,
                "timepoint": record["timepoint_label"],
            })

    # MODALITY 2: Executive Function T-SCORES
    # Files are: executive_function_{timepoint}_subscale_t-scores.csv
    exec_func_timepoints = {
        "baseline": 1,
        "post": 2,
        "followup": 3,
    }

    # EXPLORATORY: Include all BRIEF-A subscales
    tscore_mapping = {
        "executive_function_inhibit_t": "Inhibit_t",
        "executive_function_shift_t": "Shift_t",
        "executive_function_working_memory_t": "Working Memory_t",
        "executive_function_gec_t": "GEC_t",
    }

    for tp_label, t_order in exec_func_timepoints.items():
        tscore_path = (
            output_dir
            / "01_assessments"
            / "t-scores"
            / f"executive_function_{tp_label}_subscale_t-scores.csv"
        )
        if not tscore_path.exists():
            continue

        df = pd.read_csv(tscore_path)
        if exclude_invalid_brief:
            df = df[valid_administration_mask(df)]
        # standardize rather than rewrite the separator: a hyphenated "S-5"
        # would otherwise become "S_5", which matches no subject.
        df["subject_id"] = df["subject"].apply(standardize_subject_id)
        df = df[df["subject_id"].isin(subjects)]

        # Filter for rows that actually have data
        if "n_items" in df.columns:
            df = df[df["n_items"] > 0]

        for record in df.to_dict("records"):
            for out_metric, src_col in tscore_mapping.items():
                value = record.get(src_col)
                if value is None or pd.isna(value):
                    continue

                rows.append({
                    "subject_id": record["subject_id"],
                    "assessment": "executive_function",
                    "metric": out_metric,
                    "value": float(value),
                    "task_order": t_order,
                    "timepoint": tp_label,
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.dropna(subset=["task_order"])
        result.sort_values(["subject_id", "metric", "task_order"], inplace=True)

    return result


def load_roi_brain_activation(
    output_dir: Path,
    task_info: pd.DataFrame,
    task: str,
    subjects: Sequence[str],
    rois: Sequence[str],
    contrast: str,
) -> pd.DataFrame:
    """Load ROI-level activation from pre-computed ROI files.

    Returns DataFrame with columns:
        - subject_id
        - network_roi
        - contrast
        - activation
        - activation_std
        - task
        - task_order
        - timepoint
    """
    # The multi-timepoint extraction from extract_roi_alltimepoints.py.
    longi_file = (
        output_dir / "10_longitudinal" / "roi_activations_alltimepoints" / task
        / "all_subjects_roi_activation_network_roi.csv"
    )
    # Only the all-timepoints file has post/follow-up sessions; the
    # cross-sectional file is baseline-only, and reading it would turn every
    # brain level into insufficient_data without an error.
    if not longi_file.exists():
        raise FileNotFoundError(
            f"{longi_file} not found; run scripts/longitudinal/extract_roi_alltimepoints.py"
        )

    df = pd.read_csv(longi_file)
    # Standardize subject ID
    df["subject_id"] = df["subject"].apply(standardize_subject_id)

    # Every selected session of these subjects must be in the file; a stale
    # or partial extraction would otherwise drop brain levels silently.
    wanted = task_info[task_info["subject_id"].isin(subjects)]
    missing = sorted(set(wanted["task_id"]) - set(df["session_id"]))
    if missing:
        raise ValueError(f"{longi_file} lacks selected {task} session(s) {missing}")

    # Filter for eligible subjects, ROIs, and contrast
    subset = df[
        (df["subject_id"].isin(subjects))
        & (df["network_roi"].isin(rois))
        & (df["contrast"] == contrast)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    # Join with task_info to get task_order/timepoint
    merged = pd.merge(
        subset,
        task_info[["subject_id", "task_id", "task_order"]],
        left_on=["subject_id", "session_id"],
        right_on=["subject_id", "task_id"],
        how="inner",
    )

    if not merged.empty:
        merged["timepoint"] = merged["task_order"].map(label_timepoint)
        merged.sort_values(["subject_id", "network_roi", "task_order"], inplace=True)

    return merged


# =============================================================================
# DATA TRANSFORMATION FUNCTIONS
# =============================================================================


def melt_behavior(
    df: pd.DataFrame, metrics_by_task: Optional[Dict[str, Sequence[str]]] = None
) -> pd.DataFrame:
    """Convert wide behavior data to long format for plotting.

    Transforms from:
        subject_id | task | task_order | metric1 | metric2 | ...
    To:
        subject_id | task | task_order | metric | value

    Returns DataFrame with columns:
        - subject_id
        - task
        - task_order
        - timepoint
        - metric
        - value
    """
    if metrics_by_task is None:
        metrics_by_task = BEHAVIOR_METRICS
    frames = []
    for task, metrics in metrics_by_task.items():
        subset = df[df["task"] == task]
        if subset.empty:
            continue

        keep_cols = [col for col in metrics if col in subset.columns]
        melted = subset.melt(
            id_vars=["subject_id", "task", "task_order", "timepoint"],
            value_vars=keep_cols,
            var_name="metric",
            value_name="value",
        )
        frames.append(melted.dropna(subset=["value"]))

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame(
        columns=["subject_id", "task", "task_order", "timepoint", "metric", "value"]
    )


def compute_within_subject_changes(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    direction_lookup: Dict[Tuple[str, str], str],
    label_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute change scores and improvement direction.

    For each subject, computes sequential changes between timepoints:
        - delta: raw change score (post - baseline)
        - improved: whether change is in expected direction (bool; None if
          unchanged or the metric has no direction)

    Args:
        df: Long-format data with subject_id, task_order
        group_cols: Columns defining unique metrics (e.g., ["subject_id", "task"])
        value_col: Column with metric values
        direction_lookup: Dict mapping (task, metric) -> "higher"/"lower"
        label_cols: Additional columns to preserve as labels (e.g., ["metric"])

    Returns DataFrame with columns:
        - subject_id
        - from_timepoint
        - to_timepoint
        - delta
        - improved
        - All group_cols and label_cols
    """
    records: List[Dict[str, object]] = []
    label_cols = label_cols or []

    # Group by the label columns too: without them, consecutive rows of
    # different metrics were differenced against each other and labelled
    # with the first one.
    for keys, group in df.groupby(list(group_cols) + list(label_cols)):
        group = group.sort_values("task_order")

        for idx in range(len(group) - 1):
            current = group.iloc[idx]
            nxt = group.iloc[idx + 1]

            if current["task_order"] == nxt["task_order"]:
                continue

            # Determine metric direction; keys already end with the labels
            metric_key = tuple(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
            direction = direction_lookup.get(
                metric_key[-2:], direction_lookup.get(metric_key[-1:], None)
            )

            # Handle assessment direction lookup
            if direction is None:
                if len(metric_key) >= 2 and metric_key[-2] == "assessments":
                    direction = direction_lookup.get((metric_key[-2], metric_key[-1]))

            value_current = current[value_col]
            value_next = nxt[value_col]

            if pd.isna(value_current) or pd.isna(value_next):
                continue

            diff = value_next - value_current

            # Determine if change is improvement. An unchanged value is
            # neither (None), not a failure to improve.
            improved = None
            if diff == 0:
                improved = None
            elif direction == "higher":
                improved = diff > 0
            elif direction == "lower":
                improved = diff < 0

            record = {
                "subject_id": current["subject_id"],
                "from_timepoint": current["timepoint"],
                "to_timepoint": nxt["timepoint"],
                "delta": diff,
                "improved": improved,
            }

            for col in group_cols:
                record[col] = current[col]
            for col in label_cols:
                record[col] = current[col]

            records.append(record)

    return pd.DataFrame(records)


def compute_consistency(summary: pd.DataFrame, metric_col: str = "metric") -> pd.DataFrame:
    """Compute percentage of subjects showing improvement per metric.

    For each metric, calculates:
        - n_pairs: number of subject pairs that changed in a scored direction
        - n_unchanged: number of pairs with no change (left out of n_pairs)
        - n_improved: number showing improvement
        - pct_improved: percentage of n_pairs showing improvement

    Args:
        summary: Output from compute_within_subject_changes
        metric_col: Column name containing metric identifiers

    Returns DataFrame with one row per metric
    """
    records = []

    for metric, group in summary.groupby(metric_col):
        # astype(bool) before summing: with None present the column is
        # object-typed, and numpy booleans add as logical OR (True + True is
        # True), which would count every improvement as one.
        valid = group["improved"].dropna().astype(bool)
        total = len(valid)
        improved = int(valid.sum()) if total else 0
        n_unchanged = int((group["delta"] == 0).sum()) if "delta" in group else 0

        records.append({
            metric_col: metric,
            "n_pairs": total,
            "n_unchanged": n_unchanged,
            "n_improved": float(improved),
            "pct_improved": float(improved) / total * 100 if total else np.nan,
        })

    return pd.DataFrame(records)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def generate_behavior_plot(df: pd.DataFrame, subject: str, output_dir: Path) -> None:
    """Generate individual behavioral trajectory plot for a subject, split by task.

    Creates separate line plots for WCST and Stroop, showing behavioral metrics across timepoints.

    Args:
        df: Long-format behavior data
        subject: Subject ID
        output_dir: Directory to save plot
    """
    data = df[df["subject_id"] == subject]
    if data.empty:
        return

    data = data.copy()
    data["timepoint"] = pd.Categorical(
        data["timepoint"],
        categories=sorted(
            data["timepoint"].unique(), key=lambda x: TIMEPOINT_ORDER.get(x, 99)
        ),
        ordered=True,
    )

    for task in data["task"].unique():
        task_data = data[data["task"] == task]
        if task_data.empty:
            continue
            
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
        g = sns.relplot(
            data=task_data,
            x="timepoint",
            y="value",
            col="metric",
            col_wrap=3,
            kind="line",
            marker="o",
            markersize=8,
            linewidth=2.5,
            facet_kws={'sharey': False},
            height=3.5,
            aspect=1.2,
            color="royalblue"
        )
        
        g.fig.suptitle(f"Behavior Trajectories ({task.upper()}): {subject}", y=1.05, fontweight="bold")
        g.set_axis_labels("Timepoint", "Metric Value")
        g.set_titles(col_template="{col_name}")

        outfile = output_dir / f"{subject}_{task}_behavior_trajectory.png"
        g.savefig(outfile, dpi=300, bbox_inches="tight")
        g.savefig(outfile.with_suffix(".svg"), bbox_inches="tight")
        plt.close()



def generate_assessment_plot(df: pd.DataFrame, subject: str, output_dir: Path) -> None:
    """Generate individual assessment trajectory plot for a subject.

    Creates a line plot showing all assessment scores (PHQ-9, GAD-7, BRIEF-A)
    across timepoints.

    Args:
        df: Long-format assessment data
        subject: Subject ID
        output_dir: Directory to save plot
    """
    data = df[df["subject_id"] == subject]
    if data.empty:
        return

    data = data.copy()
    data["timepoint"] = pd.Categorical(
        data["timepoint"],
        categories=sorted(
            data["timepoint"].unique(), key=lambda x: TIMEPOINT_ORDER.get(x, 99)
        ),
        ordered=True,
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=data,
        x="timepoint",
        y="value",
        hue="metric",
        marker="o",
        linewidth=2.5,
    )

    plt.title(f"Assessment Scores: {subject}", fontsize=14, fontweight="bold")
    plt.xlabel("Timepoint", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outfile = output_dir / f"{subject}_assessment_trajectory.png"
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()


def generate_roi_trajectory_plot(df: pd.DataFrame, subject: str, output_dir: Path) -> List[Path]:
    """Generate one brain ROI trajectory plot per task for a subject.

    The ROI frame holds both tasks (Stroop and WCST, each with its own
    contrast), and both have a Cont_PFCl series. One plot per task keeps
    them apart; a single plot titled from the first row merged the two
    Cont_PFCl lines and showed WCST ROIs under a Stroop title.

    Returns the files written.
    """
    data = df[df["subject_id"] == subject]
    written: List[Path] = []
    if data.empty:
        return written

    data = data.copy()
    data["timepoint"] = pd.Categorical(
        data["timepoint"],
        categories=sorted(
            data["timepoint"].unique(), key=lambda x: TIMEPOINT_ORDER.get(x, 99)
        ),
        ordered=True,
    )

    for task in sorted(data["task"].unique()):
        task_data = data[data["task"] == task]
        contrasts = task_data["contrast"].unique()
        if len(contrasts) != 1:
            raise ValueError(
                f"{subject} {task}: expected one contrast in the ROI frame, got {list(contrasts)}")

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=task_data,
            x="timepoint",
            y="activation",
            hue="network_roi",
            marker="o",
            linewidth=2.5,
        )

        plt.title(
            f"Brain ROI Activation: {subject}\n{task.upper()} - {contrasts[0]}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Timepoint", fontsize=12)
        plt.ylabel("ROI Activation (HbO)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        outfile = output_dir / f"{subject}_{task}_brain_roi_trajectory.png"
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close()
        written.append(outfile)
    return written


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory Longitudinal Case-Series Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze every subject with a second timepoint
  python longitudinal_exploratory.py

  # Analyze specific subjects only
  python longitudinal_exploratory.py --subjects S1 S3 S7
        """,
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional list of subject IDs (e.g., S1 S3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override OUTPUT_DIR from environment",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Override BASE_DIR from environment",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    base_dir_env = os.getenv("BASE_DIR")
    output_dir_env = os.getenv("OUTPUT_DIR")

    if args.base_dir:
        base_dir = args.base_dir
    else:
        if not base_dir_env:
            raise SystemExit("ERROR: Set BASE_DIR environment variable or use --base-dir")
        base_dir = Path(base_dir_env)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        if not output_dir_env:
            raise SystemExit("ERROR: Set OUTPUT_DIR environment variable or use --output-dir")
        output_dir = Path(output_dir_env)
    try:
        refuse_protected_tree(str(output_dir))
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")

    print("=" * 80)
    print("EXPLORATORY LONGITUDINAL CASE-SERIES ANALYSIS")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directories
    exploratory_dir = output_dir / "10_longitudinal"
    figures_dir = exploratory_dir / "figures"
    exploratory_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    variant = load_variant(output_dir)
    print(f"Analysis variant: {json.dumps(dataclasses.asdict(variant))}")
    metrics_by_task = {task: list(metrics) for task, metrics in BEHAVIOR_METRICS.items()}
    # roi_utils.read_glm_contrasts resolves the run tree from the environment
    os.environ["OUTPUT_DIR"] = str(output_dir)
    os.environ["BASE_DIR"] = str(base_dir)

    # Load task info
    print("Loading task info...")
    stroop_info = read_task_info(base_dir, "stroop")
    wcst_info = read_task_info(base_dir, "wcst")

    if args.subjects:
        try:
            requested = sort_subjects(set(args.subjects))
        except ValueError as exc:
            raise SystemExit(
                f"ERROR: bad --subjects value: {exc}. Expected canonical "
                "S-codes -- the letter S followed by an unpadded number, "
                "e.g. --subjects S1 S3 S7."
            ) from exc
    else:
        requested = None

    # The same selection unified_ef_longitudinal.py makes: drop excluded
    # sessions and invalid BRIEF-A administrations, choose each subject's
    # second timepoint by level coverage, and keep only baseline and that
    # timepoint (relabelled post) in every modality.
    stroop_info = stroop_info[
        ~stroop_info["task_id"].isin(excluded_sessions("stroop", output_dir, base_dir))
    ]
    wcst_info = wcst_info[
        ~wcst_info["task_id"].isin(excluded_sessions("wcst", output_dir, base_dir))
    ]
    roster = sort_subjects(set(stroop_info["subject_id"]) | set(wcst_info["subject_id"]))
    assessments_df = load_assessment_scores(
        output_dir,
        roster,
        exclude_invalid_brief=variant.exclude_invalid_brief,
    )
    brief = (assessments_df[assessments_df["assessment"] == "executive_function"]
             if not assessments_df.empty else assessments_df)
    t2 = choose_second_timepoints({"stroop": stroop_info, "wcst": wcst_info}, brief)
    for rec in t2.itertuples(index=False):
        print(f"  {rec.subject_id}: second timepoint = task_order {rec.t2_order} "
              f"({rec.levels})")
    stroop_info = apply_second_timepoint(stroop_info, t2)
    wcst_info = apply_second_timepoint(wcst_info, t2)
    # A PHQ-9 or GAD-7 missing at the selected second-timepoint visit comes
    # from the closest post-baseline visit that has it. BRIEF-A is never
    # substituted. Descriptive only.
    assessments_df = apply_second_timepoint_questionnaires(assessments_df, t2)
    if not assessments_df.empty:
        t2_of = assessments_df["subject_id"].map(dict(zip(t2["subject_id"], t2["t2_order"])))
        paired = assessments_df[(assessments_df["task_order"] == 2)
                                & (assessments_df["source_task_order"] != t2_of)]
        for rec in paired[["subject_id", "assessment", "source_task_order"]].drop_duplicates().itertuples(index=False):
            print(f"  {rec.subject_id}: {rec.assessment} second timepoint from "
                  f"task_order {rec.source_task_order} (questionnaire pairing)")

    subjects = sort_subjects(set(t2["subject_id"]))
    if requested is not None:
        subjects = [s for s in requested if s in subjects]
        print(f"Using {len(subjects)} subjects from command line with a second timepoint: {subjects}")
    else:
        print(f"Selected {len(subjects)} subjects with a second timepoint: {subjects}")
    if not subjects:
        raise SystemExit("ERROR: No eligible subjects found")
    if not assessments_df.empty:
        assessments_df = assessments_df[assessments_df["subject_id"].isin(subjects)]
    print()

    # Load behavioral data
    print("Loading behavioral metrics...")
    behavior_data = []
    for task, info in [("stroop", stroop_info), ("wcst", wcst_info)]:
        df = load_behavior_metrics(output_dir, info, task, subjects)
        if not df.empty:
            behavior_data.append(df)
            print(f"  Loaded {len(df)} {task.upper()} behavioral records")

    behavior_df = pd.concat(behavior_data, ignore_index=True) if behavior_data else pd.DataFrame()

    # Convert to long format
    behavior_long = melt_behavior(behavior_df, metrics_by_task)
    if not behavior_long.empty:
        print(f"  Melted to {len(behavior_long)} long-format records")
    print()

    # Load assessment data
    print("Loading assessment scores...")
    if not assessments_df.empty:
        print(f"  Loaded {len(assessments_df)} assessment records")
        print(f"  Metrics: {sorted(assessments_df['metric'].unique())}")
    print()

    # Load brain ROI data
    print("Loading brain ROI activations...")
    brain_roi_data = []
    for task, info in [("stroop", stroop_info), ("wcst", wcst_info)]:
        rois = BRAIN_ROIS_OF_INTEREST.get(task, [])
        contrast = BRAIN_CONTRAST_OF_INTEREST.get(task)
        if not rois or not contrast:
            continue

        df = load_roi_brain_activation(output_dir, info, task, subjects, rois, contrast)
        if not df.empty:
            brain_roi_data.append(df)
            print(f"  Loaded {len(df)} {task.upper()} ROI records")

    brain_roi_df = (
        pd.concat(brain_roi_data, ignore_index=True) if brain_roi_data else pd.DataFrame()
    )
    print()

    # Compute within-subject changes
    print("Computing within-subject changes...")

    behavior_changes = pd.DataFrame()
    if not behavior_long.empty:
        behavior_changes = compute_within_subject_changes(
            behavior_long,
            group_cols=["subject_id", "task"],
            value_col="value",
            direction_lookup=METRIC_DIRECTIONS,
            label_cols=["metric"],
        )
        print(f"  Behavior: {len(behavior_changes)} change pairs")

    assessments_changes = pd.DataFrame()
    if not assessments_df.empty:
        assessments_changes = compute_within_subject_changes(
            assessments_df,
            group_cols=["subject_id", "assessment"],
            value_col="value",
            direction_lookup=METRIC_DIRECTIONS,
            label_cols=["metric"],
        )
        print(f"  Assessments: {len(assessments_changes)} change pairs")

    brain_roi_changes = pd.DataFrame()
    if not brain_roi_df.empty:
        brain_roi_changes = compute_within_subject_changes(
            brain_roi_df,
            group_cols=["subject_id", "task", "network_roi"],
            value_col="activation",
            direction_lookup={},  # No direction for brain (exploratory)
        )
        print(f"  Brain ROI: {len(brain_roi_changes)} change pairs")

    print()

    # Compute consistency summaries
    print("Computing consistency summaries...")

    consistency_frames = []
    if not behavior_changes.empty:
        # Combine task and metric for unique identifier
        behavior_changes_copy = behavior_changes.copy()
        behavior_changes_copy["metric_id"] = (
            behavior_changes_copy["task"] + "_" + behavior_changes_copy["metric"]
        )
        behavior_consistency = compute_consistency(
            behavior_changes_copy, metric_col="metric_id"
        )
        # Rename metric_id to metric for consistency
        behavior_consistency = behavior_consistency.rename(columns={"metric_id": "metric"})
        consistency_frames.append(behavior_consistency)
        print(f"  Behavior consistency: {len(behavior_consistency)} metrics")

    if not assessments_changes.empty:
        assessment_consistency = compute_consistency(assessments_changes)
        consistency_frames.append(assessment_consistency)
        print(f"  Assessment consistency: {len(assessment_consistency)} metrics")

    if consistency_frames:
        consistency_summary = pd.concat(consistency_frames, ignore_index=True)
    else:
        consistency_summary = pd.DataFrame()

    print()

    # Save outputs
    print("Saving outputs...")

    # Name the real visit behind each relabelled second-timepoint row.
    behavior_df = with_source_visit(behavior_df, t2)
    behavior_long = with_source_visit(behavior_long, t2)
    assessments_df = with_source_visit(assessments_df, t2)
    brain_roi_df = with_source_visit(brain_roi_df, t2)

    if not behavior_df.empty:
        behavior_df.to_csv(exploratory_dir / "behavior_wide.csv", index=False)
        print(f"  Saved: behavior_wide.csv ({len(behavior_df)} rows)")

    if not behavior_long.empty:
        behavior_long.to_csv(exploratory_dir / "behavior_long.csv", index=False)
        print(f"  Saved: behavior_long.csv ({len(behavior_long)} rows)")

    if not assessments_df.empty:
        assessments_df.to_csv(exploratory_dir / "assessments_long.csv", index=False)
        print(f"  Saved: assessments_long.csv ({len(assessments_df)} rows)")

    if not brain_roi_df.empty:
        brain_roi_df.to_csv(exploratory_dir / "brain_roi_activations.csv", index=False)
        print(f"  Saved: brain_roi_activations.csv ({len(brain_roi_df)} rows)")

    if not behavior_changes.empty:
        behavior_changes.to_csv(exploratory_dir / "behavior_changes.csv", index=False)
        print(f"  Saved: behavior_changes.csv ({len(behavior_changes)} rows)")

    if not assessments_changes.empty:
        assessments_changes.to_csv(exploratory_dir / "assessment_changes.csv", index=False)
        print(f"  Saved: assessment_changes.csv ({len(assessments_changes)} rows)")

    if not brain_roi_changes.empty:
        brain_roi_changes.to_csv(exploratory_dir / "brain_roi_changes.csv", index=False)
        print(f"  Saved: brain_roi_changes.csv ({len(brain_roi_changes)} rows)")

    if not consistency_summary.empty:
        consistency_summary.to_csv(exploratory_dir / "consistency_summary.csv", index=False)
        print(f"  Saved: consistency_summary.csv ({len(consistency_summary)} metrics)")

    print()

    # Generate figures
    print("Generating individual subject plots...")

    if not behavior_long.empty:
        for subject in subjects:
            generate_behavior_plot(behavior_long, subject, figures_dir)
        print(f"  Generated {len(subjects)} behavioral trajectory plots")

    if not assessments_df.empty:
        for subject in subjects:
            generate_assessment_plot(assessments_df, subject, figures_dir)
        print(f"  Generated {len(subjects)} assessment trajectory plots")

    if not brain_roi_df.empty:
        for subject in subjects:
            generate_roi_trajectory_plot(brain_roi_df, subject, figures_dir)
        print(f"  Generated {len(subjects)} brain ROI trajectory plots")

    print()

    # Generate summary JSON
    print("Generating analysis summary...")

    summary = {
        "analysis_type": "exploratory_longitudinal",
        "n_subjects": len(subjects),
        "subjects": subjects,
        "timepoints_analyzed": "baseline_to_second_timepoint",
        "behavior_metrics": {
            "stroop": len(metrics_by_task["stroop"]),
            "wcst": len(metrics_by_task["wcst"]),
        },
        "data_coverage": {
            "behavior_pairs": len(behavior_changes) if not behavior_changes.empty else 0,
            "assessment_pairs": len(assessments_changes) if not assessments_changes.empty else 0,
            "brain_roi_pairs": len(brain_roi_changes) if not brain_roi_changes.empty else 0,
        },
        "outputs": {
            "behavior_wide": str(
                (exploratory_dir / "behavior_wide.csv").relative_to(output_dir)
            ),
            "behavior_long": str(
                (exploratory_dir / "behavior_long.csv").relative_to(output_dir)
            ),
            "assessments": str(
                (exploratory_dir / "assessments_long.csv").relative_to(output_dir)
            ),
            "brain_roi": str(
                (exploratory_dir / "brain_roi_activations.csv").relative_to(output_dir)
            ),
            "behavior_changes": str(
                (exploratory_dir / "behavior_changes.csv").relative_to(output_dir)
            ),
            "assessment_changes": str(
                (exploratory_dir / "assessment_changes.csv").relative_to(output_dir)
            ),
            "brain_roi_changes": str(
                (exploratory_dir / "brain_roi_changes.csv").relative_to(output_dir)
            ),
            "consistency": str(
                (exploratory_dir / "consistency_summary.csv").relative_to(output_dir)
            ),
            "figures": str(figures_dir.relative_to(output_dir)),
        },
    }

    with (exploratory_dir / "analysis_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"  Saved: analysis_summary.json")
    print()

    # Print consistency summary
    if not consistency_summary.empty:
        print("=" * 80)
        print("CONSISTENCY SUMMARY (Percentage of Subjects Showing Improvement)")
        print("=" * 80)

        # Drop NaN metrics before filtering
        summary_clean = consistency_summary.dropna(subset=["metric"])

        # Group by assessment type for cleaner display
        for assessment_type in ["executive_function", "phq9", "gad7"]:
            type_data = summary_clean[
                summary_clean["metric"].str.startswith(assessment_type)
            ]
            if not type_data.empty:
                print(f"\n{assessment_type.upper()}:")
                for _, row in type_data.iterrows():
                    print(
                        f"  {row['metric']}: {row['pct_improved']:.1f}% "
                        f"({row['n_improved']:.0f}/{row['n_pairs']:.0f} subjects)"
                    )

        # Behavior metrics
        print("\nBEHAVIOR:")
        for task in ["stroop", "wcst"]:
            task_data = summary_clean[
                summary_clean["metric"].str.startswith(task)
            ]
            if not task_data.empty:
                print(f"\n{task.upper()}:")
                for _, row in task_data.iterrows():
                    print(
                        f"  {row['metric']}: {row['pct_improved']:.1f}% "
                        f"({row['n_improved']:.0f}/{row['n_pairs']:.0f} subjects)"
                    )

        print()

    print("=" * 80)
    print(f"All outputs saved to: {exploratory_dir}")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
