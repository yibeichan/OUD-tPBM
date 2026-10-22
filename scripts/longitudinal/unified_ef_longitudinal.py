#!/usr/bin/env python3
"""Unified EF Longitudinal Analysis - H1-H5 Convergence Testing.

This script tests whether subjects show convergent improvements across self-report,
neural activation, and behavioral performance domains for executive function
constructs. It implements:

1. Convergence Index (CI) measuring directional consistency across domains,
   which alone sets each classification
2. Reliable Change Index (RCI) per domain (threshold 3.18, the two-tailed
   t critical value for df=3; the baseline SD behind it has df=4 with n=5),
   written to the outputs but not used to classify or reported

Hypotheses H1-H5:
    H1: Inhibition (Inhibit_t, stroop_effect, interference)
    H2: Flexibility (Shift_t, change_vs_regular, perseverative_error_rate in main;
        perseverative_errors under a variant that sets h2_metric accordingly)
    H3: Learning (WM_t, learning, wm_maintenance_decline)
    H4: Attention (WM_t, mixing_cost, rt_variability)
    H5: Global EF (GEC_t, global_performance_z)

BEHAVIORAL METRIC NAMES:
    Uses theoretically meaningful construct names (e.g., 'interference', 'rt_variability')
    that map directly to executive function hypotheses. The per-session results JSON
    stores them under their computed names (stroop_effect_rt, overall_rt_cv), so
    load_behavioral_metrics adds the construct names via BEHAVIORAL_CONSTRUCT_SOURCES.

ROI DEFINITION (HARMONIZED WITH CROSS-SECTIONAL ANALYSIS):
- Uses sub-regional ROIs for theoretical precision (e.g., SalVentAttn_PFCl, Cont_PFCl, Cont_Par)
- Sub-regional approach preserves effect sizes better than network-level aggregation
- With n=12 cross-sectional and n=5 longitudinal, theoretical precision > statistical convenience
- Network-level aggregation would further dilute already-small effects

Rationale:
- H1 Inhibition: IFG (SalVentAttn_PFCl) is core of inhibitory control (Aron, 2007)
- H2 Flexibility: dlPFC + IPL network for set-shifting (Cont_PFCl + Cont_Par)
- H3 Learning: dlPFC feedback processing (Cont_PFCl + Cont_Par)
- H4 Attention: Sustained control in dlPFC (Cont_PFCl)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subject_ids import sort_subjects, standardize_subject_id  # noqa: E402
from seeding import DEFAULT_SEED  # noqa: E402
from analysis_variant import load_variant, refuse_protected_tree  # noqa: E402
from brief_validity import valid_administration_mask  # noqa: E402
from session_exclusions import excluded_sessions  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from timepoint_selection import (  # noqa: E402
    apply_second_timepoint,
    choose_second_timepoints,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Timepoint mappings
TIMEPOINT_LABELS = {1: "baseline", 2: "post"}

# Hypotheses definition
HYPOTHESES = {
    "H1": {
        "name": "Inhibition",
        "self_report": "Inhibit_t",
        "neural_contrast": "stroop_effect",
        "neural_roi": ["SalVentAttn_PFCl", "Cont_PFCl"],  # Sub-regional ROIs (harmonized with cross-sectional)
        "behavioral": "interference",  # Theoretically meaningful: maps to H1 Inhibition construct
        "task": "stroop",
        "improvement_direction": {"self_report": -1, "neural": None, "behavioral": -1},
        "description": "Inhibitory control improved (lower interference)",
    },
    "H2": {
        "name": "Flexibility",
        "self_report": "Shift_t",
        "neural_contrast": "change_vs_regular",
        "neural_roi": ["Cont_PFCl", "Cont_Par"],  # Sub-regional ROIs (harmonized with cross-sectional)
        "behavioral": "perseverative_errors",
        "task": "wcst",
        "improvement_direction": {"self_report": -1, "neural": None, "behavioral": -1},
        "description": "Cognitive flexibility improved (lower perseveration)",
    },
    "H3": {
        "name": "Learning",
        "self_report": "Working Memory_t",
        "neural_contrast": "learning",
        "neural_roi": ["Cont_PFCl", "Cont_Par"],  # Sub-regional ROIs (harmonized with cross-sectional)
        "behavioral": "wm_maintenance_decline",
        "task": "wcst",
        "improvement_direction": {"self_report": -1, "neural": None, "behavioral": -1},
        "description": "Working memory learning improved (less decline)",
    },
    "H4": {
        "name": "Attention",
        "self_report": "Working Memory_t",
        "neural_contrast": "mixing_cost",
        "neural_roi": ["Cont_PFCl"],  # Sub-regional ROI (harmonized with cross-sectional)
        "behavioral": "rt_variability",  # Theoretically meaningful: maps to H4 Attentional Stability construct
        "task": "stroop",
        "improvement_direction": {"self_report": -1, "neural": None, "behavioral": -1},
        "description": "Attentional control improved (lower RT variability)",
    },
    "H5": {
        "name": "Global EF",
        "self_report": "GEC_t",
        "neural_contrast": None,  # No neural measure for H5
        "neural_roi": [],
        "behavioral": "global_performance_z",
        "task": "mixed",  # Special case: combines both tasks
        "improvement_direction": {"self_report": -1, "neural": None, "behavioral": 1},
        "description": "Global EF improved (lower GEC, higher performance)",
    },
}

# HYPOTHESES names behavioural arms by construct; the per-session results JSON
# stores them under their computed names. Same mapping as the cross-sectional
# loader (scripts/hypo_test/utils/statistical_methods.py). Without it the
# classifier's metric filter matches nothing and H1/H4 lose their behavioural
# arm silently.
BEHAVIORAL_CONSTRUCT_SOURCES = {
    "interference": "stroop_effect_rt",
    "rt_variability": "overall_rt_cv",
}


def resolve_longitudinal_hypotheses(hypotheses: Dict[str, Dict], variant) -> Dict[str, Dict]:
    """Return a deep copy of ``hypotheses`` with H2's behavioral metric set from ``variant``.

    Only ``hypotheses["H2"]["behavioral"]`` is replaced, with
    ``variant.h2_metric``; every other hypothesis is returned unchanged. The
    module-level ``HYPOTHESES`` keeps ``"perseverative_errors"`` as its own
    default -- callers that read it directly (tests, other scripts) still see
    a stable value -- and it is only ``main()`` that resolves against the
    run's variant.
    """
    resolved = copy.deepcopy(hypotheses)
    resolved["H2"]["behavioral"] = variant.h2_metric
    return resolved


# RCI threshold: t(df=3, alpha=0.05, two-tailed). Conservative for this
# cohort; only written to the outputs, never used to classify.
RCI_THRESHOLD = 3.18
RCI_TREND_THRESHOLD = 1.96  # Standard z-score threshold for trends

# T-score reliability estimates for SEM calculation
# These are conservative estimates based on test-retest reliability studies
BRIEF_A_RELIABILITY = {
    "Inhibit_t": 0.85,
    "Shift_t": 0.82,
    "Working Memory_t": 0.88,
    "GEC_t": 0.90,
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def label_timepoint(order: int) -> str:
    """Map numeric task order to the reporting label."""
    return TIMEPOINT_LABELS.get(order, f"timepoint_{order}")


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


def load_assessment_t_scores(
    output_dir: Path,
    subjects: List[str],
    *,
    exclude_invalid: bool,
) -> pd.DataFrame:
    """Load BRIEF-A T-scores for executive function subscales.

    ``exclude_invalid`` drops administrations that fail a validity scale
    (see brief_validity). Every administration keeps its recorded order;
    the second timepoint is chosen later by timepoint_selection.

    Returns DataFrame with columns:
        - subject_id
        - metric (e.g., Inhibit_t, Shift_t, WM_t, GEC_t)
        - value
        - task_order (1=baseline, 2=post, 3=followup)
        - timepoint (baseline/post/followup)
    """
    rows = []
    timepoint_map = {"baseline": 1, "post": 2, "followup": 3}

    # T-score file naming convention
    for tp_label, t_order in timepoint_map.items():
        tscore_path = (
            output_dir
            / "01_assessments"
            / "t-scores"
            / f"executive_function_{tp_label}_subscale_t-scores.csv"
        )
        if not tscore_path.exists():
            continue

        df = pd.read_csv(tscore_path)
        if exclude_invalid:
            df = df[valid_administration_mask(df)]
        # standardize rather than rewrite the separator: a hyphenated "S-5"
        # would otherwise become "S_5", which matches no subject.
        df["subject_id"] = df["subject"].apply(standardize_subject_id)
        df = df[df["subject_id"].isin(subjects)]

        # Filter for rows with valid data
        if "n_items" in df.columns:
            df = df[df["n_items"] > 0]

        # Map column names to standardized metric names
        tscore_mapping = {
            "Inhibit_t": "Inhibit_t",
            "Shift_t": "Shift_t",
            "Working Memory_t": "Working Memory_t",
            "GEC_t": "GEC_t",
        }

        for record in df.to_dict("records"):
            for out_metric, src_col in tscore_mapping.items():
                value = record.get(src_col)
                if value is None or pd.isna(value):
                    continue

                rows.append({
                    "subject_id": record["subject_id"],
                    "metric": out_metric,
                    "value": float(value),
                    "task_order": t_order,
                    "timepoint": tp_label,
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values(["subject_id", "metric", "task_order"], inplace=True)

    return result


def load_behavioral_metrics(
    output_dir: Path, task_info: pd.DataFrame, task: str, subjects: List[str]
) -> pd.DataFrame:
    """Load behavioral metrics from task results JSON files.

    Returns DataFrame with columns:
        - subject_id
        - task
        - metric (e.g., stroop_effect_rt, perseverative_errors)
        - value
        - task_order
        - timepoint
    """
    records = []
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
            "task": task,
            "task_order": int(row.task_order),
            "timepoint": label_timepoint(int(row.task_order)),
        }

        for key, value in metrics.items():
            record[key] = value if value is not None else np.nan
        for construct, source in BEHAVIORAL_CONSTRUCT_SOURCES.items():
            if source in record:
                record[construct] = record[source]

        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values(["subject_id", "task", "task_order"], inplace=True)

    return df


def load_roi_activations(
    output_dir: Path,
    task_info: pd.DataFrame,
    task: str,
    subjects: List[str],
    rois: List[str],
    contrast: str,
) -> pd.DataFrame:
    """Load ROI-level activations from pre-computed ROI files.

    Returns DataFrame with columns:
        - subject_id
        - task
        - network_roi
        - contrast
        - activation (mean across ROI)
        - activation_std
        - task_order
        - timepoint
    """
    # The multi-timepoint extraction from extract_roi_alltimepoints.py.
    longi_file = (
        output_dir
        / "10_longitudinal"
        / "roi_activations_alltimepoints"
        / task
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
    df["subject_id"] = df["subject"].apply(standardize_subject_id)

    # Every selected session of these subjects must be in the file; a stale
    # or partial extraction would otherwise drop brain levels silently.
    wanted = task_info[task_info["subject_id"].isin(subjects)]
    missing = sorted(set(wanted["task_id"]) - set(df["session_id"]))
    if missing:
        raise ValueError(f"{longi_file} lacks selected {task} session(s) {missing}")

    # Filter for subjects, ROIs, and contrast
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


def compute_global_performance_z(
    stroop_df: pd.DataFrame, wcst_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute global performance z-score combining Stroop and WCST.

    Global performance = mean of:
        - Stroop overall_accuracy
        - WCST prop_rule_periods_mastered (categories completed / rule
          periods; the count alone tracks trials completed)
    Each is z-scored with the mean and SD of the baseline (task_order=1)
    sessions of every subject that has one in the loaded data, then applied
    to all timepoints.

    Higher values indicate better performance. A subject-timepoint is
    dropped unless both tasks contributed a z-score to it: the composite
    is a Stroop+WCST average, and averaging over whichever task happens to
    be present would silently turn a one-task value into a two-task
    composite.

    Returns DataFrame with columns:
        - subject_id
        - metric (global_performance_z)
        - value
        - task_order
        - timepoint
    """
    records = []

    # Process each task separately
    for task, df, metric_name in [
        ("stroop", stroop_df, "overall_accuracy"),
        ("wcst", wcst_df, "prop_rule_periods_mastered"),
    ]:
        if df.empty:
            continue
        # A loaded task without its H5 metric stops here, as the
        # cross-sectional loader does, rather than dropping that arm of H5.
        if metric_name not in df.columns:
            raise KeyError(
                f"{task} behaviour has no {metric_name!r} column; H5 needs it")

        # Get baseline statistics for z-scoring
        baseline = df[df["task_order"] == 1]
        if baseline.empty:
            continue

        baseline_mean = baseline[metric_name].mean()
        baseline_std = baseline[metric_name].std()

        if pd.isna(baseline_std) or baseline_std == 0:
            continue

        # Z-score all timepoints using baseline parameters
        for row in df.itertuples():
            value = getattr(row, metric_name)
            if pd.isna(value):
                continue

            z_score = (value - baseline_mean) / baseline_std

            records.append({
                "subject_id": row.subject_id,
                "task": task,
                "metric": f"{metric_name}_z",
                "value": z_score,
                "task_order": row.task_order,
                "timepoint": row.timepoint,
            })

    if not records:
        return pd.DataFrame()

    z_df = pd.DataFrame(records)

    # Combine across tasks to get global performance z
    # Pivot wider to have one row per subject-timepoint
    wide = z_df.pivot_table(
        index=["subject_id", "task_order", "timepoint"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )

    # Both tasks' z-scores must be present for a subject-timepoint to get a
    # composite; a column is entirely absent when a task loaded no data at
    # all, so add it as all-NaN rather than let the mask below raise.
    required_cols = ["overall_accuracy_z", "prop_rule_periods_mastered_z"]
    for col in required_cols:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide[wide[required_cols].notna().all(axis=1)]

    wide["global_performance_z"] = wide[required_cols].mean(axis=1)

    # Reset index for long format
    result = wide.reset_index()[["subject_id", "task_order", "timepoint", "global_performance_z"]]

    return result


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================


def compute_rci(
    baseline_value: float,
    post_value: float,
    baseline_sd: float,
    reliability: float,
) -> Tuple[float, str]:
    """Compute Reliable Change Index with small-sample adjustments.

    Args:
        baseline_value: Baseline score
        post_value: Post-treatment score
        baseline_sd: Standard deviation at baseline (across subjects)
        reliability: Test-retest reliability coefficient

    Returns:
        (RCI, classification) where classification is one of:
            - "reliable_change": |RCI| >= 3.18
            - "trend": 1.96 <= |RCI| < 3.18
            - "stable": |RCI| < 1.96
    """
    if pd.isna(baseline_value) or pd.isna(post_value) or pd.isna(baseline_sd):
        return np.nan, "missing"

    if baseline_sd == 0:
        return np.nan, "invalid"

    # Standard Error of Measurement
    sem = baseline_sd * np.sqrt(1 - reliability)

    if sem == 0:
        return np.nan, "invalid"

    # RCI calculation
    rci = (post_value - baseline_value) / (np.sqrt(2) * sem)

    # Classification
    abs_rci = abs(rci)
    if abs_rci >= RCI_THRESHOLD:
        classification = "reliable_change"
    elif abs_rci >= RCI_TREND_THRESHOLD:
        classification = "trend"
    else:
        classification = "stable"

    return rci, classification


def compute_convergence_index(
    baseline_values: Dict[str, float],
    post_values: Dict[str, float],
    baseline_sds: Dict[str, float],
    directions: Dict[str, int],
) -> Tuple[float, str, Dict[str, float]]:
    """Compute Convergence Index measuring directional consistency across domains.

    CI = mean of direction signs for all variables
        - Direction: -1 if change is toward improvement, +1 if away from improvement
        - Improvement direction determined by 'directions' dict

    Args:
        baseline_values: Dict mapping domain->baseline value
        post_values: Dict mapping domain->post value
        baseline_sds: Dict mapping domain->baseline SD (for z-scoring)
        directions: Dict mapping domain->expected direction (-1=lower is better, +1=higher is better)

    Returns:
        (CI, classification, domain_directions) where:
            - CI: Convergence Index (-1 to +1)
            - classification: One of "convergent_improvement", "partial_convergence",
              "mixed", "partial_divergence", "divergent"
            - domain_directions: Dict with individual direction calculations
    """
    domain_directions = {}
    valid_domains = []

    for domain in baseline_values.keys():
        if domain not in post_values or domain not in baseline_sds:
            continue

        base_val = baseline_values[domain]
        post_val = post_values[domain]
        base_sd = baseline_sds[domain]

        if pd.isna(base_val) or pd.isna(post_val) or pd.isna(base_sd):
            continue

        if base_sd == 0:
            continue

        # Standardized change score
        z_change = (post_val - base_val) / base_sd

        # Determine if change is toward improvement
        expected_dir = directions.get(domain, -1)  # Default: lower is better

        # Direction: -1 if moving toward improvement, +1 if moving away
        if expected_dir == -1:  # Lower is better
            direction = -1 if z_change < 0 else (1 if z_change > 0 else 0)
        else:  # Higher is better
            direction = -1 if z_change > 0 else (1 if z_change < 0 else 0)

        domain_directions[domain] = {
            "z_change": z_change,
            "direction": direction,
        }
        valid_domains.append(direction)

    if not valid_domains:
        return np.nan, "insufficient_data", domain_directions

    # Compute CI as mean of directions
    ci = np.mean(valid_domains)

    # Classification
    if ci <= -0.67:
        classification = "convergent_improvement"
    elif ci <= -0.33:
        classification = "partial_convergence"
    elif ci < 0.33:
        classification = "mixed"
    elif ci < 0.67:
        classification = "partial_divergence"
    else:
        classification = "divergent"

    return ci, classification, domain_directions


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def run_hypothesis_analysis(
    hypothesis_id: str,
    hypothesis_config: Dict,
    assessment_data: pd.DataFrame,
    behavior_data: pd.DataFrame,
    roi_data: pd.DataFrame,
    subjects: List[str],
    brain_direction: int,
) -> pd.DataFrame:
    """Run convergence analysis for a single hypothesis.

    ``brain_direction`` is the a-priori expected direction for the neural
    domain (-1 = lower activation is the improvement, +1 = higher is), fed
    into the convergence index as an assumption -- it is never estimated
    from the data.

    Returns DataFrame with one row per subject containing:
        - subject_id
        - hypothesis
        - All change scores (delta_self_report, delta_brain, delta_behavior)
        - RCI values for each domain
        - Convergence Index
        - Classification
    """
    results = []
    hyp_name = hypothesis_config["name"]

    # Get relevant metrics
    self_report_metric = hypothesis_config["self_report"]
    behavioral_metric = hypothesis_config["behavioral"]
    task = hypothesis_config["task"]
    neural_contrast = hypothesis_config["neural_contrast"]
    neural_rois = hypothesis_config["neural_roi"]

    # Special handling for H5 (global EF)
    if hypothesis_id == "H5":
        # H5 uses global_performance_z (already computed)
        behavioral_metric = "global_performance_z"

    for subject in subjects:
        result = {
            "subject_id": subject,
            "hypothesis": hypothesis_id,
            "hypothesis_name": hyp_name,
        }

        # === SELF-REPORT ===
        # Restrict to baseline (task_order=1) and post (task_order=2); ignore
        # additional timepoints (e.g., followup) when present so that subjects
        # with three or more loaded timepoints are not falsely flagged
        # insufficient by a strict total-row-count check.
        sr_subset = assessment_data[
            (assessment_data["subject_id"] == subject)
            & (assessment_data["metric"] == self_report_metric)
            & (assessment_data["task_order"].isin([1, 2]))
        ]

        baseline_row = sr_subset[sr_subset["task_order"] == 1]
        post_row = sr_subset[sr_subset["task_order"] == 2]

        if not baseline_row.empty and not post_row.empty:
            base_val = baseline_row["value"].iloc[0]
            post_val = post_row["value"].iloc[0]

            result["self_report_baseline"] = base_val
            result["self_report_post"] = post_val
            result["delta_self_report"] = post_val - base_val

            # Compute baseline SD for RCI
            base_sd = assessment_data[
                (assessment_data["task_order"] == 1)
                & (assessment_data["metric"] == self_report_metric)
            ]["value"].std()

            # Get reliability
            reliability = BRIEF_A_RELIABILITY.get(self_report_metric, 0.85)

            rci, rci_class = compute_rci(base_val, post_val, base_sd, reliability)
            result["rci_self_report"] = rci
            result["rci_self_report_class"] = rci_class
        else:
            result["self_report_baseline"] = np.nan
            result["self_report_post"] = np.nan
            result["delta_self_report"] = np.nan
            result["rci_self_report"] = np.nan
            result["rci_self_report_class"] = "insufficient_data"

        # === NEURAL (if applicable) ===
        if neural_contrast and neural_rois:
            # Get ROI activations for this hypothesis
            if task == "mixed":
                # For H5, no neural measure
                result["brain_baseline"] = np.nan
                result["brain_post"] = np.nan
                result["delta_brain"] = np.nan
                result["rci_brain"] = np.nan
                result["rci_brain_class"] = "not_applicable"
            else:
                # Filter for this task's ROI data
                task_roi_data = roi_data[roi_data["task"] == task]

                roi_subset = task_roi_data[
                    (task_roi_data["subject_id"] == subject)
                    & (task_roi_data["contrast"] == neural_contrast)
                    & (task_roi_data["network_roi"].isin(neural_rois))
                ]

                if len(roi_subset) > 0:
                    # Aggregate across ROIs (mean activation)
                    roi_wide = roi_subset.pivot_table(
                        index="task_order",
                        columns="network_roi",
                        values="activation",
                        aggfunc="mean",
                    )

                    # Compute mean across ROIs
                    # Both timepoints must carry the same ROI set: a mean
                    # over two ROIs against a mean over one is not a change.
                    both = 1 in roi_wide.index and 2 in roi_wide.index
                    same_rois = both and bool(roi_wide.loc[[1, 2]].notna().all().all())
                    if both and not same_rois:
                        print(f"  WARNING: {subject} {hypothesis_id}: ROI set differs between "
                              f"baseline and post ({neural_rois}); brain level left out")
                    if same_rois:
                        base_val = roi_wide.loc[1].mean()
                        post_val = roi_wide.loc[2].mean()

                        result["brain_baseline"] = base_val
                        result["brain_post"] = post_val
                        result["delta_brain"] = post_val - base_val

                        # Compute baseline SD for RCI
                        base_data = task_roi_data[
                            (task_roi_data["task_order"] == 1)
                            & (task_roi_data["contrast"] == neural_contrast)
                            & (task_roi_data["network_roi"].isin(neural_rois))
                        ]
                        base_agg = (
                            base_data.groupby("subject_id")["activation"].mean().reset_index()
                        )
                        base_sd = base_agg["activation"].std()

                        # Neural reliability estimate (conservative)
                        reliability = 0.75  # fNIRS test-retest reliability

                        rci, rci_class = compute_rci(base_val, post_val, base_sd, reliability)
                        result["rci_brain"] = rci
                        result["rci_brain_class"] = rci_class
                    else:
                        result["brain_baseline"] = np.nan
                        result["brain_post"] = np.nan
                        result["delta_brain"] = np.nan
                        result["rci_brain"] = np.nan
                        result["rci_brain_class"] = "insufficient_data"
                else:
                    result["brain_baseline"] = np.nan
                    result["brain_post"] = np.nan
                    result["delta_brain"] = np.nan
                    result["rci_brain"] = np.nan
                    result["rci_brain_class"] = "insufficient_data"
        else:
            result["brain_baseline"] = np.nan
            result["brain_post"] = np.nan
            result["delta_brain"] = np.nan
            result["rci_brain"] = np.nan
            result["rci_brain_class"] = "not_applicable"

        # === BEHAVIORAL ===
        # Behavior data is now always in long format with 'metric' column
        if task == "mixed":
            # H5: Use global_performance_z
            behav_subset = behavior_data[
                (behavior_data["subject_id"] == subject)
                & (behavior_data["metric"] == "global_performance_z")
            ]
        else:
            # Filter for specific task and metric
            behav_subset = behavior_data[
                (behavior_data["subject_id"] == subject)
                & (behavior_data["task"] == task)
                & (behavior_data["metric"] == behavioral_metric)
            ]

        # Restrict to baseline + post (ignore followup or extra rows) — same
        # rationale as for self-report above; a strict len == 2 check would
        # silently drop subjects with 3+ loaded timepoints.
        behav_subset = behav_subset[behav_subset["task_order"].isin([1, 2])]

        baseline_row = behav_subset[behav_subset["task_order"] == 1]
        post_row = behav_subset[behav_subset["task_order"] == 2]

        if not baseline_row.empty and not post_row.empty:
            base_val = baseline_row["value"].iloc[0]
            post_val = post_row["value"].iloc[0]

            result["behavior_baseline"] = base_val
            result["behavior_post"] = post_val
            result["delta_behavior"] = post_val - base_val

            # Compute baseline SD for RCI
            if task == "mixed":
                base_sd = behavior_data[
                    (behavior_data["task_order"] == 1)
                    & (behavior_data["metric"] == "global_performance_z")
                ]["value"].std()
            else:
                # Behavior data is in long format
                base_sd = behavior_data[
                    (behavior_data["task_order"] == 1)
                    & (behavior_data["task"] == task)
                    & (behavior_data["metric"] == behavioral_metric)
                ]["value"].std()

            # Behavioral reliability estimate
            reliability = 0.80  # Task performance test-retest

            rci, rci_class = compute_rci(base_val, post_val, base_sd, reliability)
            result["rci_behavior"] = rci
            result["rci_behavior_class"] = rci_class
        else:
            result["behavior_baseline"] = np.nan
            result["behavior_post"] = np.nan
            result["delta_behavior"] = np.nan
            result["rci_behavior"] = np.nan
            result["rci_behavior_class"] = "insufficient_data"

        # === CONVERGENCE INDEX ===
        # Gather baseline SDs for z-scoring
        baseline_sds = {}
        baseline_values = {}
        post_values = {}
        directions = {}

        # Self-report
        if "self_report_baseline" in result and not pd.isna(result["self_report_baseline"]):
            base_sd = assessment_data[
                (assessment_data["task_order"] == 1)
                & (assessment_data["metric"] == self_report_metric)
            ]["value"].std()

            if not pd.isna(base_sd) and base_sd > 0:
                baseline_sds["self_report"] = base_sd
                baseline_values["self_report"] = result["self_report_baseline"]
                post_values["self_report"] = result["self_report_post"]
                directions["self_report"] = hypothesis_config["improvement_direction"]["self_report"]

        # Brain
        if "brain_baseline" in result and not pd.isna(result["brain_baseline"]):
            if neural_contrast and neural_rois:
                task_roi_data = roi_data[roi_data["task"] == task]
                base_data = task_roi_data[
                    (task_roi_data["task_order"] == 1)
                    & (task_roi_data["contrast"] == neural_contrast)
                    & (task_roi_data["network_roi"].isin(neural_rois))
                ]
                base_agg = (
                    base_data.groupby("subject_id")["activation"].mean().reset_index()
                )
                base_sd = base_agg["activation"].std()

                if not pd.isna(base_sd) and base_sd > 0:
                    baseline_sds["brain"] = base_sd
                    baseline_values["brain"] = result["brain_baseline"]
                    post_values["brain"] = result["brain_post"]
                    # Neural direction: an a-priori assumption (None in
                    # config), never estimated from the observed change --
                    # set by the run's variant (efficiency=-1, activation=+1).
                    directions["brain"] = brain_direction

        # Behavior
        if "behavior_baseline" in result and not pd.isna(result["behavior_baseline"]):
            if task == "mixed":
                base_sd = behavior_data[
                    (behavior_data["task_order"] == 1)
                    & (behavior_data["metric"] == "global_performance_z")
                ]["value"].std()
            else:
                task_behav = behavior_data[behavior_data["task"] == task]
                base_sd = task_behav[
                    (task_behav["task_order"] == 1)
                    & (task_behav["metric"] == behavioral_metric)
                ]["value"].std()

            if not pd.isna(base_sd) and base_sd > 0:
                baseline_sds["behavior"] = base_sd
                baseline_values["behavior"] = result["behavior_baseline"]
                post_values["behavior"] = result["behavior_post"]
                directions["behavior"] = hypothesis_config["improvement_direction"]["behavioral"]

        # Compute CI
        n_variables = len(baseline_values)
        if n_variables >= 2:
            ci, ci_class, domain_directions = compute_convergence_index(
                baseline_values, post_values, baseline_sds, directions
            )
            result["convergence_index"] = ci
            result["convergence_class"] = ci_class
            result["n_domains"] = n_variables
        else:
            result["convergence_index"] = np.nan
            result["convergence_class"] = "insufficient_data"
            result["n_domains"] = n_variables

        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def generate_convergence_heatmap(
    results_df: pd.DataFrame, output_path: Path
) -> None:
    """Generate heatmap of Convergence Index values across subjects and hypotheses."""
    # Pivot for heatmap
    pivot_df = results_df.pivot_table(
        index="subject_id", columns="hypothesis", values="convergence_index"
    )

    # Sort subjects
    pivot_df = pivot_df.sort_index()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom colormap: -1 (convergent) = blue, 0 (mixed) = white, +1 (divergent) = red
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Convergence Index\n(-1: Convergent, +1: Divergent)"},
        ax=ax,
    )

    plt.title("Executive Function Convergence Patterns", fontsize=14, fontweight="bold")
    plt.xlabel("Hypothesis", fontsize=12)
    plt.ylabel("Subject", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_trajectory_plots(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate individual subject trajectory plots for each hypothesis."""
    for subject in results_df["subject_id"].unique():
        subject_results = results_df[results_df["subject_id"] == subject]

        # Create subplot grid for H1-H5
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (hyp_id, hyp_config) in enumerate(HYPOTHESES.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            hyp_results = subject_results[subject_results["hypothesis"] == hyp_id]

            if hyp_results.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{hyp_id}: {hyp_config['name']}")
                continue

            row = hyp_results.iloc[0]

            # Plot each domain
            domains = []
            values = []

            # Self-report
            if not pd.isna(row.get("self_report_baseline")):
                domains.append("Self-report")
                values.append([row["self_report_baseline"], row["self_report_post"]])

            # Brain
            if not pd.isna(row.get("brain_baseline")):
                domains.append("Brain")
                values.append([row["brain_baseline"], row["brain_post"]])

            # Behavior
            if not pd.isna(row.get("behavior_baseline")):
                domains.append("Behavior")
                values.append([row["behavior_baseline"], row["behavior_post"]])

            if not values:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{hyp_id}: {hyp_config['name']}")
                continue

            # Plot trajectories
            x = [1, 2]
            for domain, val_list in zip(domains, values):
                ax.plot(x, val_list, marker="o", label=domain, linewidth=2)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Baseline", "Post"])
            ax.set_xlabel("Timepoint")
            ax.set_ylabel("Value")
            ax.set_title(
                f"{hyp_id}: {hyp_config['name']}\n"
                f"CI: {row.get('convergence_index', np.nan):.2f} "
                f"({row.get('convergence_class', 'N/A')})",
                fontsize=10,
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty subplot
        if len(axes) > len(HYPOTHESES):
            fig.delaxes(axes[-1])

        plt.suptitle(f"Subject: {subject}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / f"subject_{subject}_trajectories.png", dpi=200, bbox_inches="tight")
        plt.close()


def generate_hypothesis_convergence_plots(
    results_df: pd.DataFrame, output_dir: Path
) -> None:
    """Generate per-hypothesis convergence visualization."""
    for hyp_id in HYPOTHESES.keys():
        hyp_data = results_df[results_df["hypothesis"] == hyp_id]
        hyp_name = HYPOTHESES[hyp_id]["name"]

        if hyp_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by CI
        hyp_data = hyp_data.sort_values("convergence_index")

        # Color by classification
        color_map = {
            "convergent_improvement": "#2E8B57",  # Sea green
            "partial_convergence": "#90EE90",  # Light green
            "mixed": "#FFD700",  # Gold
            "partial_divergence": "#FFA500",  # Orange
            "divergent": "#DC143C",  # Crimson
            "insufficient_data": "#D3D3D3",  # Light gray
        }

        colors = hyp_data["convergence_class"].map(color_map).fillna("#D3D3D3")

        # Plot bar chart
        ax.barh(
            hyp_data["subject_id"],
            hyp_data["convergence_index"],
            color=colors,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add reference lines
        ax.axvline(x=-0.67, color="green", linestyle="--", alpha=0.5, label="Convergent threshold")
        ax.axvline(x=0.67, color="red", linestyle="--", alpha=0.5, label="Divergent threshold")
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        ax.set_xlabel("Convergence Index", fontsize=12)
        ax.set_ylabel("Subject", fontsize=12)
        ax.set_title(f"{hyp_id}: {hyp_name} - Convergence Across Subjects", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"hypothesis_{hyp_id}_convergence.png", dpi=200, bbox_inches="tight")
        plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified EF Longitudinal Analysis - H1-H5 Convergence Testing"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "RNG seed for resampling. This stage is currently deterministic "
            "(the convergence index and RCI are closed-form), so the flag "
            "records provenance and keeps the pipeline's invocation uniform; "
            "any resampling added here must draw from it."
        ),
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

    print("=" * 70)
    print("UNIFIED EF LONGITUDINAL ANALYSIS")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"RNG seed: {args.seed}")
    print()

    # Create output directories
    unified_ef_dir = output_dir / "10_longitudinal" / "unified_ef"
    figures_dir = unified_ef_dir / "figures"
    unified_ef_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    variant = load_variant(output_dir)
    print(f"Analysis variant: {json.dumps(dataclasses.asdict(variant))}")
    brain_direction = {"efficiency": -1, "activation": 1}[variant.brain_direction]
    hyps = resolve_longitudinal_hypotheses(HYPOTHESES, variant)

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

    stroop_info = stroop_info[
        ~stroop_info["task_id"].isin(excluded_sessions("stroop", output_dir, base_dir))
    ]
    wcst_info = wcst_info[
        ~wcst_info["task_id"].isin(excluded_sessions("wcst", output_dir, base_dir))
    ]
    roster = sort_subjects(set(stroop_info["subject_id"]) | set(wcst_info["subject_id"]))
    print("Loading assessment T-scores...")
    assessment_data = load_assessment_t_scores(
        output_dir,
        roster,
        exclude_invalid=variant.exclude_invalid_brief,
    )
    t2 = choose_second_timepoints({"stroop": stroop_info, "wcst": wcst_info}, assessment_data)
    t2.to_csv(unified_ef_dir / "second_timepoint_selection.csv", index=False)
    print(f"  Saved: {unified_ef_dir / 'second_timepoint_selection.csv'}")
    stroop_info = apply_second_timepoint(stroop_info, t2)
    wcst_info = apply_second_timepoint(wcst_info, t2)
    assessment_data = apply_second_timepoint(assessment_data, t2)

    subjects = sort_subjects(set(t2["subject_id"]))
    if requested is not None:
        subjects = [s for s in requested if s in subjects]
        print(f"Using {len(subjects)} subjects from command line with a second timepoint: {subjects}")
    else:
        print(f"Selected {len(subjects)} subjects with a second timepoint: {subjects}")
    if not subjects:
        raise SystemExit("ERROR: No eligible subjects found")
    if not assessment_data.empty:
        assessment_data = assessment_data[assessment_data["subject_id"].isin(subjects)]
    print(f"  Loaded {len(assessment_data)} assessment records")
    print()

    print("Loading behavioral metrics...")
    stroop_behav = load_behavioral_metrics(output_dir, stroop_info, "stroop", subjects)
    wcst_behav = load_behavioral_metrics(output_dir, wcst_info, "wcst", subjects)
    print(f"  Loaded {len(stroop_behav)} Stroop behavioral records")
    print(f"  Loaded {len(wcst_behav)} WCST behavioral records")

    # Compute global performance z-score for H5
    print("Computing global performance z-score...")
    global_perf = compute_global_performance_z(stroop_behav, wcst_behav)
    print(f"  Computed {len(global_perf)} global performance z-scores")

    # Melt behavioral data to long format with 'metric' column
    behavior_list = []

    for task, task_behav in [("stroop", stroop_behav), ("wcst", wcst_behav)]:
        if task_behav.empty:
            continue

        # Get metric columns (exclude metadata)
        metadata_cols = ["subject_id", "task", "task_order", "timepoint", "source"]
        metric_cols = [col for col in task_behav.columns if col not in metadata_cols]

        # Melt to long format
        melted = task_behav.melt(
            id_vars=["subject_id", "task", "task_order", "timepoint"],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        )
        behavior_list.append(melted.dropna(subset=["value"]))

    # Add global performance z-score
    if not global_perf.empty:
        global_perf_long = global_perf.melt(
            id_vars=["subject_id", "task_order", "timepoint"],
            value_vars=["global_performance_z"],
            var_name="metric",
            value_name="value",
        )
        global_perf_long["task"] = "mixed"  # Special marker for H5
        behavior_list.append(global_perf_long.dropna(subset=["value"]))

    if behavior_list:
        behavior_data = pd.concat(behavior_list, ignore_index=True)
    else:
        behavior_data = pd.DataFrame()
        print("  WARNING: No behavioral data loaded")

    print("Loading ROI activations...")
    roi_list = []
    for task, info in [("stroop", stroop_info), ("wcst", wcst_info)]:
        for hyp_id, hyp_config in hyps.items():
            if hyp_config["task"] != task or not hyp_config["neural_contrast"]:
                continue

            roi_df = load_roi_activations(
                output_dir,
                info,
                task,
                subjects,
                hyp_config["neural_roi"],
                hyp_config["neural_contrast"],
            )
            if not roi_df.empty:
                roi_list.append(roi_df)

    if roi_list:
        roi_data = pd.concat(roi_list, ignore_index=True)
        print(f"  Loaded {len(roi_data)} ROI activation records")
    else:
        roi_data = pd.DataFrame()
        print("  WARNING: No ROI data loaded")

    print()

    # Run hypothesis analyses
    print("Running hypothesis convergence analyses...")
    all_results = []

    for hyp_id, hyp_config in hyps.items():
        print(f"  Analyzing {hyp_id}: {hyp_config['name']}...")
        hyp_results = run_hypothesis_analysis(
            hyp_id, hyp_config, assessment_data, behavior_data, roi_data, subjects,
            brain_direction,
        )
        all_results.append(hyp_results)

    results_df = pd.concat(all_results, ignore_index=True)
    print(f"  Generated {len(results_df)} subject-hypothesis results")

    print()

    # Save outputs
    print("Saving outputs...")
    results_with_variant = results_df.copy()
    results_with_variant["variant"] = variant.name
    results_with_variant.to_csv(unified_ef_dir / "hypothesis_convergence.csv", index=False)
    print(f"  Saved: {unified_ef_dir / 'hypothesis_convergence.csv'}")

    # Generate summary CSVs
    # Subject changes (wide format) - numeric values only
    numeric_cols = [
        "delta_self_report",
        "delta_brain",
        "delta_behavior",
        "convergence_index",
    ]
    subject_changes = results_df.pivot_table(
        index="subject_id",
        columns="hypothesis",
        values=numeric_cols,
        aggfunc="first",  # Take first value if multiple
    )
    subject_changes.to_csv(unified_ef_dir / "subject_changes.csv")
    print(f"  Saved: {unified_ef_dir / 'subject_changes.csv'}")

    # Save classifications separately
    convergence_classifications = results_df.pivot_table(
        index="subject_id",
        columns="hypothesis",
        values="convergence_class",
        aggfunc="first",
    )
    convergence_classifications.to_csv(unified_ef_dir / "subject_convergence_classifications.csv")
    print(f"  Saved: {unified_ef_dir / 'subject_convergence_classifications.csv'}")

    # Convergence summary
    convergence_summary = (
        results_df.groupby("hypothesis")["convergence_class"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )
    convergence_summary.to_csv(unified_ef_dir / "convergence_summary.csv")
    print(f"  Saved: {unified_ef_dir / 'convergence_summary.csv'}")

    # Generate figures
    print()
    print("Generating figures...")
    generate_convergence_heatmap(results_df, figures_dir / "convergence_heatmap.png")
    print(f"  Saved: {figures_dir / 'convergence_heatmap.png'}")

    generate_trajectory_plots(results_df, figures_dir)
    print(f"  Saved: Individual subject trajectory plots")

    generate_hypothesis_convergence_plots(results_df, figures_dir)
    print(f"  Saved: Per-hypothesis convergence plots")

    # Generate JSON summary
    print()
    print("Generating JSON summary...")
    summary = {
        "analysis_type": "unified_ef_longitudinal",
        "n_subjects": len(subjects),
        "subjects": subjects,
        "hypotheses": list(hyps.keys()),
        "outputs": {
            "hypothesis_convergence": str(
                (unified_ef_dir / "hypothesis_convergence.csv").relative_to(output_dir)
            ),
            "subject_changes": str(
                (unified_ef_dir / "subject_changes.csv").relative_to(output_dir)
            ),
            "convergence_summary": str(
                (unified_ef_dir / "convergence_summary.csv").relative_to(output_dir)
            ),
            "figures": {
                "convergence_heatmap": str(
                    (figures_dir / "convergence_heatmap.png").relative_to(output_dir)
                ),
                "subject_trajectories": str(
                    (figures_dir / "subject_*_trajectories.png").relative_to(output_dir)
                ),
                "hypothesis_convergence": str(
                    (figures_dir / "hypothesis_*_convergence.png").relative_to(output_dir)
                ),
            },
        },
        "convergence_patterns": {},
    }

    # Add convergence patterns to summary
    for hyp_id in hyps.keys():
        hyp_data = results_df[results_df["hypothesis"] == hyp_id]
        if not hyp_data.empty:
            class_counts = hyp_data["convergence_class"].value_counts().to_dict()
            summary["convergence_patterns"][hyp_id] = class_counts

    with (unified_ef_dir / "analysis_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {unified_ef_dir / 'analysis_summary.json'}")

    # Print summary
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Convergence Summary:")
    print("-" * 70)

    for hyp_id in hyps.keys():
        hyp_data = results_df[results_df["hypothesis"] == hyp_id]
        if hyp_data.empty:
            continue

        print(f"\n{hyp_id}: {hyps[hyp_id]['name']}")
        print(f"  N = {len(hyp_data)}")

        class_counts = hyp_data["convergence_class"].value_counts()
        for cls, count in class_counts.items():
            pct = (count / len(hyp_data)) * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")

        # Mean CI
        mean_ci = hyp_data["convergence_index"].mean()
        print(f"  Mean CI: {mean_ci:.2f}")

    print()
    print("=" * 70)
    print(f"All outputs saved to: {unified_ef_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
