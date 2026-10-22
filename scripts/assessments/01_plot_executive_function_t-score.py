#!/usr/bin/env python3
"""Generate BRIEF-A t-score visualizations across baseline, post, follow-up."""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subject_ids import sort_subjects, standardize_subject_id  # noqa: E402
from brief_validity import valid_administration_mask  # noqa: E402
from analysis_variant import refuse_protected_tree  # noqa: E402

SUBSCALE_COLUMNS: Sequence[str] = (
    "Inhibit_t",
    "Shift_t",
    "Emotional Control_t",
    "Self-Monitor_t",
    "Initiate_t",
    "Working Memory_t",
    "Plan/Organize_t",
    "Task Monitor_t",
    "Org Materials_t",
    "BRI_t",
    "MI_t",
    "GEC_t",
)
SUBSCALE_LABELS: Tuple[str, ...] = tuple(
    column[:-2] if column.endswith("_t") else column for column in SUBSCALE_COLUMNS
)
SUBSCALE_LIST: List[str] = list(SUBSCALE_COLUMNS)
TIMEPOINT_FILE_MAP: Dict[str, str] = {
    "baseline": "executive_function_baseline_subscale_t-scores.csv",
    "post": "executive_function_post_subscale_t-scores.csv",
    "followup": "executive_function_followup_subscale_t-scores.csv",
}
COLOR_MAP: Dict[str, str] = {
    "baseline": "#1E88E5",
    "post": "#FFC107",
    "followup": "#004D40",
}
DEFAULT_FIGURE_DPI = 300



def format_subject_label(subject: str) -> str:
    """Return the display form of a subject identifier.

    Canonical codes are already display-ready; this exists so callers have
    one place to change if that stops being true.
    """
    return standardize_subject_id(subject)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Create BRIEF-A t-score summary plots across study timepoints."
    )
    parser.add_argument(
        "--t-score-dir",
        type=str,
        default=None,
        help=(
            "Directory containing timepoint CSVs (baseline/post/followup). "
            "Defaults to ${OUTPUT_DIR}/assessments/t-scores if OUTPUT_DIR is set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Destination directory for generated figures. Defaults to "
            "${OUTPUT_DIR}/assessments/figures/t_scores if OUTPUT_DIR is set, "
            "otherwise ./logs/t_score_plots."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_FIGURE_DPI,
        help="Figure resolution (dots per inch).",
    )
    return parser.parse_args()


def resolve_t_score_dir(cli_value: Optional[str]) -> Path:
    """Resolve the directory with t-score CSV files."""

    if cli_value:
        return Path(cli_value).expanduser().resolve()

    output_dir_env = os.getenv("OUTPUT_DIR")
    if output_dir_env:
        return (Path(output_dir_env).expanduser() / "01_assessments" / "t-scores").resolve()

    raise ValueError(
        "Path to t-score CSVs is required. Supply --t-score-dir or set OUTPUT_DIR."
    )


def resolve_output_dir(cli_value: Optional[str]) -> Path:
    """Resolve the directory where figures will be stored."""

    if cli_value:
        return Path(cli_value).expanduser().resolve()

    output_dir_env = os.getenv("OUTPUT_DIR")
    if output_dir_env:
        return (
            Path(output_dir_env).expanduser()
            / "01_assessments"
            / "figures"
        ).resolve()

    return (Path.cwd() / "logs" / "t_score_plots").resolve()


def load_timepoint_scores(timepoint: str, source_dir: Path) -> pd.DataFrame:
    """Load a single timepoint CSV and retain valid BRIEF-A t-score columns."""

    file_path = source_dir / TIMEPOINT_FILE_MAP[timepoint]
    if not file_path.exists():
        raise FileNotFoundError(f"Missing expected CSV for {timepoint}: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = [column.strip() for column in df.columns]

    numeric_columns = SUBSCALE_LIST + ["total_score", "n_items"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Not administered (no items), or failing a validity scale: the analyses
    # leave both out, so the figures do too.
    for column in ("total_score", "n_items"):
        if column not in df.columns:
            raise KeyError(f"{file_path} has no {column!r} column")
    not_administered = (df["total_score"].fillna(0) <= 0) | (df["n_items"].fillna(0) <= 0)
    invalid_mask = not_administered | ~valid_administration_mask(df)
    df.loc[invalid_mask, SUBSCALE_LIST] = pd.NA

    filtered = (
        df[["subject", *SUBSCALE_LIST]]
        .dropna(subset=SUBSCALE_LIST, how="any")
        .assign(subject=lambda frame: frame["subject"].astype(str).str.strip())
    )

    return filtered


def ensure_output_directory(path: Path) -> None:
    """Create the destination directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def plot_baseline_individuals(
    df: pd.DataFrame,
    group_means: pd.Series,
    output_path: Path,
    dpi: int,
) -> None:
    """Create a grid of baseline t-scores with group means indicated."""

    subjects = df["subject"].tolist()
    if not subjects:
        raise ValueError("No baseline participants available for plotting.")

    n_cols = min(4, max(1, len(subjects)))
    n_rows = max(1, -(-len(subjects) // n_cols))
    x_positions = range(len(SUBSCALE_LIST))
    subject_max = float(df[SUBSCALE_LIST].max().max()) if not df.empty else 0
    mean_max = float(group_means.max()) if not group_means.empty else 0
    shared_ylim = max(90, max(subject_max, mean_max) + 10)
    mean_series = group_means.reindex(SUBSCALE_LIST)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 3.5),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle("Baseline BRIEF-A by Participant", fontsize=14)

    for ax, subject in zip(axes.flatten(), subjects):
        scores = df.loc[df["subject"] == subject, SUBSCALE_LIST].iloc[0]
        ax.bar(x_positions, scores.values, color=COLOR_MAP["baseline"], width=0.6)
        for idx, column in enumerate(SUBSCALE_LIST):
            mean_value = mean_series[column]
            if pd.notna(mean_value):
                ax.hlines(
                    mean_value,
                    idx - 0.3,
                    idx + 0.3,
                    colors="#D81B60",
                    linestyles="-",
                    linewidth=1.2,
                    alpha=0.9,
                )
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(SUBSCALE_LABELS, rotation=90, ha="center")
        ax.set_ylim(0, shared_ylim)
        ax.set_title(format_subject_label(subject), fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for ax in axes.flatten()[len(subjects):]:
        ax.axis("off")

    axes[0][0].set_ylabel("T-Score")
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_baseline_summary(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot the baseline group mean t-scores with individual scatter."""

    mean_scores = df[SUBSCALE_LIST].mean()
    x_positions = range(len(SUBSCALE_LIST))
    y_limit = max(90, float(mean_scores.max()) + 10)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(
        x_positions,
        mean_scores.values,
        color=COLOR_MAP["baseline"],
        alpha=0.8,
        width=0.8,
        edgecolor=COLOR_MAP["baseline"],
        linewidth=1.2,
    )

    rng = np.random.default_rng(seed=314159)
    for idx, column in enumerate(SUBSCALE_LIST):
        values = df[column].dropna().values
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            idx + jitter,
            values,
            color="#D81B60",
            s=24,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(SUBSCALE_LABELS, rotation=90, ha="center")
    ax.set_ylabel("T-Score")
    ax.set_ylim(0, y_limit)
    ax.set_title("Baseline BRIEF-A (Group Mean)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def align_subject_scores(datasets: Dict[str, pd.DataFrame]) -> List[Dict[str, pd.Series]]:
    """Return per-subject score vectors for the shared participants."""

    if not datasets:
        return []

    subject_sets = [set(df["subject"].tolist()) for df in datasets.values()]
    # Numeric order: this drives subplot and panel order, and sorted() on
    # unpadded codes places S10 before S2.
    try:
        shared_subjects = (
            sort_subjects(set.intersection(*subject_sets)) if subject_sets else []
        )
    except ValueError as exc:
        # format_subject_label tolerates S01/s1/S-1 on the way in, so a
        # loosely spelled subject column reaches here. Report it rather than
        # dumping a traceback; the validation itself stays strict.
        raise SystemExit(
            f"ERROR: {exc}. The subject column must hold canonical S-codes "
            "-- the letter S followed by an unpadded number, e.g. S1, S12."
        ) from exc

    aligned: List[Dict[str, pd.Series]] = []
    for subject in shared_subjects:
        entry = {"subject": subject}
        valid = True
        for timepoint, df in datasets.items():
            subset = df.loc[df["subject"] == subject, SUBSCALE_LIST]
            if subset.empty:
                valid = False
                break
            entry[timepoint] = subset.iloc[0]
        if valid:
            aligned.append(entry)
    return aligned


def plot_timepoint_trajectories(
    aligned_scores: List[Dict[str, pd.Series]],
    timepoints: Sequence[str],
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    """Plot trajectories across specified timepoints for each participant."""

    if not aligned_scores:
        print(f"No participants available for {title.lower()}. Skipping plot.")
        return

    # Limit to two panels per row so four participants render as a 2x2 grid.
    n_cols = min(2, max(1, len(aligned_scores)))
    n_rows = max(1, -(-len(aligned_scores) // n_cols))
    x_positions = range(len(SUBSCALE_LIST))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.2, n_rows * 3.6),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14)

    for ax, subject_entry in zip(axes.flatten(), aligned_scores):
        subject = subject_entry["subject"]
        y_limit = max(
            90,
            max(float(subject_entry[tp].max()) for tp in timepoints) + 10,
        )
        for timepoint in timepoints:
            scores = subject_entry[timepoint]
            ax.plot(
                x_positions,
                scores.values,
                marker="o",
                linewidth=2,
                label=timepoint.capitalize(),
                color=COLOR_MAP[timepoint],
            )
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(SUBSCALE_LABELS, rotation=90, ha="center")
        ax.set_ylim(0, y_limit)
        ax.set_title(format_subject_label(subject), fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for ax in axes.flatten()[len(aligned_scores):]:
        ax.axis("off")

    axes[0][0].set_ylabel("T-Score")

    handles = [
        Line2D([0], [0], color=COLOR_MAP[tp], marker="o", label=tp.capitalize())
        for tp in timepoints
    ]
    fig.legend(
        handles,
        [tp.capitalize() for tp in timepoints],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(timepoints),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    """Entry point for the plotting workflow."""

    load_dotenv()
    args = parse_args()

    try:
        t_score_dir = resolve_t_score_dir(args.t_score_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = resolve_output_dir(args.output_dir)
    try:
        refuse_protected_tree(str(output_dir))
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    ensure_output_directory(output_dir)

    baseline_df = load_timepoint_scores("baseline", t_score_dir)
    post_df = load_timepoint_scores("post", t_score_dir)
    followup_df = load_timepoint_scores("followup", t_score_dir)

    baseline_means = baseline_df[SUBSCALE_LIST].mean() if not baseline_df.empty else pd.Series(dtype=float)

    baseline_grid_path = output_dir / "brief-a_baseline_individual_grid_t-score.png"
    baseline_summary_path = output_dir / "brief-a_baseline_mean_summary_t-score.png"
    post_trajectory_path = output_dir / "brief-a_baseline_post_trajectories_t-score.png"
    followup_trajectory_path = output_dir / "brief-a_baseline_post_followup_trajectories_t-score.png"

    plot_baseline_individuals(
        baseline_df, baseline_means, baseline_grid_path, dpi=args.dpi
    )
    plot_baseline_summary(baseline_df, baseline_summary_path, dpi=args.dpi)

    post_aligned = align_subject_scores({"baseline": baseline_df, "post": post_df})
    plot_timepoint_trajectories(
        post_aligned,
        timepoints=("baseline", "post"),
        output_path=post_trajectory_path,
        dpi=args.dpi,
        title="Baseline vs. Post BRIEF-A Trajectories",
    )

    followup_aligned = align_subject_scores(
        {"baseline": baseline_df, "post": post_df, "followup": followup_df}
    )
    plot_timepoint_trajectories(
        followup_aligned,
        timepoints=("baseline", "post", "followup"),
        output_path=followup_trajectory_path,
        dpi=args.dpi,
        title="Baseline, Post, and Follow-up BRIEF-A Trajectories",
    )

    print("Saved figures:")
    for path in (
        baseline_grid_path,
        baseline_summary_path,
        post_trajectory_path,
        followup_trajectory_path,
    ):
        if path.exists():
            print(f" - {path}")
        else:
            print(f" - {path} (not generated)")


if __name__ == "__main__":
    main()

# the t-score csv files are manually corrected after running 01_analyze_executive_function.py, 
# "inconsistency_score" has been corrected in 01_analyze_executive_function.py but not in the t-score files. 
# so we don't regenerate the t-score files here and only use "inconsistency_score" from the corrected raw score files.