#!/usr/bin/env python3
"""
04_analyze_craving.py

Analyze craving assessment data:
- Calculate total craving scores: the sum of the six 0-4 ratings, with the
  number of ratings answered; substance codes and occurrence counts are
  separate columns
- Calculate the standard 3-item BSCS total (primary substance, 0-12;
  `bscs_total`, NaN unless all three primary ratings are answered, with
  `bscs_n_items`) alongside the six-rating total
- Create enhanced visualizations
- Individual participant profiles

"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dotenv import load_dotenv
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subject_ids import standardize_subject_id, subject_column  # noqa: E402
from analysis_variant import refuse_protected_tree  # noqa: E402
from natsort import natsorted
from typing import Dict, List, Sequence

# Load environment variables
load_dotenv()

# Define paths
BASE_DIR = os.getenv("BASE_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

if not BASE_DIR or not OUTPUT_DIR:
    print("Error: BASE_DIR and OUTPUT_DIR environment variables must be set")
    print("Please source the .env file or set the environment variables")
    exit(1)

SCORES_DIR = Path(OUTPUT_DIR) / "01_assessments" / "scores"
FIGURES_DIR = Path(OUTPUT_DIR) / "01_assessments" / "figures"


DEFAULT_FIGURE_DPI = 300
COLOR_MAP = {
    'baseline': '#1E88E5',
    'mid': '#6A1B9A',
    'post': '#FFC107',
    'followup': '#004D40',
}
MEAN_LINE_COLOR = '#D81B60'


def format_subject_label(subject: str) -> str:
    """Return the display form of a subject identifier.

    Canonical codes are already display-ready; this exists so callers have
    one place to change if that stops being true.
    """
    return standardize_subject_id(subject)

def load_craving_data():
    """Load preprocessed craving data."""
    filepath = SCORES_DIR / 'craving_processed.json'
    
    if not filepath.exists():
        print("Error: Preprocessed data not found. Run 00_load_assessment_data.py first.")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

# The sheet's ten rows are three kinds of answer. Only the 0-4 ratings
# (intensity, frequency and length of craving, for the primary and the
# second substance) are summed into the craving total. The
# substance-type codes are categories and the occurrence counts are
# unbounded, so they are reported as their own columns, never added in.
LIKERT_QUESTION_PREFIXES = ("INTENSITY", "FREQUENCY", "LENGTH")
SEPARATE_ITEMS = {
    "Primary Substance Type": "primary_substance_type",
    "Second Substance Type": "second_substance_type",
    "NUMBER of times craving occurred": "craving_occurrences_primary",
    "NUMBER of times craving second substance occurred": "craving_occurrences_second",
}
N_LIKERT_ITEMS = 6
N_BSCS_ITEMS = 3


def is_second_substance_item(question: str) -> bool:
    """True for the second-substance rating rows (their text names the second substance)."""
    return "second" in str(question).lower()


def classify_craving_item(question: str) -> str:
    """Return 'likert' or the output column of a separately reported item; raise otherwise."""
    text = str(question).strip()
    if text.upper().startswith(LIKERT_QUESTION_PREFIXES):
        return "likert"
    for prefix, column in SEPARATE_ITEMS.items():
        if text.startswith(prefix):
            return column
    raise ValueError(f"Unrecognised craving item: {question!r}")


def calculate_craving_scores(subject_data, timepoint):
    """Calculate the craving total (0-4 ratings only) and item-level scores."""
    if timepoint not in subject_data:
        return None
    
    items = subject_data[timepoint]
    df = pd.DataFrame(items)
    
    # The column named by the subject's code holds the responses
    subject_col = subject_column(df.columns)
    
    # Convert to numeric
    df[subject_col] = pd.to_numeric(df[subject_col], errors='coerce')

    kinds = df['Question'].map(classify_craving_item)
    likert = df.loc[kinds == 'likert', subject_col]
    if len(likert) != N_LIKERT_ITEMS:
        raise ValueError(
            f"Expected {N_LIKERT_ITEMS} craving rating items, found {len(likert)}")
    
    # The total is the sum of the answered ratings; n_items records how many
    # of the six it is based on (a second-substance block is blank when there
    # is no second substance).
    n_items = int(likert.notna().sum())
    total_score = likert.sum() if n_items else np.nan

    # Standard 3-item Brief Substance Craving Scale total: the sum of the
    # primary substance's intensity, frequency
    # and length ratings (each 0-4, total 0-12). Second-substance ratings are
    # excluded. Missing-item rule: the total is NaN unless all three primary
    # ratings are answered (no proration); bscs_n_items records how many of
    # the three were answered.
    primary = df.loc[(kinds == 'likert') & ~df['Question'].map(is_second_substance_item),
                     subject_col]
    if len(primary) != N_BSCS_ITEMS:
        raise ValueError(
            f"Expected {N_BSCS_ITEMS} primary-substance craving ratings, found {len(primary)}")
    bscs_n_items = int(primary.notna().sum())
    bscs_total = float(primary.sum()) if bscs_n_items == N_BSCS_ITEMS else np.nan

    separate = {
        column: (df.loc[kinds == column, subject_col].iloc[0]
                 if (kinds == column).any() else np.nan)
        for column in SEPARATE_ITEMS.values()
    }
    
    # Get item-level scores
    item_scores = {}
    for idx, row in df.iterrows():
        item_num = row['Item_Number']
        item_scores[f'item_{item_num}'] = row[subject_col]
    
    return {
        'total_score': total_score,
        'n_items': n_items,
        'bscs_total': bscs_total,
        'bscs_n_items': bscs_n_items,
        'separate': separate,
        'item_scores': item_scores
    }

def analyze_all_subjects(data):
    """Analyze craving data for all subjects."""
    results = []
    
    for subject, timepoints in data.items():
        for timepoint in ['baseline', 'mid', 'post', 'followup']:
            scores = calculate_craving_scores(timepoints, timepoint)
            if scores:
                result = {
                    'subject': subject,
                    'timepoint': timepoint,
                    'total_score': scores['total_score'],
                    'n_items': scores['n_items'],
                    **scores['separate'],
                }
                
                # Add individual item scores
                for item_key, item_score in scores['item_scores'].items():
                    result[item_key] = item_score
                
                # BSCS columns are appended after the item columns so the
                # pre-existing column order is untouched.
                result['bscs_total'] = scores['bscs_total']
                result['bscs_n_items'] = scores['bscs_n_items']

                results.append(result)
    
    return pd.DataFrame(results)

def get_item_columns(df: pd.DataFrame) -> List[str]:
    """Return ordered craving item columns present in the dataframe."""

    item_cols = [
        column
        for column in df.columns
        if column.startswith('item_') and column.split('_')[1].isdigit()
    ]
    item_cols.sort(key=lambda col: int(col.split('_')[1]))
    return item_cols


def _resolve_item_labels(columns: Sequence[str]) -> List[str]:
    """Return concise labels for craving item columns."""

    labels: List[str] = []
    for column in columns:
        suffix = column.split('_')[1]
        labels.append(f'Q{int(suffix)}' if suffix.isdigit() else column)
    return labels


def plot_baseline_individuals(
    df: pd.DataFrame,
    group_means: pd.Series,
    item_cols: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Create a grid of baseline craving item scores with group means indicated."""

    subjects = natsorted(df['subject'].dropna().astype(str).unique())
    if not subjects or not item_cols:
        return

    n_cols = min(4, max(1, len(subjects)))
    n_rows = int(np.ceil(len(subjects) / n_cols))
    x_positions = range(len(item_cols))
    item_labels = _resolve_item_labels(item_cols)
    subject_max = df[item_cols].max().max()
    mean_max = group_means.max() if not group_means.empty else np.nan
    maxima = [value for value in [subject_max, mean_max] if pd.notna(value)]
    shared_upper = max(maxima) if maxima else 0.0
    y_limit = max(5.0, shared_upper * 1.1 if shared_upper else 5.0)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.0, n_rows * 3.5),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle('Baseline Craving Item Scores by Participant', fontsize=14)
    flat_axes = axes.flatten()

    for ax, subject in zip(flat_axes, subjects):
        subject_row = df[df['subject'] == subject]
        if subject_row.empty:
            ax.axis('off')
            continue
        scores = subject_row.iloc[0][item_cols].apply(pd.to_numeric, errors='coerce')
        scores = scores.reindex(item_cols)
        if scores.isna().all():
            ax.text(
                0.5,
                0.5,
                'No Data',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=11,
                color='gray',
            )
        else:
            for idx, column in enumerate(item_cols):
                value = scores[column]
                if pd.notna(value):
                    ax.bar(
                        [idx],
                        [value],
                        color=COLOR_MAP['baseline'],
                        width=0.6,
                    )
                mean_value = group_means.get(column)
                if pd.notna(mean_value):
                    ax.hlines(
                        mean_value,
                        idx - 0.3,
                        idx + 0.3,
                        colors=MEAN_LINE_COLOR,
                        linestyles='-',
                        linewidth=1.2,
                        alpha=0.9,
                    )
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(item_labels, rotation=90, ha='center')
        ax.set_ylim(0, y_limit)
        ax.set_title(format_subject_label(subject), fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    for ax in flat_axes[len(subjects):]:
        ax.axis('off')

    flat_axes[0].set_ylabel('Item Score')
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_baseline_summary(
    df: pd.DataFrame,
    item_cols: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot baseline group mean item scores with individual scatter."""

    if not item_cols:
        return

    numeric_df = df.copy()
    numeric_df[item_cols] = numeric_df[item_cols].apply(pd.to_numeric, errors='coerce')
    mean_scores = numeric_df[item_cols].mean()
    item_labels = _resolve_item_labels(item_cols)
    x_positions = range(len(item_cols))
    max_candidate = mean_scores.max()
    upper_bound = max(5.0, float(max_candidate) * 1.1 if pd.notna(max_candidate) else 5.0)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.bar(
        x_positions,
        mean_scores.values,
        color=COLOR_MAP['baseline'],
        alpha=0.8,
        width=0.8,
        edgecolor=COLOR_MAP['baseline'],
        linewidth=1.2,
    )

    rng = np.random.default_rng(seed=161803)
    for idx, column in enumerate(item_cols):
        values = numeric_df[column].dropna().values
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            idx + jitter,
            values,
            color=MEAN_LINE_COLOR,
            s=28,
            alpha=0.75,
            edgecolors='white',
            linewidths=0.5,
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(item_labels, rotation=90, ha='center')
    ax.set_ylabel('Item Score')
    ax.set_ylim(0, upper_bound)
    ax.set_title('Baseline Craving (Group Mean)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def align_subject_item_scores(
    datasets: Dict[str, pd.DataFrame],
    item_cols: Sequence[str],
) -> List[Dict[str, pd.Series]]:
    """Return per-subject item vectors for subjects present across datasets."""

    if not datasets or not item_cols:
        return []

    subject_sets: List[set] = []
    for df_dataset in datasets.values():
        if df_dataset.empty:
            return []
        subject_sets.append(set(df_dataset['subject'].dropna().astype(str)))

    if not subject_sets:
        return []

    shared_subjects = natsorted(set.intersection(*subject_sets)) if subject_sets else []
    aligned: List[Dict[str, pd.Series]] = []

    for subject in shared_subjects:
        entry: Dict[str, pd.Series] = {'subject': subject}
        valid = True
        for timepoint, df_tp in datasets.items():
            subject_rows = df_tp[df_tp['subject'] == subject]
            if subject_rows.empty:
                valid = False
                break
            series = subject_rows.iloc[0][item_cols].apply(pd.to_numeric, errors='coerce')
            series = series.reindex(item_cols)
            if series.isna().all():
                valid = False
                break
            entry[timepoint] = series
        if valid:
            aligned.append(entry)

    return aligned


def plot_timepoint_item_trajectories(
    aligned_scores: List[Dict[str, pd.Series]],
    item_cols: Sequence[str],
    timepoints: Sequence[str],
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    """Plot trajectories across specified timepoints for each participant."""

    if not aligned_scores:
        print(f"No participants available for {title.lower()}. Skipping plot.")
        return

    n_cols = min(2, max(1, len(aligned_scores)))
    n_rows = int(np.ceil(len(aligned_scores) / n_cols))
    x_positions = range(len(item_cols))
    item_labels = _resolve_item_labels(item_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.2, n_rows * 3.6),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14)
    flat_axes = axes.flatten()

    for ax, subject_entry in zip(flat_axes, aligned_scores):
        subject = subject_entry['subject']
        max_values = []
        for tp in timepoints:
            series = subject_entry[tp]
            max_values.append(series.max())
            ax.plot(
                x_positions,
                series.values,
                marker='o',
                linewidth=2,
                label=tp.capitalize(),
                color=COLOR_MAP.get(tp, '#424242'),
            )
        valid_max = [value for value in max_values if not pd.isna(value)]
        upper_bound = max(5.0, max(valid_max) * 1.1 if valid_max else 5.0)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(item_labels, rotation=90, ha='center')
        ax.set_ylim(0, upper_bound)
        ax.set_title(format_subject_label(subject), fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    for ax in flat_axes[len(aligned_scores):]:
        ax.axis('off')

    flat_axes[0].set_ylabel('Item Score')

    handles = [
        Line2D([0], [0], color=COLOR_MAP.get(tp, '#424242'), marker='o', label=tp.capitalize())
        for tp in timepoints
    ]
    fig.legend(
        handles,
        [tp.capitalize() for tp in timepoints],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(timepoints),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def create_visualizations(df: pd.DataFrame) -> None:
    """Create craving figures following the harmonized plotting logic."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    item_cols = get_item_columns(df)
    if not item_cols:
        print('No craving item columns available for plotting. Skipping visualization.')
        return

    baseline_df = df[df['timepoint'] == 'baseline'].copy()
    if baseline_df.empty:
        print('No baseline craving data available. Skipping visualization.')
        return

    baseline_df[item_cols] = baseline_df[item_cols].apply(pd.to_numeric, errors='coerce')
    baseline_means = baseline_df[item_cols].mean()

    baseline_grid_path = FIGURES_DIR / 'craving_baseline_individual_grid.png'
    baseline_summary_path = FIGURES_DIR / 'craving_baseline_mean_summary.png'

    plot_baseline_individuals(
        baseline_df,
        baseline_means,
        item_cols,
        baseline_grid_path,
        dpi=DEFAULT_FIGURE_DPI,
    )
    plot_baseline_summary(
        baseline_df,
        item_cols,
        baseline_summary_path,
        dpi=DEFAULT_FIGURE_DPI,
    )

    saved_paths = [baseline_grid_path, baseline_summary_path]

    mid_df = df[df['timepoint'] == 'mid'].copy()
    post_df = df[df['timepoint'] == 'post'].copy()
    followup_df = df[df['timepoint'] == 'followup'].copy()

    for subset in (mid_df, post_df, followup_df):
        if not subset.empty:
            subset[item_cols] = subset[item_cols].apply(pd.to_numeric, errors='coerce')

    if not mid_df.empty:
        mid_aligned = align_subject_item_scores({
            'baseline': baseline_df,
            'mid': mid_df,
        }, item_cols)
        mid_path = FIGURES_DIR / 'craving_baseline_mid_trajectories.png'
        if mid_aligned:
            plot_timepoint_item_trajectories(
                mid_aligned,
                item_cols,
                timepoints=('baseline', 'mid'),
                output_path=mid_path,
                dpi=DEFAULT_FIGURE_DPI,
                title='Baseline vs. Mid Craving Item Trajectories',
            )
            saved_paths.append(mid_path)

    if not post_df.empty:
        post_aligned = align_subject_item_scores({
            'baseline': baseline_df,
            'post': post_df,
        }, item_cols)
        post_path = FIGURES_DIR / 'craving_baseline_post_trajectories.png'
        if post_aligned:
            plot_timepoint_item_trajectories(
                post_aligned,
                item_cols,
                timepoints=('baseline', 'post'),
                output_path=post_path,
                dpi=DEFAULT_FIGURE_DPI,
                title='Baseline vs. Post Craving Item Trajectories',
            )
            saved_paths.append(post_path)

    if not followup_df.empty:
        followup_datasets: Dict[str, pd.DataFrame] = {'baseline': baseline_df}
        sequence: List[str] = ['baseline']
        if not mid_df.empty:
            followup_datasets['mid'] = mid_df
            sequence.append('mid')
        if not post_df.empty:
            followup_datasets['post'] = post_df
            sequence.append('post')
        followup_datasets['followup'] = followup_df
        sequence.append('followup')
        followup_aligned = align_subject_item_scores(followup_datasets, item_cols)
        if followup_aligned:
            if 'mid' in followup_datasets and 'post' in followup_datasets:
                followup_path = FIGURES_DIR / 'craving_baseline_mid_post_followup_trajectories.png'
                followup_timepoints: Sequence[str] = ('baseline', 'mid', 'post', 'followup')
                followup_title = 'Baseline, Mid, Post, and Follow-up Craving Item Trajectories'
            elif 'post' in followup_datasets:
                followup_path = FIGURES_DIR / 'craving_baseline_post_followup_trajectories.png'
                followup_timepoints = ('baseline', 'post', 'followup')
                followup_title = 'Baseline, Post, and Follow-up Craving Item Trajectories'
            elif 'mid' in followup_datasets:
                followup_path = FIGURES_DIR / 'craving_baseline_mid_followup_trajectories.png'
                followup_timepoints = ('baseline', 'mid', 'followup')
                followup_title = 'Baseline, Mid, and Follow-up Craving Item Trajectories'
            else:
                followup_path = FIGURES_DIR / 'craving_baseline_followup_trajectories.png'
                followup_timepoints = ('baseline', 'followup')
                followup_title = 'Baseline vs. Follow-up Craving Item Trajectories'

            plot_timepoint_item_trajectories(
                followup_aligned,
                item_cols,
                timepoints=followup_timepoints,
                output_path=followup_path,
                dpi=DEFAULT_FIGURE_DPI,
                title=followup_title,
            )
            saved_paths.append(followup_path)

    existing_paths = [path for path in saved_paths if path.exists()]
    if existing_paths:
        print('Saved figures:')
        for path in existing_paths:
            print(f' - {path}')


def generate_summary_statistics(df):
    """Generate summary statistics for craving data."""
    summary = {}
    
    for timepoint in ['baseline', 'mid', 'post', 'followup']:
        # Only administrations with a score: a row with no data at this
        # timepoint is not a subject assessed.
        tp_data = df[(df['timepoint'] == timepoint) & df['total_score'].notna()]
        if len(tp_data) > 0:
            summary[timepoint] = {
                'n_subjects': len(tp_data),
                'total_score': {
                    'mean': tp_data['total_score'].mean(),
                    'std': tp_data['total_score'].std(),
                    'min': tp_data['total_score'].min(),
                    'max': tp_data['total_score'].max(),
                    'median': tp_data['total_score'].median()
                }
            }
            
            # Item-level statistics
            item_cols = [col for col in tp_data.columns if col.startswith('item_')]
            item_means = {}
            for item_col in item_cols:
                item_means[item_col] = tp_data[item_col].mean()
            summary[timepoint]['item_means'] = item_means

            # 3-item BSCS total (primary substance, 0-12); n counts the
            # administrations with all three primary ratings answered.
            bscs = (tp_data['bscs_total'] if 'bscs_total' in tp_data
                    else pd.Series(dtype=float)).dropna()
            summary[timepoint]['bscs_total'] = {
                'n': int(len(bscs)),
                'mean': float(bscs.mean()) if len(bscs) else None,
                'std': float(bscs.std()) if len(bscs) > 1 else None,
                'min': float(bscs.min()) if len(bscs) else None,
                'max': float(bscs.max()) if len(bscs) else None,
                'median': float(bscs.median()) if len(bscs) else None,
            }
    
    return summary


def main():
    """Main execution function."""
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    print("Craving Assessment Analysis")
    print("=" * 50)
    
    # Load data
    data = load_craving_data()
    if data is None:
        return
    
    # Analyze all subjects
    results_df = analyze_all_subjects(data)
    print(f"\nAnalyzed {len(results_df)} subject-timepoint combinations")
    
    # Save results
    results_df.to_csv(SCORES_DIR / 'craving_scores.csv', index=False)
    print(f"\nScores saved to: {SCORES_DIR / 'craving_scores.csv'}")
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(results_df)
    with open(SCORES_DIR / 'craving_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create visualizations
    create_visualizations(results_df)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    
    # Print baseline summary
    print("\nBaseline Summary:")
    print("-" * 30)
    baseline_df = results_df[results_df['timepoint'] == 'baseline']
    print(f"N subjects: {len(baseline_df)}")
    print(f"Mean craving total: {baseline_df['total_score'].mean():.1f} ± {baseline_df['total_score'].std():.1f}")
    print(f"Range: {baseline_df['total_score'].min():.0f} - {baseline_df['total_score'].max():.0f}")
    
    # Item-level summary (if items exist)
    item_cols = [col for col in baseline_df.columns if col.startswith('item_')]
    if item_cols:
        print(f"\nItem-level means:")
        for item_col in sorted(item_cols):
            mean_score = baseline_df[item_col].mean()
            item_num = item_col.split('_')[1]
            print(f"  Item {item_num}: {mean_score:.1f}")

if __name__ == "__main__":
    main()
