#!/usr/bin/env python3
"""
02_analyze_phq9.py

Analyze PHQ-9 (Patient Health Questionnaire-9) depression assessment data:
- Calculate total depression scores
- Classify depression severity levels
- Create enhanced visualizations with severity-based colors
- Individual participant profiles

"""

import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
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


# PHQ-9 severity classifications (standard clinical cutoffs)
SEVERITY_CUTOFFS = {
    'Minimal': (0, 4),
    'Mild': (5, 9),
    'Moderate': (10, 14),
    'Moderately severe': (15, 19),
    'Severe': (20, 27)
}

# Define severity colors (more distinguishable)
SEVERITY_COLORS = {
    'Minimal': '#4CAF50',      # Green
    'Mild': '#FFEB3B',         # Bright Yellow
    'Moderate': '#FF9800',     # Orange
    'Moderately severe': '#E91E63',  # Pink/Magenta
    'Severe': '#F44336'        # Red
}

DEFAULT_FIGURE_DPI = 300
COLOR_MAP = {
    'baseline': '#1E88E5',
    'mid': '#6A1B9A',
    'post': '#FFC107',
    'followup': '#004D40',
}
TIMEPOINT_DISPLAY_ORDER: Sequence[str] = ('baseline', 'mid', 'post', 'followup')
MEAN_LINE_COLOR = '#D81B60'



def format_subject_label(subject: str) -> str:
    """Return the display form of a subject identifier.

    Canonical codes are already display-ready; this exists so callers have
    one place to change if that stops being true.
    """
    return standardize_subject_id(subject)

# PHQ-9 items (for reference)
PHQ9_ITEMS = {
    1: "Little interest or pleasure in doing things",
    2: "Feeling down, depressed, or hopeless",
    3: "Trouble falling/staying asleep, or sleeping too much",
    4: "Feeling tired or having little energy",
    5: "Poor appetite or overeating",
    6: "Feeling bad about yourself or that you are a failure",
    7: "Trouble concentrating on things",
    8: "Moving or speaking slowly/being fidgety or restless",
    9: "Thoughts that you would be better off dead"
}

# Functional impairment follow-up question (item 10)
PHQ9_FUNCTIONAL_IMPACT_TEXT = (
    'Difficulty doing work, caring for home, or getting along with others'
)

def load_phq9_data():
    """Load preprocessed PHQ-9 data."""
    filepath = SCORES_DIR / 'phq9_processed.json'
    
    if not filepath.exists():
        print("Error: Preprocessed data not found. Run 00_load_assessment_data.py first.")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

def calculate_phq9_scores(subject_data, timepoint):
    """Calculate PHQ-9 total score and item-level scores."""
    if timepoint not in subject_data:
        return None
    
    items = subject_data[timepoint]
    df = pd.DataFrame(items)
    
    # The column named by the subject's code holds the responses
    subject_col = subject_column(df.columns)
    
    # Convert to numeric
    df[subject_col] = pd.to_numeric(df[subject_col], errors='coerce')
    
    # Filter to only PHQ-9 items (1-9, exclude item 10)
    df_phq9 = df[df['Item_Number'].between(1, 9)]
    
    # Count valid responses (only items 1-9)
    n_items = df_phq9[subject_col].notna().sum()
    
    # If no valid responses, return None for total score
    if n_items == 0:
        total_score = np.nan
    else:
        total_score = df_phq9[subject_col].sum()
    
    # Get item-level scores and retain functional impairment item (10)
    item_scores = {}
    for idx, row in df.iterrows():
        item_num = row['Item_Number']
        if 1 <= item_num <= 10:
            item_scores[f'item_{item_num}'] = row[subject_col]
    
    # Calculate suicidal ideation flag (item 9 > 0)
    # Only if item 9 has valid data
    item_9_value = item_scores.get('item_9', np.nan)
    # Unknown, not "absent", when item 9 was not answered
    suicidal_ideation = bool(item_9_value > 0) if pd.notna(item_9_value) else np.nan
    
    return {
        'total_score': total_score,
        'n_items': n_items,
        'suicidal_ideation': suicidal_ideation,
        'item_scores': item_scores
    }

def classify_severity(total_score):
    """Classify depression severity based on PHQ-9 total score."""
    if pd.isna(total_score):
        return 'Unknown'
    
    for severity, (min_score, max_score) in SEVERITY_CUTOFFS.items():
        if min_score <= total_score <= max_score:
            return severity
    
    return 'Unknown'

def analyze_all_subjects(data):
    """Analyze PHQ-9 data for all subjects."""
    results = []
    
    for subject, timepoints in data.items():
        for timepoint in ['baseline', 'mid', 'post', 'followup']:
            scores = calculate_phq9_scores(timepoints, timepoint)
            if scores:
                result = {
                    'subject': subject,
                    'timepoint': timepoint,
                    'total_score': scores['total_score'],
                    'n_items': scores['n_items'],
                    'suicidal_ideation': scores['suicidal_ideation']
                }
                
                # Add individual item scores
                for item_key, item_score in scores['item_scores'].items():
                    result[item_key] = item_score
                
                # Add severity classification
                result['severity'] = classify_severity(scores['total_score'])
                
                results.append(result)
    
    return pd.DataFrame(results)

def get_item_columns(df: pd.DataFrame) -> List[str]:
    """Return ordered PHQ-9 item columns present in the dataframe."""

    item_cols = [
        col
        for col in df.columns
        if col.startswith('item_') and col.split('_')[1].isdigit()
    ]
    item_cols.sort(key=lambda col: int(col.split('_')[1]))
    return item_cols


def _resolve_item_labels(columns: Sequence[str]) -> List[str]:
    """Return concise labels for PHQ-9 item columns."""

    labels: List[str] = []
    for column in columns:
        suffix = column.split('_')[1]
        if suffix.isdigit():
            number = int(suffix)
            labels.append(f'Q{number}')
        else:
            labels.append(column)
    return labels


def plot_baseline_individuals(
    df: pd.DataFrame,
    group_means: pd.Series,
    item_cols: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Create a grid of baseline PHQ-9 item scores with group means indicated."""

    subjects = natsorted(df['subject'].dropna().astype(str).unique())
    if not subjects or not item_cols:
        return

    n_cols = min(4, max(1, len(subjects)))
    n_rows = math.ceil(len(subjects) / n_cols)
    x_positions = range(len(item_cols))
    item_labels = _resolve_item_labels(item_cols)
    subject_max = df[item_cols].max().max()
    mean_max = group_means.max() if not group_means.empty else np.nan
    candidates = [3.0]
    if not pd.isna(subject_max):
        candidates.append(float(subject_max))
    if not pd.isna(mean_max):
        candidates.append(float(mean_max))
    shared_ylim = max(candidates) + 0.5

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.0, n_rows * 3.5),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle('Baseline PHQ-9 Item Scores by Participant', fontsize=14)
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
        ax.set_ylim(0, shared_ylim)
        ax.set_title(format_subject_label(subject), fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    for ax in flat_axes[len(subjects):]:
        ax.axis('off')

    flat_axes[0].set_ylabel('Item Score (0-3)')
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
    upper_bound = max(3.0, float(max_candidate) if pd.notna(max_candidate) else 0.0) + 0.5

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(
        x_positions,
        mean_scores.values,
        color=COLOR_MAP['baseline'],
        alpha=0.8,
        width=0.8,
        edgecolor=COLOR_MAP['baseline'],
        linewidth=1.2,
    )

    rng = np.random.default_rng(seed=314159)
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
    ax.set_ylabel('Item Score (0-3)')
    ax.set_ylim(0, upper_bound)
    ax.set_title('Baseline PHQ-9 (Group Mean)', fontsize=12)
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
    for df in datasets.values():
        if df.empty:
            return []
        subject_sets.append(set(df['subject'].dropna().astype(str)))

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
    n_rows = math.ceil(len(aligned_scores) / n_cols)
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
        upper_bound = max(3.0, max(v for v in max_values if not pd.isna(v))) + 0.5
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(item_labels, rotation=90, ha='center')
        ax.set_ylim(0, upper_bound)
        ax.set_title(format_subject_label(subject), fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    for ax in flat_axes[len(aligned_scores):]:
        ax.axis('off')

    flat_axes[0].set_ylabel('Item Score (0-3)')

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
    """Create PHQ-9 figures following the executive function plotting logic."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_style('whitegrid')
    item_cols = get_item_columns(df)
    if not item_cols:
        print('No PHQ-9 item columns available for plotting. Skipping visualization.')
        return

    baseline_df = df[df['timepoint'] == 'baseline'].copy()
    if baseline_df.empty:
        print('No baseline PHQ-9 data available. Skipping visualization.')
        return

    baseline_df[item_cols] = baseline_df[item_cols].apply(pd.to_numeric, errors='coerce')
    baseline_means = baseline_df[item_cols].mean()

    baseline_grid_path = FIGURES_DIR / 'phq9_baseline_individual_grid.png'
    baseline_summary_path = FIGURES_DIR / 'phq9_baseline_mean_summary.png'
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
        mid_path = FIGURES_DIR / 'phq9_baseline_mid_trajectories.png'
        if mid_aligned:
            plot_timepoint_item_trajectories(
                mid_aligned,
                item_cols,
                timepoints=('baseline', 'mid'),
                output_path=mid_path,
                dpi=DEFAULT_FIGURE_DPI,
                title='Baseline vs. Mid PHQ-9 Item Trajectories',
            )
            saved_paths.append(mid_path)

    if not post_df.empty:
        post_aligned = align_subject_item_scores({
            'baseline': baseline_df,
            'post': post_df,
        }, item_cols)
        post_path = FIGURES_DIR / 'phq9_baseline_post_trajectories.png'
        if post_aligned:
            plot_timepoint_item_trajectories(
                post_aligned,
                item_cols,
                timepoints=('baseline', 'post'),
                output_path=post_path,
                dpi=DEFAULT_FIGURE_DPI,
                title='Baseline vs. Post PHQ-9 Item Trajectories',
            )
            saved_paths.append(post_path)

    if not followup_df.empty:
        followup_datasets: Dict[str, pd.DataFrame] = {'baseline': baseline_df}
        timepoint_sequence: List[str] = ['baseline']
        if not post_df.empty:
            followup_datasets['post'] = post_df
            timepoint_sequence.append('post')
        followup_datasets['followup'] = followup_df
        timepoint_sequence.append('followup')
        followup_aligned = align_subject_item_scores(followup_datasets, item_cols)
        if followup_aligned:
            if 'post' in followup_datasets:
                followup_path = FIGURES_DIR / 'phq9_baseline_post_followup_trajectories.png'
                followup_title = 'Baseline, Post, and Follow-up PHQ-9 Item Trajectories'
            else:
                followup_path = FIGURES_DIR / 'phq9_baseline_followup_trajectories.png'
                followup_title = 'Baseline vs. Follow-up PHQ-9 Item Trajectories'
            plot_timepoint_item_trajectories(
                followup_aligned,
                item_cols,
                timepoints=tuple(timepoint_sequence),
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
    """Generate summary statistics for PHQ-9 data."""
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
                },
                'severity_distribution': tp_data['severity'].value_counts().to_dict(),
                'suicidal_ideation_rate': pd.to_numeric(
                    tp_data['suicidal_ideation'], errors='coerce').mean() * 100
            }
            
            # Item-level statistics
            item_cols = sorted(
                (col for col in tp_data.columns if col.startswith('item_')),
                key=lambda col: int(col.split('_')[1]) if col.split('_')[1].isdigit() else float('inf')
            )
            item_means = {col: tp_data[col].mean() for col in item_cols}
            summary[timepoint]['item_means'] = item_means
    
    return summary

def main():
    """Main execution function."""
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    print("PHQ-9 Depression Analysis")
    print("=" * 50)
    
    # Load data
    data = load_phq9_data()
    if data is None:
        return
    
    # Analyze all subjects
    results_df = analyze_all_subjects(data)
    print(f"\nAnalyzed {len(results_df)} subject-timepoint combinations")
    
    # Save results
    results_df.to_csv(SCORES_DIR / 'phq9_scores.csv', index=False)
    print(f"\nScores saved to: {SCORES_DIR / 'phq9_scores.csv'}")
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(results_df)
    with open(SCORES_DIR / 'phq9_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create visualizations
    create_visualizations(results_df)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    
    # Print baseline summary
    print("\nBaseline Summary:")
    print("-" * 30)
    baseline_df = results_df[results_df['timepoint'] == 'baseline']
    print(f"N subjects: {len(baseline_df)}")
    print(f"Mean PHQ-9 total: {baseline_df['total_score'].mean():.1f} ± {baseline_df['total_score'].std():.1f}")
    print(f"Range: {baseline_df['total_score'].min():.0f} - {baseline_df['total_score'].max():.0f}")
    
    print("\nSeverity distribution:")
    severity_counts = baseline_df['severity'].value_counts()
    for severity in ['Minimal', 'Mild', 'Moderate', 'Moderately severe', 'Severe']:
        count = severity_counts.get(severity, 0)
        percent = (count / len(baseline_df)) * 100
        print(f"  {severity}: {count} ({percent:.1f}%)")
    
    si = pd.to_numeric(baseline_df['suicidal_ideation'], errors='coerce')
    print(f"\nSuicidal ideation present: {int(si.sum())} subjects ({(si.mean() * 100):.1f}%)")
    
    # Item-level summary
    print(f"\nItem-level means (0-3 scale):")
    for i in range(1, 10):
        item_col = f'item_{i}'
        if item_col in baseline_df.columns:
            mean_score = baseline_df[item_col].mean()
            item_desc = PHQ9_ITEMS[i][:40] + "..." if len(PHQ9_ITEMS[i]) > 40 else PHQ9_ITEMS[i]
            print(f"  Q{i}: {mean_score:.1f} - {item_desc}")
    if 'item_10' in baseline_df.columns:
        mean_score = baseline_df['item_10'].mean()
        print(f"  Q10: {mean_score:.1f} - {PHQ9_FUNCTIONAL_IMPACT_TEXT}")

if __name__ == "__main__":
    main()
