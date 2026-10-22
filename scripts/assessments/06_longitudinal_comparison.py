#!/usr/bin/env python3
"""
06_longitudinal_comparison.py

Generate longitudinal comparison tables for assessments across timepoints.
Compares BRIEF-A t-scores (Inhibit, Shift, Working Memory, GEC) and
PHQ-9/GAD-7 total scores across different timepoint combinations.

Output: Tables with subjects as rows, measures as columns for each timepoint.

BRIEF-A administrations that fail a validity scale (brief_validity) are left
out, as in every analysis. The tables compare calendar visits
(baseline, post, follow-up); they are not the longitudinal analysis's
baseline-to-second-timepoint pairs, which stage 10 builds with
longitudinal/timepoint_selection.py (S3's and S5's second timepoint is their
follow-up).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_variant import refuse_protected_tree  # noqa: E402
from subject_ids import sort_subjects  # noqa: E402
from brief_validity import valid_administration_mask  # noqa: E402

# Load environment variables
load_dotenv()

# Define paths (OUTPUT_DIR is checked in main(), so this module can be imported without it)
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

SCORES_DIR = Path(OUTPUT_DIR) / "01_assessments" / "scores" if OUTPUT_DIR else None
T_SCORES_DIR = Path(OUTPUT_DIR) / "01_assessments" / "t-scores" if OUTPUT_DIR else None

# BRIEF-A t-score columns to extract
BRIEF_A_COLUMNS = {
    'Inhibit_t': 'Inhibit (T)',
    'Shift_t': 'Shift (T)',
    'Working Memory_t': 'Working Memory (T)',
    'GEC_t': 'GEC (T)',
}

# Timepoint file mappings
TIMEPOINT_FILE_MAP = {
    "baseline": "executive_function_baseline_subscale_t-scores.csv",
    "post": "executive_function_post_subscale_t-scores.csv",
    "followup": "executive_function_followup_subscale_t-scores.csv",
}


def load_t_score_data(timepoint: str) -> Optional[pd.DataFrame]:
    """Load BRIEF-A t-score data for a specific timepoint."""
    filepath = T_SCORES_DIR / TIMEPOINT_FILE_MAP[timepoint]
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    df = pd.read_csv(filepath)
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Filter to only valid subjects (those with non-zero total_score and n_items)
    if 'total_score' in df.columns and 'n_items' in df.columns:
        df = df[(df['total_score'] > 0) & (df['n_items'] > 0)].copy()

    return drop_invalid_administrations(df, timepoint)


# Administrations dropped for failing a validity scale, by timepoint; written
# to the metadata JSON.
INVALID_ADMINISTRATIONS: Dict[str, List[str]] = {}


def drop_invalid_administrations(df: pd.DataFrame, timepoint: str) -> pd.DataFrame:
    """Drop BRIEF-A administrations that fail a validity scale."""
    valid = valid_administration_mask(df)
    INVALID_ADMINISTRATIONS[timepoint] = sort_subjects(df.loc[~valid, 'subject'].astype(str))
    return df[valid].copy()


def load_phq9_data() -> pd.DataFrame:
    """Load PHQ-9 scores data."""
    filepath = SCORES_DIR / 'phq9_scores.csv'
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    # Filter to only rows with valid total_score (not empty and > 0 or is 0)
    df = df[df['total_score'].notna()].copy()
    return df[['subject', 'timepoint', 'total_score']]


def load_gad7_data() -> pd.DataFrame:
    """Load GAD-7 scores data."""
    filepath = SCORES_DIR / 'gad7_scores.csv'
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    # Filter to only rows with valid total_score
    df = df[df['total_score'].notna()].copy()
    return df[['subject', 'timepoint', 'total_score']]


def get_subjects_with_all_timepoints(
    data_dict: Dict[str, Optional[pd.DataFrame]],
    timepoints: List[str],
    subject_col: str = 'subject'
) -> List[str]:
    """Get list of subjects who have data for ALL specified timepoints."""
    subject_sets = []

    for tp in timepoints:
        df = data_dict.get(tp)
        if df is None or df.empty:
            return []

        subjects = set(df[subject_col].astype(str).unique())
        subject_sets.append(subjects)

    # Return intersection of all subject sets
    if not subject_sets:
        return []

    common_subjects = set.intersection(*subject_sets)
    # Numeric order: this drives the row order of every table below, and
    # sorted() on unpadded codes places S10 before S2.
    try:
        return sort_subjects(common_subjects)
    except ValueError as exc:
        # A subject column that does not hold S-codes lands here. Say what
        # is wrong rather than dumping a traceback; the validation itself
        # stays strict.
        raise SystemExit(
            f"ERROR: {exc}. The subject column must hold canonical S-codes "
            "-- the letter S followed by an unpadded number, e.g. S1, S12."
        ) from exc


def get_timepoint_sample_sizes(
    brief_data: Dict[str, Optional[pd.DataFrame]],
    phq9_data: pd.DataFrame,
    gad7_data: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """Get sample sizes for each timepoint across all assessments."""
    sample_sizes = {}

    timepoints = ['baseline', 'post', 'followup']

    for tp in timepoints:
        sample_sizes[tp] = {}

        # BRIEF-A t-scores
        df = brief_data.get(tp)
        if df is not None and not df.empty:
            sample_sizes[tp]['brief_a'] = len(df)
        else:
            sample_sizes[tp]['brief_a'] = 0

        # PHQ-9
        tp_phq9 = phq9_data[phq9_data['timepoint'] == tp]
        sample_sizes[tp]['phq9'] = len(tp_phq9)

        # GAD-7
        tp_gad7 = gad7_data[gad7_data['timepoint'] == tp]
        sample_sizes[tp]['gad7'] = len(tp_gad7)

    return sample_sizes


def create_comparison_table(
    subjects: List[str],
    brief_data: Dict[str, Optional[pd.DataFrame]],
    phq9_data: pd.DataFrame,
    gad7_data: pd.DataFrame,
    timepoints: List[str],
) -> pd.DataFrame:
    """Create a comparison table for specified subjects and timepoints."""
    if not subjects:
        return pd.DataFrame()

    # Build column mapping
    columns = ['subject']
    for tp in timepoints:
        for col, display_name in BRIEF_A_COLUMNS.items():
            columns.append(f"{display_name}_{tp}")
        columns.append(f"PHQ-9_total_{tp}")
        columns.append(f"GAD-7_total_{tp}")

    # Initialize data
    data = []

    for subj in subjects:
        row = {'subject': subj}

        for tp in timepoints:
            # BRIEF-A t-scores
            brief_df = brief_data.get(tp)
            if brief_df is not None:
                subj_row = brief_df[brief_df['subject'] == subj]
                if not subj_row.empty:
                    for col, display_name in BRIEF_A_COLUMNS.items():
                        row[f"{display_name}_{tp}"] = subj_row.iloc[0].get(col, pd.NA)
                else:
                    for display_name in BRIEF_A_COLUMNS.values():
                        row[f"{display_name}_{tp}"] = pd.NA
            else:
                for display_name in BRIEF_A_COLUMNS.values():
                    row[f"{display_name}_{tp}"] = pd.NA

            # PHQ-9 total
            phq9_row = phq9_data[(phq9_data['subject'] == subj) & (phq9_data['timepoint'] == tp)]
            if not phq9_row.empty:
                row[f"PHQ-9_total_{tp}"] = phq9_row.iloc[0]['total_score']
            else:
                row[f"PHQ-9_total_{tp}"] = pd.NA

            # GAD-7 total
            gad7_row = gad7_data[(gad7_data['subject'] == subj) & (gad7_data['timepoint'] == tp)]
            if not gad7_row.empty:
                row[f"GAD-7_total_{tp}"] = gad7_row.iloc[0]['total_score']
            else:
                row[f"GAD-7_total_{tp}"] = pd.NA

        data.append(row)

    return pd.DataFrame(data)


def compute_summary_statistics(
    df: pd.DataFrame,
    timepoints: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute mean and SD for each measure at each timepoint."""
    summary = {}

    if df.empty:
        return summary

    measures = list(BRIEF_A_COLUMNS.values()) + ['PHQ-9_total', 'GAD-7_total']

    for tp in timepoints:
        summary[tp] = {}
        for measure in measures:
            col = f"{measure}_{tp}"
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 0:
                    summary[tp][measure] = {
                        'n': len(values),
                        'mean': float(values.mean()),
                        'sd': float(values.std()),
                        'median': float(values.median()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                    }

    return summary


def print_summary_statistics(summary: Dict, timepoints: List[str]) -> None:
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    measures = list(BRIEF_A_COLUMNS.values()) + ['PHQ-9_total', 'GAD-7_total']

    for measure in measures:
        print(f"\n{measure}:")
        print("-" * 60)
        for tp in timepoints:
            if tp in summary and measure in summary[tp]:
                stats = summary[tp][measure]
                print(f"  {tp.capitalize():10s}: n={stats['n']:2d}, "
                      f"M={stats['mean']:6.2f}, SD={stats['sd']:5.2f}, "
                      f"range=[{stats['min']:.1f}, {stats['max']:.1f}]")
            else:
                print(f"  {tp.capitalize():10s}: No data")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate longitudinal comparison tables for assessments."
    )
    parser.add_argument(
        "--comparison",
        type=str,
        choices=["baseline_post", "baseline_post_followup", "all"],
        default="all",
        help="Which comparison to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tables (defaults to SCORES_DIR)",
    )
    args = parser.parse_args()

    if not OUTPUT_DIR:
        print("Error: OUTPUT_DIR must be set (in .env or the environment)")
        sys.exit(1)
    try:
        refuse_protected_tree(args.output_dir or OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")

    # Set output directory. SCORES_DIR is created only when it is the output
    # directory, so a protected OUTPUT_DIR is never written through it.
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = SCORES_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LONGITUDINAL ASSESSMENT COMPARISON")
    print("=" * 80)

    # Load data
    print("\nLoading data...")

    brief_data = {
        'baseline': load_t_score_data('baseline'),
        'post': load_t_score_data('post'),
        'followup': load_t_score_data('followup'),
    }

    phq9_data = load_phq9_data()
    gad7_data = load_gad7_data()

    # Get sample sizes for each timepoint
    print("\nSample sizes by timepoint:")
    print("-" * 50)
    sample_sizes = get_timepoint_sample_sizes(brief_data, phq9_data, gad7_data)
    for tp, sizes in sample_sizes.items():
        print(f"\n{tp.capitalize()}:")
        for measure, n in sizes.items():
            print(f"  {measure:10s}: n={n}")

    # Determine which comparisons to run
    comparisons = []
    if args.comparison == "baseline_post":
        comparisons.append(("baseline_post", ['baseline', 'post']))
    elif args.comparison == "baseline_post_followup":
        comparisons.append(("baseline_post_followup", ['baseline', 'post', 'followup']))
    else:  # all
        comparisons.append(("baseline_post", ['baseline', 'post']))
        comparisons.append(("baseline_post_followup", ['baseline', 'post', 'followup']))

    # Sample size metadata
    all_metadata = {
        'timepoint_sample_sizes': sample_sizes,
        'excluded_invalid_brief_a': INVALID_ADMINISTRATIONS,
        'comparisons': {},
    }

    # Generate tables for each comparison
    for name, timepoints in comparisons:
        print(f"\n{'=' * 80}")
        print(f"Generating comparison: {name}")
        print(f"Timepoints: {', '.join(timepoints)}")
        print(f"{'=' * 80}")

        # Get subjects with data for all timepoints
        subjects = get_subjects_with_all_timepoints(brief_data, timepoints)

        if not subjects:
            print(f"\nNo subjects with complete data for {', '.join(timepoints)}")
            all_metadata['comparisons'][name] = {
                'timepoints': timepoints,
                'n_complete': 0,
                'subjects': [],
            }
            continue

        print(f"\nSubjects with complete data for all timepoints: {len(subjects)}")
        for subj in subjects:
            print(f"  - {subj}")

        # Also check PHQ-9 and GAD-7 complete data
        phq9_complete = []
        gad7_complete = []

        for subj in subjects:
            has_all_phq9 = all(
                len(phq9_data[(phq9_data['subject'] == subj) & (phq9_data['timepoint'] == tp)]) > 0
                for tp in timepoints
            )
            has_all_gad7 = all(
                len(gad7_data[(gad7_data['subject'] == subj) & (gad7_data['timepoint'] == tp)]) > 0
                for tp in timepoints
            )

            if has_all_phq9:
                phq9_complete.append(subj)
            if has_all_gad7:
                gad7_complete.append(subj)

        print(f"\n  Complete BRIEF-A data: {len(subjects)}")
        print(f"  Complete PHQ-9 data: {len(phq9_complete)}")
        print(f"  Complete GAD-7 data: {len(gad7_complete)}")

        # Store metadata
        all_metadata['comparisons'][name] = {
            'timepoints': timepoints,
            'n_complete_brief_a': len(subjects),
            'n_complete_phq9': len(phq9_complete),
            'n_complete_gad7': len(gad7_complete),
            'subjects_brief_a': subjects,
            'subjects_phq9': phq9_complete,
            'subjects_gad7': gad7_complete,
        }

        # Create and save table
        df = create_comparison_table(subjects, brief_data, phq9_data, gad7_data, timepoints)

        output_file = output_path / f"longitudinal_comparison_{name}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nTable saved to: {output_file}")

        # Compute and print summary statistics
        summary = compute_summary_statistics(df, timepoints)
        print_summary_statistics(summary, timepoints)

    # Save metadata
    metadata_file = output_path / "longitudinal_comparison_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
