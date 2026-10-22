#!/usr/bin/env python3
"""
01_analyze_executive_function.py

Calculate executive function subscale and composite raw scores by timepoint.

"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from natsort import natsorted

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subject_ids import subject_column  # noqa: E402
from analysis_variant import refuse_protected_tree  # noqa: E402

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

# Define subscale mappings (using 1-based indexing as Item_Number is 1-based)
SUBSCALES = {
    'Inhibit': {
        'items': [5, 16, 29, 36, 43, 55, 58, 73],
        'description': 'Control over impulses and behavior'
    },
    'Shift': {
        'items': [8, 22, 32, 44, 61, 67],
        'description': 'Flexible switching between tasks or mental sets'
    },
    'Emotional Control': {
        'items': [1, 12, 19, 28, 33, 42, 51, 57, 69, 72],
        'description': 'Modulation of emotional responses'
    },
    'Self-Monitor': {
        'items': [13, 23, 37, 50, 64, 70],
        'description': 'Awareness of effect of behavior on others'
    },
    'Initiate': {
        'items': [6, 14, 20, 25, 45, 49, 53, 62],
        'description': 'Beginning tasks or activities independently'
    },
    'Working Memory': {
        'items': [4, 11, 17, 26, 35, 46, 56, 68],
        'description': 'Holding information in mind for task completion'
    },
    'Plan/Organize': {
        'items': [9, 15, 21, 34, 39, 47, 54, 63, 66, 71],
        'description': 'Planning and organizing problem-solving approaches'
    },
    'Task Monitor': {
        'items': [2, 18, 24, 41, 52, 75],
        'description': 'Task monitoring and checking work'
    },
    'Org Materials': {
        'items': [3, 7, 30, 31, 40, 60, 65, 74],
        'description': 'Organization of materials and belongings'
    }
}

INDICES = {
    'BRI': {
        'subscales': ['Inhibit', 'Shift', 'Emotional Control', 'Self-Monitor'],
        'description': 'Behavioral Regulation Index'
    },
    'MI': {
        'subscales': ['Initiate', 'Working Memory', 'Plan/Organize', 'Task Monitor', 'Org Materials'],
        'description': 'Metacognition Index'
    }
}

# BRIEF-A Inconsistency scale: 10 pairs of items with similar content
# Each pair should have similar responses; large differences indicate inconsistent responding
# Inconsistency score = sum of absolute differences between paired items
INCONSISTENCY_PAIRS = [
    (2, 41),    # Pair 1
    (25, 49),   # Pair 2
    (28, 42),   # Pair 3
    (33, 72),   # Pair 4
    (34, 63),   # Pair 5
    (44, 61),   # Pair 6
    (46, 56),   # Pair 7
    (52, 75),   # Pair 8
    (60, 74),   # Pair 9
    (64, 70),   # Pair 10
]

TIMEPOINT_ORDER = ['baseline', 'mid', 'post', 'followup']


def load_executive_function_data():
    """Load preprocessed executive function data."""
    filepath = SCORES_DIR / 'executive_function_processed.json'

    if not filepath.exists():
        print("Error: Preprocessed data not found. Run 00_load_assessment_data.py first.")
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def prorate_missing_items(responses: pd.Series, item_numbers: pd.Series) -> tuple[pd.Series, dict]:
    """Fill each scale's missing items from the mean of its answered items.

    A missing or unreadable item (``-``) is given the mean of the
    answered items of the same clinical scale, rounded to the nearest integer
    (halves round up). A raw score summed over only the answered items would
    otherwise be read against the norms as if the item had been answered
    "Never" or not at all. A scale with no answered item (an administration
    that did not happen) is left unfilled.

    Returns the prorated responses and the number of prorated items per scale.
    """
    filled = responses.copy()
    n_prorated = {}
    for subscale, info in SUBSCALES.items():
        in_scale = item_numbers.isin(info['items'])
        answered = filled[in_scale].dropna()
        missing = in_scale & filled.isna()
        n_prorated[subscale] = 0
        if answered.empty or not missing.any():
            continue
        filled[missing] = float(np.floor(answered.mean() + 0.5))
        n_prorated[subscale] = int(missing.sum())
    return filled, n_prorated


def calculate_scores(subject_data, timepoint):
    """Calculate total, subscale, and index raw scores for a subject/timepoint."""
    if timepoint not in subject_data:
        return None

    items = subject_data[timepoint]
    df = pd.DataFrame(items)

    # The column named by the subject's code holds the responses
    subject_col = subject_column(df.columns)
    df[subject_col] = pd.to_numeric(df[subject_col], errors='coerce')

    # n_items counts the items actually answered; the raw scores are summed
    # over the prorated responses.
    n_items = df[subject_col].notna().sum()
    prorated, n_prorated = prorate_missing_items(df[subject_col], df['Item_Number'])
    total_score = prorated.sum()

    subscale_scores = {}
    for subscale, info in SUBSCALES.items():
        in_scale = df['Item_Number'].isin(info['items'])
        subscale_score = prorated[in_scale].sum()
        subscale_n = df.loc[in_scale, subject_col].notna().sum()
        subscale_scores[subscale] = {'raw_score': subscale_score, 'n_items': subscale_n,
                                     'n_prorated': n_prorated[subscale]}

    index_scores = {}
    bri_score = sum(subscale_scores[s]['raw_score'] for s in INDICES['BRI']['subscales'])
    index_scores['BRI'] = bri_score
    mi_score = sum(subscale_scores[s]['raw_score'] for s in INDICES['MI']['subscales'])
    index_scores['MI'] = mi_score
    index_scores['GEC'] = bri_score + mi_score

    # Calculate inconsistency score as sum of absolute differences between paired items
    # (on the answered items only: a prorated value is not a response)
    inconsistency_score = 0
    n_pairs_valid = 0
    for item1, item2 in INCONSISTENCY_PAIRS:
        val1_rows = df[df['Item_Number'] == item1][subject_col]
        val2_rows = df[df['Item_Number'] == item2][subject_col]
        if not val1_rows.empty and not val2_rows.empty:
            val1 = val1_rows.iloc[0]
            val2 = val2_rows.iloc[0]
            if pd.notna(val1) and pd.notna(val2):
                inconsistency_score += abs(val1 - val2)
                n_pairs_valid += 1
    # BRIEF-A manual cutoff: an Inconsistency score above 7 marks the
    # administration as questionable (acceptable = 7 or less).
    inconsistency_acceptable = inconsistency_score <= 7

    return {
        'total_score': total_score,
        'n_items': n_items,
        'subscales': subscale_scores,
        'indices': index_scores,
        'inconsistency_score': inconsistency_score,
        'n_inconsistency_pairs': n_pairs_valid,
        'inconsistency_acceptable': inconsistency_acceptable
    }


def compile_subject_scores(data):
    """Build a long-format dataframe of subject scores across timepoints."""
    records = []

    for subject, timepoints in data.items():
        for timepoint in TIMEPOINT_ORDER:
            scores = calculate_scores(timepoints, timepoint)
            if not scores:
                continue

            record = {
                'subject': subject,
                'timepoint': timepoint,
                'total_score': scores['total_score'],
                'n_items': scores['n_items'],
                'inconsistency_score': scores['inconsistency_score'],
                'n_inconsistency_pairs': scores['n_inconsistency_pairs'],
                'inconsistency_acceptable': scores['inconsistency_acceptable']
            }

            for subscale, details in scores['subscales'].items():
                record[subscale] = details['raw_score']
                record[f'{subscale}_n_prorated'] = details['n_prorated']

            for index_name, value in scores['indices'].items():
                record[index_name] = value

            records.append(record)

    return pd.DataFrame(records)


def write_timepoint_csvs(scores_df):
    """Write per-timepoint CSVs containing subscale and composite raw scores.

    Raw scores only. The age- and gender-normed T-scores (and the Infrequency
    and Negativity validity flags) that later stages read from
    01_assessments/t-scores/ are entered by hand from the published norm
    tables; nothing in this repository computes them.
    """
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    if scores_df.empty:
        print("No executive function data available after processing.")
        return

    subscale_columns = list(SUBSCALES.keys())
    index_columns = list(INDICES.keys()) + ['GEC']
    extra_columns = ['inconsistency_score', 'n_inconsistency_pairs', 'inconsistency_acceptable']
    prorated_columns = [f'{subscale}_n_prorated' for subscale in SUBSCALES]

    for timepoint in TIMEPOINT_ORDER:
        timepoint_df = scores_df[scores_df['timepoint'] == timepoint].copy()
        if timepoint_df.empty:
            continue

        ordered_subjects = natsorted(timepoint_df['subject'].tolist())
        timepoint_df = timepoint_df.set_index('subject').loc[ordered_subjects].reset_index()

        columns = (['subject', 'total_score', 'n_items'] + subscale_columns + index_columns
                   + extra_columns + prorated_columns)
        timepoint_df = timepoint_df[columns]

        csv_path = SCORES_DIR / f"executive_function_{timepoint}_subscale_scores.csv"
        timepoint_df.to_csv(csv_path, index=False)
        print(f"Saved {timepoint} subscale scores to: {csv_path}")


def main():
    """Entry point for generating executive function raw score tables."""
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    data = load_executive_function_data()
    if data is None:
        return

    scores_df = compile_subject_scores(data)
    write_timepoint_csvs(scores_df)


if __name__ == "__main__":
    main()
