#!/usr/bin/env python3
"""
00_load_assessment_data.py

Load and preprocess all assessment data from CSV files.
This script handles data loading, basic cleaning, and standardization.

"""

import os
import re
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subject_ids import is_subject_id, sort_subjects  # noqa: E402
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

ASSESSMENT_DIR = Path(BASE_DIR) / "data" / "assessments"
OUTPUT_DIR = Path(OUTPUT_DIR) / "01_assessments" / "scores"


# Define assessment types and their files
ASSESSMENTS = {
    'executive_function': ['executive_function_baseline.csv', 'executive_function_followup.csv', 'executive_function_post.csv'],
    'phq9': ['phq9_baseline.csv', 'phq9_followup.csv', 'phq9_post.csv'],
    'gad7': ['gad7_baseline.csv', 'gad7_followup.csv', 'gad7_post.csv'],
    'craving': ['craving_baseline.csv', 'craving_followup.csv', 'craving_mid.csv', 'craving_post.csv'],
    'tea': ['tea_baseline.csv', 'tea_followup.csv', 'tea_mid.csv', 'tea_post.csv']
}

TIMEPOINT_ORDER = ['baseline', 'mid', 'post', 'followup']


def slugify(value):
    """Create a filesystem-friendly slug from a value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'unknown'

    value_str = str(value).strip().lower()
    value_str = re.sub(r'[^a-z0-9]+', '_', value_str)
    value_str = value_str.strip('_')
    return value_str or 'unknown'


def format_item_identifier(item_value, row_index=None):
    """Create stable item identifiers for column names."""
    if item_value is None or (isinstance(item_value, float) and np.isnan(item_value)):
        if row_index is not None:
            return f"{row_index + 1:03d}"
        return 'unknown'

    try:
        numeric_value = float(item_value)
        if numeric_value.is_integer():
            return f"{int(numeric_value):03d}"
    except (TypeError, ValueError):
        pass

    return slugify(item_value)


def get_subscale_column(df):
    """Return the first matching subscale column name if available."""
    subscale_candidates = ['Subscale', 'Subscale_Name', 'Subscale_Label', 'Scale', 'Domain']
    for column in subscale_candidates:
        if column in df.columns:
            return column
    return None


def build_timepoint_item_matrix(assessment_data, timepoint):
    """Create a wide matrix of item-level raw scores (with optional subscales)."""
    rows = []
    observed_item_cols = set()
    observed_subscale_cols = set()

    item_candidates = ['Item_Number', 'Item', 'ItemID', 'Item_ID', 'Question_Number']

    for subject, timepoints in assessment_data.items():
        if timepoint not in timepoints:
            continue

        timepoint_df = timepoints[timepoint].copy()

        if subject not in timepoint_df.columns:
            continue

        timepoint_df[subject] = pd.to_numeric(timepoint_df[subject], errors='coerce')

        item_column = next((col for col in item_candidates if col in timepoint_df.columns), None)
        subscale_column = get_subscale_column(timepoint_df)

        if item_column is None:
            timepoint_df = timepoint_df.reset_index().rename(columns={'index': '_Row_Index'})
            item_column = '_Row_Index'

        row = {'subject': subject}

        for idx, item_row in timepoint_df.iterrows():
            item_identifier = item_row[item_column]
            formatted_item = format_item_identifier(item_identifier, row_index=idx if item_column == '_Row_Index' else None)
            column_name = f"item_{formatted_item}"
            row[column_name] = item_row[subject]
            observed_item_cols.add(column_name)

        if subscale_column:
            aggregated = (
                timepoint_df.groupby(subscale_column)[subject]
                .apply(lambda series: series.sum(min_count=1))
            )

            for subscale_name, score in aggregated.items():
                column_name = f"subscale_{slugify(subscale_name)}"
                row[column_name] = score
                observed_subscale_cols.add(column_name)

        if any(key != 'subject' for key in row.keys()):
            rows.append(row)

    if not rows:
        return None

    matrix_df = pd.DataFrame(rows)

    def sort_item(name):
        identifier = name.replace('item_', '', 1)
        try:
            return int(identifier)
        except ValueError:
            return identifier

    ordered_columns = ['subject']
    ordered_columns.extend(sorted(observed_item_cols, key=sort_item))
    ordered_columns.extend(sorted(observed_subscale_cols))

    return matrix_df.reindex(columns=ordered_columns)

def load_assessment_file(filepath):
    """Load a single assessment CSV file; a read error raises."""
    df = pd.read_csv(filepath)
    print(f"Loaded {filepath.name}: {df.shape}")
    return df

def extract_timepoint(filename):
    """Extract timepoint from filename."""
    parts = filename.replace('.csv', '').split('_')
    timepoint = parts[-1]
    return timepoint

def parse_scale_range(scale_value):
    """Parse scale range string (e.g., '0-3', '1-8') to get min and max values."""
    if pd.isna(scale_value) or scale_value == '-':
        return None, None
    
    scale_str = str(scale_value).strip()
    if '-' in scale_str:
        try:
            parts = scale_str.split('-')
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None

def convert_response_to_numeric(value, scale_value=None, response_options=None):
    """Convert text responses to numeric values based on scale information."""
    # Handle missing data
    if pd.isna(value) or str(value).strip() == '-':
        return np.nan
    
    value_str = str(value).upper().strip()
    
    # A double response (e.g. 'N S': two Never/Sometimes/Often codes marked
    # for one item) is scored as the LOWER of the codes marked (N=1 < S=2 <
    # O=3). This is the study's rule; it is not
    # prorated. A missing or unreadable item ('-') is prorated by the BRIEF-A
    # scorer instead (01_analyze_executive_function.prorate_missing_items).
    tokens = value_str.split()
    if len(tokens) > 1:
        codes = {'N': 1, 'S': 2, 'O': 3}
        if all(token in codes for token in tokens):
            lower = min(codes[token] for token in tokens)
            print(f"Note: double response {value!r} scored as the lower code ({lower})")
            return lower
        raise ValueError(f"Cannot convert response {value!r} to a number")
    
    # Never/Sometimes/Often scale (N/S/O)
    if value_str in ['N', 'NEVER']:
        return 1
    elif value_str in ['S', 'SOMETIMES']:
        return 2
    elif value_str in ['O', 'OFTEN']:
        return 3
    
    # Try to convert to float if it's already numeric
    try:
        numeric_value = float(value)
        
        # Validate against scale range if provided
        if scale_value:
            scale_min, scale_max = parse_scale_range(scale_value)
            if scale_min is not None and scale_max is not None:
                if numeric_value < scale_min or numeric_value > scale_max:
                    print(f"Warning: Value {numeric_value} outside expected range {scale_min}-{scale_max}")
        
        return numeric_value
    except (ValueError, TypeError):
        # If conversion fails and we have response options, try to parse them
        if response_options and pd.notna(response_options):
            options = str(response_options).split('/')
            for i, option in enumerate(options):
                if option.upper().strip() == value_str:
                    # Return index based on scale range if available
                    if scale_value:
                        scale_min, scale_max = parse_scale_range(scale_value)
                        if scale_min is not None:
                            return scale_min + i
                    return i
        
        # An unreadable response raises: passed on as a string, the scoring
        # scripts would coerce it to NaN and lower a raw score silently.
        raise ValueError(f"Cannot convert response {value!r} to a number")

def process_assessment_data(assessment_type, files):
    """Process data for a specific assessment type across all timepoints."""
    print(f"\nProcessing {assessment_type}...")
    
    all_data = {}
    
    for file in files:
        filepath = ASSESSMENT_DIR / file
        if not filepath.exists():
            raise FileNotFoundError(f"Assessment file not found: {filepath}")

        df = load_assessment_file(filepath)
            
        timepoint = extract_timepoint(file)
        
        # Subjects are columns in these wide sheets; the canonical code
        # identifies them, and the remaining columns are question metadata.
        subject_cols = sort_subjects(
            [col for col in df.columns if is_subject_id(col)]
        )

        # is_subject_id is exact, where the old prefix test was loose. A sheet
        # whose headers have not been rewritten therefore selects nothing and
        # would contribute no rows at all -- silently, since the per-subject
        # loop below simply does not execute.
        if not subject_cols:
            raise ValueError(f"{file} has no canonical subject columns. "
                             f"Headers seen: {list(df.columns)}")
        
        # Process each subject
        for subject in subject_cols:
            if subject not in all_data:
                all_data[subject] = {}
            
            # Check which columns are available
            available_cols = []
            if 'Item_Number' in df.columns:
                available_cols.append('Item_Number')
            if 'Question' in df.columns:
                available_cols.append('Question')
            available_cols.append(subject)
            
            # Convert responses to numeric
            subject_data = df[available_cols].copy()
            
            # Check for scale information columns
            scale_col = None
            response_col = None
            if 'Scale_Value' in df.columns:
                scale_col = 'Scale_Value'
            elif 'Scale_Range' in df.columns:
                scale_col = 'Scale_Range'
            
            if 'Response_Options' in df.columns:
                response_col = 'Response_Options'
            
            # Apply conversion with scale information
            if scale_col or response_col:
                for idx in subject_data.index:
                    scale_val = df.loc[idx, scale_col] if scale_col else None
                    response_opt = df.loc[idx, response_col] if response_col else None
                    subject_data.loc[idx, subject] = convert_response_to_numeric(
                        subject_data.loc[idx, subject], 
                        scale_value=scale_val,
                        response_options=response_opt
                    )
            else:
                # Fallback to simple conversion
                subject_data[subject] = subject_data[subject].apply(
                    lambda x: convert_response_to_numeric(x)
                )
            
            # Add Item_Number if missing (use row index + 1)
            if 'Item_Number' not in subject_data.columns:
                subject_data.insert(0, 'Item_Number', range(1, len(subject_data) + 1))
            
            all_data[subject][timepoint] = subject_data
            
    return all_data

def serialize_assessment_data(assessment_data):
    """Convert assessment dataframes to JSON-serializable structure."""
    serialized = {}
    for subject, timepoints in assessment_data.items():
        serialized[subject] = {}
        for timepoint, df in timepoints.items():
            serialized[subject][timepoint] = df.to_dict('records')
    return serialized


def write_assessment_outputs():
    """Process assessments and write JSON plus per-timepoint raw score CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for assessment_type, files in ASSESSMENTS.items():
        assessment_data = process_assessment_data(assessment_type, files)

        # Write JSON preserving item-level structure per subject/timepoint
        json_path = OUTPUT_DIR / f"{assessment_type}_processed.json"
        with open(json_path, 'w') as f:
            json.dump(serialize_assessment_data(assessment_data), f, indent=2)
        print(f"  Saved detailed JSON to {json_path}")

        # Write item-level raw scores (plus subscales when available) per timepoint
        for timepoint in TIMEPOINT_ORDER:
            matrix_df = build_timepoint_item_matrix(assessment_data, timepoint)
            if matrix_df is None:
                continue

            matrix_path = OUTPUT_DIR / f"{assessment_type}_{timepoint}_raw_scores.csv"
            matrix_df.to_csv(matrix_path, index=False)
            print(f"  Saved {timepoint} raw scores to {matrix_path}")

def main():
    """Main execution function."""
    try:
        refuse_protected_tree(str(OUTPUT_DIR))
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    print("Assessment Data Loading and Preprocessing")
    print("=" * 50)

    write_assessment_outputs()

    print("\n" + "=" * 50)
    print("Detailed JSON and raw score exports complete.")
    print(f"Output files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
