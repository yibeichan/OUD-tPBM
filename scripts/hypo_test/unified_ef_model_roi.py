#!/usr/bin/env python3
"""
Unified EF Model - ROI-Level Analysis (Confirmatory)

This script tests the 5 primary hypotheses at the ROI level only.
ROI-level uses PRE-AGGREGATED data (mean activation across channels in each ROI).

ROI DEFINITION (HARMONIZED WITH LONGITUDINAL ANALYSIS):
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

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import json

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'brain'))

from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
# OUTPUT_DIR is validated in main(); module is importable without it.

# Import common utilities
from unified_ef_model_common import (
    DEFAULT_SEED, rng_for_key, PRIMARY_HYPOTHESES,
    load_all_data, test_correlation, create_forest_plot,
    save_results, print_summary, create_hypothesis_scatter_plots,
    resolve_hypotheses
)

from utils.statistical_methods import setup_plotting, create_scatter_plot
from analysis_variant import load_variant, refuse_protected_tree


# The confirmatory family: H1-H4 x (self-report x neural, neural x behavioural).
# Self-report x behavioural is reported but not corrected, and H5 has no neural
# term, so neither enters. Pinned because a silent change here moves every
# published p_FDR.
PRIMARY_FAMILY_SIZE = 8


def is_primary_test(hypo_id: str, test_name: str) -> bool:
    """Whether this test belongs to the confirmatory FDR family.

    The family is the self-report x brain and brain x behaviour tests of
    H1-H4 (8 in all). H5 has no brain term and never enters.
    """
    return ('_self_brain' in test_name) or ('_brain_behav' in test_name)


# =============================================================================
# ROI-LEVEL ANALYSIS
# =============================================================================

def aggregate_roi_activation(roi_df: pd.DataFrame, target_rois: list,
                            contrast: str) -> pd.DataFrame:
    """Aggregate channels within target ROIs for a specific contrast.

    The input rows are already per-ROI means (step3b), so this is the
    unweighted mean of the matched ROIs' means for each subject.

    Args:
        roi_df: ROI activation DataFrame (pre-aggregated)
        target_rois: List of ROI network names to match
        contrast: Contrast name to filter

    Returns:
        DataFrame with subject-level aggregated activation
    """
    if len(roi_df) == 0:
        return pd.DataFrame()

    # Filter to target contrast
    contrast_df = roi_df[roi_df['contrast'] == contrast].copy()

    if len(contrast_df) == 0:
        return pd.DataFrame()

    # Find matching ROIs (partial match, case-insensitive)
    matched_rois = []
    for target in target_rois:
        for roi in contrast_df['network_roi'].unique():
            if target.lower() in str(roi).lower():
                matched_rois.append(roi)

    matched_rois = list(set(matched_rois))

    if len(matched_rois) == 0:
        return pd.DataFrame()

    # Filter to matched ROIs and aggregate per subject
    roi_data = contrast_df[contrast_df['network_roi'].isin(matched_rois)]

    # A subject missing one of the matched ROIs would get the other ROI's
    # mean alone, which is not comparable with the other subjects' values.
    rois_per_subject = roi_data.groupby('subject')['network_roi'].nunique()
    incomplete = rois_per_subject[rois_per_subject < len(matched_rois)]
    if not incomplete.empty:
        raise ValueError(
            f"{contrast}: subjects {sorted(incomplete.index)} lack one of the ROIs "
            f"{sorted(matched_rois)}"
        )

    # AGGREGATION: mean of the matched ROIs' means
    agg_df = roi_data.groupby('subject').agg({
        'activation': 'mean',
        't_statistic': 'mean',
        'n_channels': 'sum'
    }).reset_index()

    agg_df['n_rois'] = len(matched_rois)
    agg_df['matched_rois'] = ','.join(matched_rois)
    agg_df['contrast'] = contrast

    return agg_df


def test_hypothesis_roi_level(hypo_id: str, hypo_def: dict,
                              data: dict, *, seed: int = DEFAULT_SEED) -> dict:
    """Test a hypothesis at ROI level.

    ROI-level uses AGGREGATED data (mean activation across channels in each ROI).

    Args:
        hypo_id: Hypothesis ID (e.g., 'H1')
        hypo_def: Hypothesis definition
        data: Data dictionary
        seed: Base seed; each test derives its own generator from its name,
            so a test's p-value does not depend on which tests ran before it

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"{hypo_id}: {hypo_def['name']}")
    print(f"{'='*60}")

    results = {
        'hypothesis_id': hypo_id,
        'name': hypo_def['name'],
        'tests': []
    }

    merged = data['merged'].copy()

    # H5 is direct correlation only
    if hypo_id == 'H5':
        brief_col = hypo_def['brief_measure']
        behav_col = hypo_def['behavioral_measure']

        missing = [c for c in (brief_col, behav_col) if c not in merged.columns]
        if missing:
            raise KeyError(f"{hypo_id}: column(s) {missing} not in the merged data")

        result = test_correlation(
            merged[brief_col].values,
            merged[behav_col].values,
            f"{hypo_id}_direct",
            rng=rng_for_key(f"{hypo_id}_direct", seed)
        )
        result['test_name'] = f"{hypo_id}_{hypo_def['behavioral_measure']}"
        result['var1'] = brief_col
        result['var2'] = behav_col

        results['tests'].append(result)
        results['direct_only'] = True

        print(f"  Direct: {brief_col} × {behav_col}")
        if result['status'] == 'completed':
            print(f"    r = {result['r']:.3f}, p = {result['p']:.4f}, n = {result['n']}")

        return results

    # For H1-H4: Need brain activation data
    task = hypo_def['task']
    contrast = hypo_def['contrast']
    target_rois = hypo_def['target_rois']

    # Get column names early (needed for scatter plots)
    brief_col = hypo_def['brief_measure']
    behav_col = hypo_def['behavioral_measure']

    # Get ROI activation (pre-aggregated)
    roi_data = data[f'roi_{task}']
    brain_agg = aggregate_roi_activation(roi_data, target_rois, contrast)

    if len(brain_agg) == 0:
        print(f"  No brain activation data available")
        results['status'] = 'no_brain_data'
        return results

    print(f"  ROIs: {brain_agg['matched_rois'].values[0]}")

    # Merge brain data with behavioral/assessment data
    test_df = merged.merge(brain_agg[['subject', 'activation']], on='subject', how='inner')

    missing = [c for c in (brief_col, behav_col) if c not in test_df.columns]
    if missing:
        raise KeyError(f"{hypo_id}: column(s) {missing} not in the merged data")

    # Store data for scatter plots
    results['test_df'] = test_df
    results['brief_col'] = brief_col
    results['behav_col'] = behav_col
    results['contrast'] = contrast

    # Test 1: Self-report × Brain
    result1 = test_correlation(
        test_df[brief_col].values,
        test_df['activation'].values,
        f"{hypo_id}_self_brain",
        rng=rng_for_key(f"{hypo_id}_self_brain", seed)
    )
    result1['test_name'] = f"{hypo_id}_self_brain"
    result1['var1'] = brief_col
    result1['var2'] = f"{contrast}_brain"
    results['tests'].append(result1)

    print(f"  Self × Brain: r = {result1['r']:.3f}, p = {result1['p']:.4f}" if result1['status'] == 'completed' else "  Self × Brain: insufficient data")

    # Test 2: Brain × Behavior
    result2 = test_correlation(
        test_df['activation'].values,
        test_df[behav_col].values,
        f"{hypo_id}_brain_behav",
        rng=rng_for_key(f"{hypo_id}_brain_behav", seed)
    )
    result2['test_name'] = f"{hypo_id}_brain_behav"
    result2['var1'] = f"{contrast}_brain"
    result2['var2'] = behav_col
    results['tests'].append(result2)

    print(f"  Brain × Behav: r = {result2['r']:.3f}, p = {result2['p']:.4f}" if result2['status'] == 'completed' else "  Brain × Behav: insufficient data")

    # Test 3: Self-report × Behavior (exploratory)
    result3 = test_correlation(
        test_df[brief_col].values,
        test_df[behav_col].values,
        f"{hypo_id}_self_behav",
        rng=rng_for_key(f"{hypo_id}_self_behav", seed)
    )
    result3['test_name'] = f"{hypo_id}_self_behav"
    result3['var1'] = brief_col
    result3['var2'] = behav_col
    results['tests'].append(result3)

    print(f"  Self × Behav: r = {result3['r']:.3f}, p = {result3['p']:.4f}" if result3['status'] == 'completed' else "  Self × Behav: insufficient data")

    results['status'] = 'completed'
    results['n_subjects'] = len(test_df)

    return results


def apply_fdr_correction(all_results: dict, alpha: float = 0.05):
    """Apply FDR correction to primary tests only."""
    primary_tests = []
    primary_locations = []

    for hypo_id, hypo_result in all_results.items():
        for i, test in enumerate(hypo_result.get('tests', [])):
            test_name = test.get('test_name', '')
            is_primary = is_primary_test(hypo_id, test_name)

            if is_primary and test.get('status') == 'completed' and not np.isnan(test.get('p', np.nan)):
                primary_tests.append(test['p'])
                primary_locations.append((hypo_id, i))

    if len(primary_tests) > 1:
        rejected, p_fdr = fdrcorrection(primary_tests, alpha=alpha)

        for idx, (hypo_id, test_idx) in enumerate(primary_locations):
            all_results[hypo_id]['tests'][test_idx]['p_fdr'] = p_fdr[idx]
            all_results[hypo_id]['tests'][test_idx]['significant_fdr'] = rejected[idx]
    elif len(primary_tests) == 1:
        # Single test: p_fdr = p
        hypo_id, test_idx = primary_locations[0]
        p = all_results[hypo_id]['tests'][test_idx]['p']
        all_results[hypo_id]['tests'][test_idx]['p_fdr'] = p
        all_results[hypo_id]['tests'][test_idx]['significant_fdr'] = (p < alpha)

    return all_results


def check_primary_family(all_results: dict) -> None:
    """Raise unless the corrected confirmatory family has PRIMARY_FAMILY_SIZE tests.

    A missing input or an untestable pair would otherwise shrink the family
    and correct the remaining p-values over fewer tests.
    """
    corrected = [
        test['test_name']
        for hypo_id, hypo_result in all_results.items()
        for test in hypo_result.get('tests', [])
        if is_primary_test(hypo_id, test.get('test_name', '')) and 'p_fdr' in test
    ]
    if len(corrected) != PRIMARY_FAMILY_SIZE:
        raise ValueError(
            f"Confirmatory FDR family has {len(corrected)} tests, expected "
            f"{PRIMARY_FAMILY_SIZE}: {corrected}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ROI-level hypothesis testing for the unified EF model',
        epilog=(
            'Needs BASE_DIR and OUTPUT_DIR (in .env or the environment) and an '
            'analysis_variant.json at the OUTPUT_DIR root (scripts/setup_run_tree.py). '
            'Upstream stages that must have run: assessments (01), behaviour (02), '
            'first-level GLM (03), step2c session info (04b) and step3b (06_roi_activation, run with its default --baseline-only).'
        ),
    )
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='RNG seed for permutation and bootstrap resampling')
    args = parser.parse_args()

    if not OUTPUT_DIR:
        print("Error: OUTPUT_DIR environment variable not set")
        sys.exit(1)
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")

    variant = load_variant()
    hypotheses = resolve_hypotheses(PRIMARY_HYPOTHESES, variant)

    print("\n" + "="*70)
    print("UNIFIED EF MODEL - ROI-LEVEL ANALYSIS")
    print("="*70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("\nROI-level: Uses PRE-AGGREGATED data (mean across channels)")
    print("  - Higher power (one aggregated value per participant)")
    print("  - FDR correction across ~8 primary tests")
    print("  - Confirmatory analysis")
    print(f"\nRNG seed: {args.seed}")
    print(f"Analysis variant: {variant.name}")
    print(f"H2 behavioral measure: {hypotheses['H2']['behavioral_measure']}")

    # Setup
    setup_plotting()
    results_dir = Path(OUTPUT_DIR) / "09_hypothesis_testing" / "unified_ef_model_roi"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_all_data()

    # Run hypothesis tests
    print("\n" + "="*70)
    print("TESTING HYPOTHESES")
    print("="*70)

    all_results = {}

    for hypo_id, hypo_def in hypotheses.items():
        result = test_hypothesis_roi_level(hypo_id, hypo_def, data, seed=args.seed)
        all_results[hypo_id] = result

    # Apply FDR correction to primary tests
    print("\n" + "="*70)
    print("APPLYING FDR CORRECTION (primary tests only)")
    print("="*70)

    all_results = apply_fdr_correction(all_results, alpha=0.05)
    check_primary_family(all_results)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    n_primary = 0
    n_sig = 0

    for hypo_id, result in all_results.items():
        for test in result.get('tests', []):
            test_name = test.get('test_name', '')
            is_primary = is_primary_test(hypo_id, test_name)

            if is_primary:
                n_primary += 1
                if test.get('significant_fdr', False):
                    n_sig += 1

    print(f"\nPrimary tests: {n_primary}")
    print(f"FDR-significant: {n_sig}")

    print(f"\nSignificant findings:")
    print_summary(all_results)

    # Generate outputs
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    print("  [1/4] Creating forest plot...")
    create_forest_plot(all_results, results_dir / 'forest_plot.png', level='roi')

    print("  [2/4] Creating scatter plots...")
    for hypo_id, hypo_def in hypotheses.items():
        result = all_results[hypo_id]
        if 'test_df' in result and len(result['test_df']) > 0:
            create_hypothesis_scatter_plots(
                hypo_id, hypo_def, result['test_df'],
                result['brief_col'], result['behav_col'],
                result['contrast'], results_dir, level='roi',
                rng=rng_for_key(f"{hypo_id}|scatter", args.seed),
                tests=result.get('tests', []), seed=args.seed
            )

    print("  [3/4] Saving results...")
    save_results(all_results, results_dir, level='roi', seed=args.seed,
                 variant_name=variant.name, h2_measure=hypotheses['H2']['behavioral_measure'])

    print("  [4/4] Saving merged dataset...")
    data['merged'].to_csv(results_dir / 'merged_data.csv', index=False)

    print("\n" + "="*70)
    print("ROI-LEVEL ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")

    return all_results


if __name__ == '__main__':
    main()
