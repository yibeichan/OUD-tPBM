#!/usr/bin/env python3
"""
Unified EF Model - Channel-Level Analysis (Exploratory)

This script tests the 5 primary hypotheses at the CHANNEL level only.
Channel-level uses RAW per-channel data (NO aggregation).

ROI DEFINITION (HARMONIZED WITH LONGITUDINAL ANALYSIS):
- Uses sub-regional ROIs for theoretical precision (e.g., SalVentAttn_PFCl, Cont_PFCl, Cont_Par)
- Sub-regional approach preserves effect sizes better than network-level aggregation
- With n=12 cross-sectional and n=5 longitudinal, theoretical precision > statistical convenience
- Channel-level tests each channel individually, then filters to target ROIs

OPTIMIZATIONS:
- Data caching to avoid re-loading GLM files
- Progress reporting for long operations
- Vectorized operations where possible
- Bootstrap 1,000 and permutations 50,000 per test (ROI level: 5,000 and 10,000)

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
from itertools import product

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'brain'))

from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
# OUTPUT_DIR is validated in main(); module is importable without it so that
# downstream consumers (tests) can import this module.

# Import common utilities
from unified_ef_model_common import (
    DEFAULT_SEED, rng_for_key, _resolve_rng, PRIMARY_HYPOTHESES,
    load_all_data, load_channel_data, test_correlation, create_forest_plot,
    save_results, print_summary, create_hypothesis_scatter_plots,
    resolve_hypotheses
)

from utils.statistical_methods import setup_plotting
from analysis_variant import load_variant, refuse_protected_tree


# =============================================================================
# CHANNEL-LEVEL ANALYSIS
# =============================================================================

def get_matched_channels(channel_df: pd.DataFrame, target_rois: list) -> tuple:
    """Get channels that match target ROIs."""
    matched_rois = []
    for target in target_rois:
        for roi in channel_df['network_roi'].unique():
            if target.lower() in str(roi).lower():
                matched_rois.append(roi)

    matched_rois = list(set(matched_rois))

    if not matched_rois:
        return [], []

    filtered = channel_df[channel_df['network_roi'].isin(matched_rois)].copy()
    unique_channels = filtered['channel'].unique()

    return unique_channels, matched_rois


def test_correlation_fast(x: np.ndarray, y: np.ndarray,
                          n_perm: int = 50000, n_boot: int = 1000, *, rng=None) -> dict:
    """Fast correlation test (for exploratory channel-level).

    With the add-one p-value estimator, ``n_perm`` permutations cannot
    resolve below 1/(n_perm+1). The default, 50000, puts that floor (2e-5)
    below the rank-1 Benjamini-Hochberg threshold of the smallest hypothesis
    family (0.05 divided by that family's number of channel tests), so a
    channel test can survive FDR.
    ``rng`` defaults to a generator on DEFAULT_SEED so a bare call is still
    reproducible.
    """
    rng = _resolve_rng(rng)

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)

    if n < 5:
        return {
            'r': np.nan, 'p': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'n': n, 'status': 'insufficient_data'
        }

    # Spearman correlation
    r, p_p = stats.spearmanr(x, y)

    # A constant variable gives rho = NaN, and the permutation count would
    # then be 0, i.e. the smallest possible p. There is no correlation to test.
    if np.isnan(r):
        return {
            'r': np.nan, 'p': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'n': n, 'status': 'insufficient_data'
        }

    # Permutation p-value
    p_perms = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    n_exceed = np.sum(np.abs(p_perms) >= np.abs(r))
    p_value = (1 + n_exceed) / (1 + n_perm)

    # Bootstrap CI (n_boot defaults to 1000, for speed)
    # Resample PAIRS together (same indices for x and y) to preserve relationship
    boot_rs = np.array([stats.spearmanr(
        x[idx], y[idx]
    )[0] for idx in [rng.choice(n, n, replace=True) for _ in range(n_boot)]])

    ci_lower = np.percentile(boot_rs, 2.5)
    ci_upper = np.percentile(boot_rs, 97.5)

    return {
        'r': float(r), 'p': float(p_value),
        'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper),
        'n': n, 'status': 'completed'
    }


def test_hypothesis_channel_level(hypo_id: str, hypo_def: dict,
                                  data: dict, channel_cache: dict,
                                  *, seed: int = DEFAULT_SEED) -> dict:
    """Test a hypothesis at CHANNEL level.

    KEY: Uses RAW per-channel data (NO aggregation).
    Tests EACH channel separately.

    Args:
        hypo_id: Hypothesis ID
        hypo_def: Hypothesis definition
        data: Data dictionary with merged data
        channel_cache: Cache for channel data
        seed: Base seed; each channel test derives its own generator from the
            channel's identity, so a channel's p-value does not depend on how
            many channels were tested before it

    Returns:
        Results dictionary with channel-level tests
    """
    print(f"\n{'='*60}")
    print(f"{hypo_id}: {hypo_def['name']}")
    print(f"{'='*60}")

    results = {
        'hypothesis_id': hypo_id,
        'name': hypo_def['name'],
        'tests': [],
        'n_channels_tested': 0
    }

    # H5 is not tested at channel level (no brain measure)
    if hypo_id == 'H5':
        print(f"  Skipped (no brain measure)")
        results['status'] = 'not_applicable'
        return results

    merged = data['merged']
    task = hypo_def['task']
    contrast = hypo_def['contrast']
    target_rois = hypo_def['target_rois']

    # Get column names early (needed for scatter plots)
    brief_col = hypo_def['brief_measure']
    behav_col = hypo_def['behavioral_measure']

    # Load RAW channel data (with caching)
    print(f"  Loading channel data for {task} - {contrast}...")
    channel_data = load_channel_data(task, contrast, cache=channel_cache)

    if len(channel_data) == 0:
        print(f"  No channel data available")
        results['status'] = 'no_channel_data'
        return results

    # Filter to target ROIs
    unique_channels, matched_rois = get_matched_channels(channel_data, target_rois)

    if len(unique_channels) == 0:
        print(f"  No channels match target ROIs")
        results['status'] = 'no_matching_channels'
        return results

    print(f"  Testing {len(unique_channels)} channels...")

    # Load ROI data for scatter plots (uses aggregated data, same as ROI-level)
    roi_data = data.get(f'roi_{task}', pd.DataFrame())
    scatter_df = pd.DataFrame()

    if len(roi_data) > 0:
        # Aggregate ROI activation (same as ROI-level)
        from unified_ef_model_roi import aggregate_roi_activation
        roi_brain_agg = aggregate_roi_activation(roi_data, target_rois, contrast)

        if len(roi_brain_agg) > 0:
            scatter_df = merged.merge(roi_brain_agg[['subject', 'activation']], on='subject', how='inner')

            missing = [c for c in (brief_col, behav_col) if c not in scatter_df.columns]
            if missing:
                raise KeyError(f"{hypo_id}: column(s) {missing} not in the merged data")

    # Store for scatter plots
    results['test_df'] = scatter_df
    results['brief_col'] = brief_col
    results['behav_col'] = behav_col
    results['contrast'] = contrast

    # Test each channel
    channel_results = []
    test_types = ['self_brain', 'brain_behav']

    for i, channel in enumerate(unique_channels):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{len(unique_channels)}...")

        ch_data = channel_data[channel_data['channel'] == channel].copy()
        network_roi = ch_data['network_roi'].iloc[0]

        # Merge with behavioral/assessment data
        ch_df = merged.merge(ch_data[['subject', 'activation']], on='subject', how='inner')

        if len(ch_df) < 5:
            continue

        # Test 1: Self-report × Brain
        result1 = test_correlation_fast(
            ch_df[brief_col].values,
            ch_df['activation'].values,
            rng=rng_for_key(f"{hypo_id}|{channel}|self_brain", seed)
        )
        result1['test_name'] = f"{hypo_id}_{channel}_self_brain"
        result1['var1'] = brief_col
        result1['var2'] = network_roi
        result1['channel'] = channel
        result1['network_roi'] = network_roi
        result1['test_type'] = 'self_brain'

        # Test 2: Brain × Behavior
        result2 = test_correlation_fast(
            ch_df['activation'].values,
            ch_df[behav_col].values,
            rng=rng_for_key(f"{hypo_id}|{channel}|brain_behav", seed)
        )
        result2['test_name'] = f"{hypo_id}_{channel}_brain_behav"
        result2['var1'] = network_roi
        result2['var2'] = behav_col
        result2['channel'] = channel
        result2['network_roi'] = network_roi
        result2['test_type'] = 'brain_behav'

        channel_results.extend([result1, result2])

    results['tests'] = channel_results
    # Channels with fewer than five participants are skipped above, so the
    # count of channels tested is the channels with a completed test, not the
    # candidates in the target ROIs. The candidate count is kept
    # under its own key.
    results['n_candidate_channels'] = len(unique_channels)
    results['n_channels_tested'] = len(
        {r['channel'] for r in channel_results if r['status'] == 'completed'})

    # Apply FDR correction across ALL channel tests for this hypothesis
    p_values = [r['p'] for r in channel_results if r['status'] == 'completed' and not np.isnan(r['p'])]

    if len(p_values) > 1:
        rejected, p_fdr = fdrcorrection(p_values, alpha=0.05)

        # Add FDR results back
        fdr_idx = 0
        for r in channel_results:
            if r['status'] == 'completed' and not np.isnan(r['p']):
                r['p_fdr'] = p_fdr[fdr_idx]
                r['significant_fdr'] = rejected[fdr_idx]
                fdr_idx += 1

    # Report significant channels
    sig_channels = [r for r in channel_results if r.get('significant_fdr', False)]

    print(f"  Results: {len(channel_results)} tests, {len(sig_channels)} FDR-significant")

    if sig_channels:
        print(f"  Significant channels:")
        for r in sig_channels:
            print(f"    {r['channel']} ({r['network_roi']}): r={r['r']:.3f}, p_fdr={r['p_fdr']:.4f}")

    results['status'] = 'completed'
    results['n_significant'] = len(sig_channels)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Channel-level hypothesis testing for the unified EF model',
        epilog=(
            'Needs BASE_DIR and OUTPUT_DIR (in .env or the environment) and an '
            'analysis_variant.json at the OUTPUT_DIR root (scripts/setup_run_tree.py). '
            'Upstream stages that must have run: assessments (01), behaviour (02), '
            'first-level GLM (03), step2c session info (04b) and step2a/2b (04_channel_mappings) and step3a.'
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
    print("UNIFIED EF MODEL - CHANNEL-LEVEL ANALYSIS")
    print("="*70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("\nChannel-level: Uses RAW per-channel data (NO aggregation)")
    print("  - Lower power (one value per participant per channel)")
    print("  - FDR correction within each hypothesis (H1-H4)")
    print("  - Exploratory analysis")
    print("\nOptimizations:")
    print("  - Data caching to avoid re-loading GLM files")
    print("  - 1,000 bootstrap resamples and 50,000 permutations per test")
    print("  - Progress reporting for long operations")
    print(f"\nRNG seed: {args.seed}")
    print(f"Analysis variant: {variant.name}")
    print(f"H2 behavioral measure: {hypotheses['H2']['behavioral_measure']}")

    # Setup
    setup_plotting()
    results_dir = Path(OUTPUT_DIR) / "09_hypothesis_testing" / "unified_ef_model_channel"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_all_data()

    # Create channel cache
    channel_cache = {}

    # Run hypothesis tests
    print("\n" + "="*70)
    print("TESTING HYPOTHESES")
    print("="*70)

    all_results = {}
    total_channels = 0
    total_sig = 0

    for hypo_id, hypo_def in hypotheses.items():
        result = test_hypothesis_channel_level(hypo_id, hypo_def, data, channel_cache,
                                               seed=args.seed)
        all_results[hypo_id] = result
        total_channels += result.get('n_channels_tested', 0)
        total_sig += result.get('n_significant', 0)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\nTotal channels tested: {total_channels}")
    print(f"Total FDR-significant findings: {total_sig}")

    print(f"\nSignificant findings:")
    print_summary(all_results)

    # Generate outputs
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)

    print("  [1/4] Creating forest plot (top 100 shown)...")
    create_forest_plot(all_results, results_dir / 'forest_plot.png', level='channel')

    print("  [2/4] Creating scatter plots...")
    for hypo_id, hypo_def in hypotheses.items():
        result = all_results[hypo_id]
        if 'test_df' in result and len(result['test_df']) > 0:
            create_hypothesis_scatter_plots(
                hypo_id, hypo_def, result['test_df'],
                result['brief_col'], result['behav_col'],
                result['contrast'], results_dir, level='channel',
                rng=rng_for_key(f"{hypo_id}|scatter", args.seed),
                seed=args.seed
            )

    print("  [3/4] Saving results...")
    save_results(all_results, results_dir, level='channel', seed=args.seed,
                 variant_name=variant.name, h2_measure=hypotheses['H2']['behavioral_measure'])

    print("  [4/4] Saving merged dataset...")
    data['merged'].to_csv(results_dir / 'merged_data.csv', index=False)

    print("\n" + "="*70)
    print("CHANNEL-LEVEL ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")

    return all_results


if __name__ == '__main__':
    main()
