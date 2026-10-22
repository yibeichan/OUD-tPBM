#!/usr/bin/env python3
"""
Utility functions for hypothesis testing scripts.

Contains common data loading, statistical testing, and visualization functions
used across all hypothesis testing analyses.
"""

import os
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from subject_ids import standardize_subject_id  # noqa: E402
from analysis_variant import load_variant  # noqa: E402
from brief_validity import valid_administration_mask  # noqa: E402
from session_exclusions import excluded_sessions  # noqa: E402

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def _require_env() -> None:
    """Raise unless BASE_DIR and OUTPUT_DIR are set; called by each loader."""
    if not BASE_DIR or not OUTPUT_DIR:
        raise RuntimeError("BASE_DIR and OUTPUT_DIR must be set (in .env or the environment)")


# Set paths (None until OUTPUT_DIR is set; the loaders check first)
BEHAV_DIR = Path(OUTPUT_DIR) / "02_behav" if OUTPUT_DIR else None
ASSESS_DIR = Path(OUTPUT_DIR) / "01_assessments" / "scores" if OUTPUT_DIR else None

# scripts/seeding.py is the single definition of the seeding helpers;
# they are re-exported from here.
from seeding import DEFAULT_SEED, rng_for_key, _resolve_rng  # noqa: E402,F401


def setup_plotting():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")


def load_behavioral_data():
    """Load baseline behavioral data (task_order=1) for hypothesis testing."""
    _require_env()

    # Load task info to identify baseline sessions
    stroop_task_info = pd.read_csv(Path(BASE_DIR) / 'data' / 'stroop_task_info.csv')
    wcst_task_info = pd.read_csv(Path(BASE_DIR) / 'data' / 'wcst_task_info.csv')
    
    # Get baseline sessions (task_order = 1) for each subject
    stroop_baseline = stroop_task_info[stroop_task_info['task_order'] == 1]
    wcst_baseline = wcst_task_info[wcst_task_info['task_order'] == 1]
    
    print(f"Loading baseline behavioral data:")
    print(f"  Stroop baseline sessions: {len(stroop_baseline)}")
    print(f"  WCST baseline sessions: {len(wcst_baseline)}")

    # Sessions the run's analysis_variant.json excludes for low accuracy.
    # Empty unless exclude_low_accuracy is set, so a variant that keeps
    # everything in reproduces the pre-exclusion behavior exactly.
    stroop_excluded = excluded_sessions('stroop')
    wcst_excluded = excluded_sessions('wcst')

    # Initialize storage
    stroop_data = []
    wcst_data = []

    # Load Stroop baseline data
    stroop_dir = BEHAV_DIR / "stroop"
    for _, row in stroop_baseline.iterrows():
        subj_id = row['subj_id']
        task_id = row['task_id']

        if task_id in stroop_excluded:
            print(f"  Skipping Stroop session {task_id} ({subj_id}): excluded by analysis_variant")
            continue

        # Construct expected JSON filename
        json_file = stroop_dir / f"{subj_id}_{task_id}_results.json"

        # Every baseline session in task_info is expected to have results;
        # a missing or malformed file would drop the subject from every test.
        if not json_file.exists():
            raise FileNotFoundError(f"Stroop baseline file not found: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        metrics = data['metrics']
        subject_id = standardize_subject_id(subj_id)
        
        # Behavioral metrics written by scripts/behav/01_stroop_behav.py
        stroop_entry = {
            'subject_id': subject_id,
            'task_id': task_id,
            # Core performance metrics
            'overall_accuracy': metrics.get('overall_accuracy'),
            'mean_rt_congruent': metrics.get('cong_in_mixed_rt'),
            'mean_rt_incongruent': metrics.get('incong_in_mixed_rt'),
            'mean_rt_neutral': metrics.get('cong_in_cong_only_rt'),
            'accuracy_congruent': metrics.get('cong_in_mixed_accuracy'),
            'accuracy_incongruent': metrics.get('incong_in_mixed_accuracy'),
            'accuracy_neutral': metrics.get('cong_in_cong_only_accuracy'),
            # Interference effects (theoretically meaningful names)
            'interference': metrics.get('stroop_effect_rt'),  # Primary: H1 Inhibition construct
            'mixing_cost': metrics.get('mixing_cost_rt'),     # H4 switching cost
            # Variability and trial-sequence metrics
            'rt_variability': metrics.get('overall_rt_cv'),    # Primary: H4 Attentional Stability construct
            'congruent_rt_cv': metrics.get('congruent_rt_cv'),
            'incongruent_rt_cv': metrics.get('incongruent_rt_cv'),
            'conflict_adaptation_rt': metrics.get('conflict_adaptation_cost_rt'),
            'max_consecutive_errors': metrics.get('max_consecutive_errors'),
            # The same metrics under their behavioural-script names
            'stroop_effect': metrics.get('stroop_effect_rt'),
            'overall_rt_cv': metrics.get('overall_rt_cv'),
            'overall_rt_sd': metrics.get('overall_rt_sd')
        }
        stroop_data.append(stroop_entry)
    
    # Load WCST baseline data
    wcst_dir = BEHAV_DIR / "wcst"
    for _, row in wcst_baseline.iterrows():
        subj_id = row['subj_id']
        task_id = row['task_id']

        if task_id in wcst_excluded:
            print(f"  Skipping WCST session {task_id} ({subj_id}): excluded by analysis_variant")
            continue

        # Construct expected JSON filename
        json_file = wcst_dir / f"{subj_id}_{task_id}_results.json"

        if not json_file.exists():
            raise FileNotFoundError(f"WCST baseline file not found: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        metrics = data['metrics']
        subject_id = standardize_subject_id(subj_id)
        
        # Behavioral metrics written by scripts/behav/01_wcst_behav.py
        wcst_entry = {
            'subject_id': subject_id,
            'task_id': task_id,
            # Core performance metrics
            'overall_accuracy': metrics.get('overall_accuracy'),
            'exp_accuracy': metrics.get('exp_accuracy'),
            'mean_rt': metrics.get('mean_rt'),
            'exp_mean_rt': metrics.get('exp_mean_rt'),
            # Rule learning metrics
            'total_rule_changes': metrics.get('total_rule_changes'),
            'avg_trials_between_rules': metrics.get('avg_trials_between_rules'),
            # Read directly: a fallback would substitute a different
            # quantity (H5 uses prop_rule_periods_mastered).
            'categories_completed': metrics['categories_completed'],
            'prop_rule_periods_mastered': metrics['prop_rule_periods_mastered'],
            'trials_to_first_category': metrics['trials_to_first_category'],
            # Ambiguity effects
            'high_ambiguity_accuracy': metrics.get('high_ambiguity_accuracy'),
            'low_ambiguity_accuracy': metrics.get('low_ambiguity_accuracy'),
            # Perseveration, set-shifting and maintenance metrics
            'mean_persev_run_length': metrics.get('mean_persev_run_length'),
            'max_persev_run_length': metrics.get('max_persev_run_length'),
            'total_persev_errors': metrics.get('total_persev_errors'),
            'set_shifting_efficiency': metrics.get('set_shifting_efficiency'),
            'mean_trials_to_learn_rule': metrics.get('mean_trials_to_learn_rule'),
            'wm_maintenance_decline': metrics.get('wm_maintenance_decline'),
            'exp_rt_cv': metrics.get('exp_rt_cv'),
            # Heaton WCST perseverative errors (proper scoring)
            'perseverative_errors': metrics.get('perseverative_errors'),
            'perseverative_error_rate': metrics.get('perseverative_error_rate'),
            'non_perseverative_errors': metrics.get('non_perseverative_errors'),
            'total_errors': metrics.get('total_errors')
        }
        wcst_data.append(wcst_entry)
    
    # Convert to DataFrames
    stroop_df = pd.DataFrame(stroop_data)
    wcst_df = pd.DataFrame(wcst_data)
    
    print(f"\nLoaded baseline behavioral data:")
    print(f"  Stroop subjects: {len(stroop_df)}")
    print(f"  WCST subjects: {len(wcst_df)}")
    
    return stroop_df, wcst_df


def load_assessment_data():
    """Load baseline assessment scores with inconsistency filtering."""
    _require_env()

    # Load executive function T-scores (baseline only)
    # T-scores file contains both raw scores and T-scores (e.g., Inhibit and Inhibit_t)
    TSCORE_DIR = Path(OUTPUT_DIR) / "01_assessments" / "t-scores"
    ef_df = pd.read_csv(TSCORE_DIR / "executive_function_baseline_subscale_t-scores.csv")
    ef_df.rename(columns={'subject': 'subject_id'}, inplace=True)

    variant = load_variant()
    if variant.exclude_invalid_brief:
        valid = valid_administration_mask(ef_df)
        blanked_subjects = sorted(ef_df.loc[~valid, 'subject_id'])
        print(f"BRIEF-A validity: blanking Inhibit_t/Shift_t/Working Memory_t/GEC_t for {blanked_subjects}")
        blanked_cols = ['Inhibit_t', 'Shift_t', 'Working Memory_t', 'GEC_t']
        ef_df.loc[~valid, blanked_cols] = np.nan

    # Load other assessments
    phq9_df = pd.read_csv(ASSESS_DIR / "phq9_scores.csv")
    phq9_df = phq9_df[phq9_df['timepoint'] == 'baseline'][['subject', 'total_score', 'severity']]
    phq9_df.rename(columns={'subject': 'subject_id'}, inplace=True)
    phq9_df.rename(columns={'total_score': 'phq9_total'}, inplace=True)
    
    gad7_df = pd.read_csv(ASSESS_DIR / "gad7_scores.csv")
    gad7_df = gad7_df[gad7_df['timepoint'] == 'baseline'][['subject', 'total_score', 'severity']]
    gad7_df.rename(columns={'subject': 'subject_id'}, inplace=True)
    gad7_df.rename(columns={'total_score': 'gad7_total'}, inplace=True)
    
    craving_df = pd.read_csv(ASSESS_DIR / "craving_scores.csv")
    craving_df = craving_df[craving_df['timepoint'] == 'baseline'][['subject', 'total_score']]
    craving_df.rename(columns={'subject': 'subject_id'}, inplace=True)
    craving_df.rename(columns={'total_score': 'craving_total'}, inplace=True)
    
    # Merge all assessments
    assess_df = ef_df.copy()
    assess_df = assess_df.merge(phq9_df[['subject_id', 'phq9_total']], on='subject_id', how='left')
    assess_df = assess_df.merge(gad7_df[['subject_id', 'gad7_total']], on='subject_id', how='left')
    assess_df = assess_df.merge(craving_df[['subject_id', 'craving_total']], on='subject_id', how='left')
    
    # Quality control: Report inconsistency scores
    print(f"\nQuality Control - Inconsistency Scores:")
    print(f"Total subjects: {len(assess_df)}")

    # Column name is lowercase in the T-scores file
    inconsistency_col = 'inconsistency_acceptable'
    if inconsistency_col in assess_df.columns:
        # Convert string 'TRUE'/'FALSE' to boolean if needed
        if assess_df[inconsistency_col].dtype == object:
            assess_df[inconsistency_col] = assess_df[inconsistency_col].str.upper() == 'TRUE'

        n_acceptable = assess_df[inconsistency_col].sum()
        n_non_acceptable = len(assess_df) - n_acceptable
        print(f"Acceptable inconsistency (≤7): {n_acceptable}")
        print(f"Non-acceptable inconsistency (>7): {n_non_acceptable}")

        if n_non_acceptable > 0:
            print("⚠️  WARNING: Some subjects have non-acceptable inconsistency scores")
            non_acceptable = assess_df[~assess_df[inconsistency_col]]
            print(f"Subjects with high inconsistency: {list(non_acceptable['subject_id'])}")
            print("(Invalid BRIEF-A administrations are blanked by the analysis variant, not by this flag.)")
        else:
            print("✓ All subjects have acceptable inconsistency scores")
    else:
        print("Warning: inconsistency_acceptable column not found")
    
    return assess_df


def permutation_correlation(x, y, n_permutations=5000, *, rng=None):
    """Compute correlation with permutation-based p-value and bootstrap CI.

    Uses Spearman correlation for robustness with small samples (n=12).

    ``rng`` defaults to a generator on DEFAULT_SEED, so the p-value and CI are
    reproducible whether or not the caller threads a generator down.
    """
    rng = _resolve_rng(rng)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return np.nan, np.nan, np.nan, np.nan

    # Observed correlation (Spearman for robustness)
    r_obs, _ = stats.spearmanr(x_clean, y_clean)

    # Permutation test
    r_perms = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_clean)
        r_perm, _ = stats.spearmanr(x_clean, y_perm)
        r_perms.append(r_perm)

    r_perms = np.array(r_perms)
    n_exceed = np.sum(np.abs(r_perms) >= np.abs(r_obs))
    p_value = (1 + n_exceed) / (1 + n_permutations)

    # Bootstrap confidence interval
    n_bootstrap = 1000
    r_boots = []
    n = len(x_clean)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        r_boot, _ = stats.spearmanr(x_clean[idx], y_clean[idx])
        r_boots.append(r_boot)

    ci_lower = np.percentile(r_boots, 2.5)
    ci_upper = np.percentile(r_boots, 97.5)

    return r_obs, p_value, ci_lower, ci_upper


def create_scatter_plot(x_data, y_data, x_label, y_label, subjects, title="", save_path=None,
                        *, rng=None, stats=None):
    """Create scatter plot with regression line and statistics.

    ``stats`` (keys ``r``, ``p``, ``ci_lower``, ``ci_upper``, ``n``) is the
    reported test the panel illustrates, and is printed as given: recomputing
    it here with other iteration counts would print a p and CI that differ
    from the results table. Without it the annotation falls back to
    ``permutation_correlation`` on ``rng``.
    """
    rng = _resolve_rng(rng)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Remove NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    subjects_clean = subjects[mask]
    
    # Scatter plot
    ax.scatter(x_clean, y_clean, s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add subject labels
    for i, subj in enumerate(subjects_clean):
        ax.annotate(subj, (x_clean[i], y_clean[i]), 
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, alpha=0.8, fontweight='bold')
    
    # Fit regression line if enough data
    if len(x_clean) > 2:
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_range, p(x_range), "red", alpha=0.8, linewidth=3, linestyle='--')
        
        # Add confidence interval
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(x_clean.reshape(-1, 1), y_clean)
        y_pred = lr.predict(x_range.reshape(-1, 1))
        
        # Compute standard error
        residuals = y_clean - lr.predict(x_clean.reshape(-1, 1))
        se = np.sqrt(np.sum(residuals**2) / (len(x_clean) - 2))
        
        # Confidence interval
        ci = 1.96 * se  # 95% CI
        ax.fill_between(x_range, y_pred - ci, y_pred + ci, alpha=0.2, color='red')
        
        # Add statistics
        if stats is not None:
            r, p = stats['r'], stats['p']
            ci_low, ci_high, n_shown = stats['ci_lower'], stats['ci_upper'], stats['n']
        else:
            r, p, ci_low, ci_high = permutation_correlation(x_clean, y_clean, rng=rng)
            n_shown = len(x_clean)
        sig_stars = get_significance_stars(p)
        
        stats_text = f'r = {r:.3f}{sig_stars}\np = {p:.3f}\n95% CI: [{ci_low:.3f}, {ci_high:.3f}]\nn = {n_shown}'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                fontsize=12, fontweight='bold')
    
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""
