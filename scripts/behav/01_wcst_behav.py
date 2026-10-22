import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_variant import refuse_protected_tree  # noqa: E402
from wcst_rules import actual_rule_changes, check_trial_order, learning_end_position  # noqa: E402

# Target card properties for traditional perseverative error calculation
# These are the 4 reference cards in this WCST implementation
# response_pressed values map to keyboard keys/positions
TARGET_CARDS = {
    0.0: {'color': 'orange', 'shape': 'cross', 'number': 3.0},   # Key 0
    7.0: {'color': 'red', 'shape': 'circle', 'number': 0.0},     # Key 7
    8.0: {'color': 'green', 'shape': 'square', 'number': 1.0},   # Key 8
    9.0: {'color': 'blue', 'shape': 'star', 'number': 2.0},      # Key 9
}

# Rule to stimulus column mapping
RULE_TO_STIMULUS = {
    'color': 'stimulus_card_color',
    'shape': 'stimulus_card_shape',
    'number': 'stimulus_card_number',
}


def would_response_be_correct(response_pressed: float, stimulus_row: pd.Series, rule: str) -> bool:
    """
    Determine if a response would be correct under a given sorting rule.
    
    A response is correct if the chosen target card matches the stimulus card
    on the dimension specified by the current rule.
    
    Parameters:
    -----------
    response_pressed : float
        The button/key pressed (maps to a target card: 0, 7, 8, or 9)
    stimulus_row : pd.Series
        Row containing stimulus card properties
    rule : str
        The sorting rule ('color', 'shape', or 'number')
        
    Returns:
    --------
    bool : True if response would be correct under the given rule
    """
    if response_pressed not in TARGET_CARDS:
        return False
    
    target = TARGET_CARDS[response_pressed]
    
    if rule == 'color':
        return target['color'] == stimulus_row['stimulus_card_color']
    elif rule == 'shape':
        return target['shape'] == stimulus_row['stimulus_card_shape']
    elif rule == 'number':
        return target['number'] == stimulus_row['stimulus_card_number']
    else:
        return False


class WCSTEventAnalyzer:
    def __init__(self, events_df: pd.DataFrame):
        self.events_df = self._clean_events(events_df.copy())
        self.trial_data = None

    def _clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Trim whitespace on object columns to avoid string/float comparison issues
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        numeric_cols = {
            'timestamp': 'float',
            'trial': 'Int64',
            'response_RT': 'float',
            'response_outcome': 'Int64',
            'response_correct_streak': 'Int64',
            'stimulus_ambiguity_level': 'Int64',
            'response_pressed': 'float',
            'stimulus_card_number': 'float'
        }

        for col, dtype in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if dtype == 'Int64':
                    df[col] = df[col].round().astype('Int64')

        return df
        
    def process_trials(self) -> pd.DataFrame:
        """Process trial events into a clean format for visualization."""
        # Filter for start_trial events only
        trial_events = self.events_df[self.events_df['event'] == 'start_trial'].copy()
        
        # Create trial data with required columns
        # Note: response_outcome NaN (no response) is treated as incorrect (0) for 'correct' column
        # Rationale: Inability to respond reflects cognitive control failure, conceptually aligned
        # with task performance deficits. 'correct' feeds accuracy and consecutive errors. The
        # perseverative scoring (get_performance_metrics) needs a choice to compare with the
        # previous rule, so it does not count a trial with no response as an error, and
        # total_errors excludes those trials too.
        trial_data = pd.DataFrame({
            'time': trial_events['timestamp'],
            'block_type': trial_events['block_type'],
            'trial_number': trial_events['trial'],
            'ambiguity_level': trial_events['stimulus_ambiguity_level'],
            'correct': (trial_events['response_outcome'] == 1).fillna(False).astype(int),
            'response_time': trial_events['response_RT'],
            'correct_streak': trial_events['response_correct_streak'],
        })
        # A rule change is an experimental trial whose sorting rule differs
        # from the previous experimental trial's, not the table's rule_change
        # flag, which also marks block starts that keep the rule. Shared with the GLM labelling (scripts/wcst_rules.py).
        exp_events_unsorted = trial_events[trial_events['block_type'] == 'experimental']
        # The GLM orders trials by timestamp; refuse a table where that order
        # differs from trial-number order, or the stages would disagree.
        check_trial_order(exp_events_unsorted)
        exp_order = exp_events_unsorted.sort_values('trial', kind='stable')
        trial_data['rule_change'] = 0
        if not exp_order.empty:
            changes = actual_rule_changes(exp_order['trial_type'])
            trial_data.loc[changes.index, 'rule_change'] = changes.astype(int)

        # Ensure numeric types for downstream comparisons
        trial_data['response_time'] = pd.to_numeric(trial_data['response_time'], errors='coerce')
        trial_data['trial_number'] = pd.to_numeric(trial_data['trial_number'], errors='coerce')
        trial_data['trial_number'] = trial_data['trial_number'].round().astype('Int64')
        trial_data['ambiguity_level'] = pd.to_numeric(trial_data['ambiguity_level'], errors='coerce').round().astype('Int64')
        trial_data['correct_streak'] = pd.to_numeric(trial_data['correct_streak'], errors='coerce').round().astype('Int64')
        
        # Find control block regions
        trial_data['control_block'] = (trial_data['block_type'] == 'control').fillna(False).astype(int)
        
        # Add block transition markers
        trial_data['block_transition'] = (
            trial_data['block_type'] != trial_data['block_type'].shift(1)
        ).fillna(False).astype(int)
        
        self.trial_data = trial_data
        return trial_data
    
    def get_control_blocks(self) -> List[Dict[str, float]]:
        """Get time ranges for control blocks."""
        if self.trial_data is None:
            self.process_trials()
            
        control_blocks = []
        current_block = None
        
        for _, row in self.trial_data.iterrows():
            if row['control_block'] == 1:
                if current_block is None:
                    current_block = {'start': row['time']}
            elif current_block is not None:
                current_block['end'] = self.trial_data.loc[
                    self.trial_data.index[self.trial_data.index.get_loc(row.name) - 1],
                    'time'
                ]
                control_blocks.append(current_block)
                current_block = None
                
        # Handle case where last block is control
        if current_block is not None:
            current_block['end'] = self.trial_data['time'].iloc[-1]
            control_blocks.append(current_block)
            
        return control_blocks
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        if self.trial_data is None:
            self.process_trials()
            
        exp_trials = self.trial_data[self.trial_data['block_type'] == 'experimental']
        
        metrics = {
            'overall_accuracy': self.trial_data['correct'].mean(),
            'exp_accuracy': exp_trials['correct'].mean(),
            'mean_rt': self.trial_data['response_time'].mean(),
            'exp_mean_rt': exp_trials['response_time'].mean(),
            'total_rule_changes': exp_trials['rule_change'].sum(),
            'avg_trials_between_rules': len(exp_trials) / max(1, exp_trials['rule_change'].sum()),
            'high_ambiguity_accuracy': exp_trials[exp_trials['ambiguity_level'] == 2]['correct'].mean(),
            'low_ambiguity_accuracy': exp_trials[exp_trials['ambiguity_level'] == 1]['correct'].mean()
        }
        
        # COMBINED METRIC CALCULATIONS (FMS + PERSEVERATIVE ERRORS)
        # Using vectorized operations for efficiency
        if len(exp_trials) > 10:
            exp_sorted = exp_trials.sort_values(['trial_number']).copy()

            # FAILURES TO MAINTAIN SET (FMS) - Standard WCST Metric
            # Definition: Breaking a correct streak (≥5 consecutive correct, then an error)
            # Interpretation: Working-memory or sustained attention failure
            # Note: FMS requires sequential logic, cannot be fully vectorized
            fms_count = 0
            consecutive_correct = 0

            for _, row in exp_sorted.iterrows():
                if row['rule_change']:
                    consecutive_correct = 0  # Rule change resets the streak
                elif row['correct'] == 1:
                    consecutive_correct += 1  # Correct response - increment streak
                else:
                    # Error - check if this breaks a maintained set
                    if consecutive_correct >= 5:
                        fms_count += 1
                    consecutive_correct = 0

            metrics['failures_to_maintain_set'] = fms_count

            # PERSEVERATIVE ERRORS - Standard WCST Heaton Scoring
            # Definition: An error where the response would have been CORRECT under 
            # the PREVIOUS sorting rule. This captures "stuck in set" behavior.
            # 
            # Algorithm:
            # 1. Track the previous sorting rule before each rule change
            # 2. For each error after a rule change, check if the response would
            #    have been correct under the OLD (previous) rule
            # 3. If yes → perseverative error; if no → non-perseverative error
            #
            # A rule change requires a different sorting rule, so the
            # "previous rule" is the last *different* rule, the previously
            # correct principle, even across a block start that keeps the
            # current rule.
            #
            # Reference: Heaton et al. (1993) Wisconsin Card Sorting Test Manual
            
            # Need original events data for trial_type, response_pressed, and stimulus properties
            exp_events = self.events_df[
                (self.events_df['event'] == 'start_trial') & 
                (self.events_df['block_type'] == 'experimental')
            ].copy()
            exp_events = exp_events.sort_values('trial', kind='stable')
            exp_rule_changes = actual_rule_changes(exp_events['trial_type'])
            
            previous_rule = None
            current_rule = None
            persev_errors = 0
            non_persev_errors = 0
            total_errors = 0
            rule_change_count = 0
            
            for idx, row in exp_events.iterrows():
                trial_rule = row.get('trial_type')
                is_rule_change = bool(exp_rule_changes.loc[idx])
                response_outcome = row.get('response_outcome')
                response_pressed = row.get('response_pressed')
                
                # Handle rule change first: the rule changes at the block
                # boundary whether or not the participant responded, so a
                # rule-change trial with no response still starts a new rule
                # period. Skipping it first left the block, and the
                # next one, scored against a stale "previous rule".
                if is_rule_change:
                    previous_rule = current_rule  # Old rule becomes the "perseverated" rule
                    current_rule = trial_rule
                    rule_change_count += 1
                elif current_rule is None:
                    current_rule = trial_rule  # First trial initializes current rule

                # A trial with no response has no choice to score against a
                # rule, so it is not counted as an error here (it is still
                # incorrect in 'correct' and the accuracy metrics).
                # (trial_type is a string after _clean_events, so only the
                # response can be missing)
                if pd.isna(response_pressed):
                    continue
                    
                # Check if this trial is an error. NA first: response_outcome
                # is Int64, and comparing pd.NA with 0 cannot be cast to bool.
                if pd.isna(response_outcome) or response_outcome == 0:
                    total_errors += 1
                    
                    # Can only be perseverative if we know the previous rule
                    if previous_rule is not None:
                        # Check if response would have been correct under the previous rule
                        if would_response_be_correct(response_pressed, row, previous_rule):
                            persev_errors += 1
                        else:
                            non_persev_errors += 1
                    else:
                        # Before first rule change - cannot be perseverative
                        non_persev_errors += 1
            
            # Store metrics
            metrics['perseverative_errors'] = persev_errors
            metrics['non_perseverative_errors'] = non_persev_errors
            metrics['total_errors'] = total_errors
            
            # Calculate rates (based on total trials, consistent with standard definition)
            total_trials = len(exp_events)
            if total_trials > 0:
                metrics['perseverative_error_rate'] = persev_errors / total_trials
                metrics['non_perseverative_error_rate'] = non_persev_errors / total_trials
                # Also include error-conditional rates for detailed analysis
                if total_errors > 0:
                    metrics['perseverative_ratio_of_errors'] = persev_errors / total_errors
            else:
                metrics['perseverative_error_rate'] = 0.0
                metrics['non_perseverative_error_rate'] = 0.0
        
        # SET-SHIFTING EFFICIENCY METRICS (grouped by rule period, as the
        # maintenance metrics are). A session with too few experimental trials
        # or rule changes has these as missing (NaN), never absent, so every
        # reader finds the keys.
        metrics['categories_completed'] = np.nan
        metrics['rule_periods'] = np.nan
        metrics['prop_rule_periods_mastered'] = np.nan
        metrics['trials_to_first_category'] = np.nan
        if len(exp_trials) > 15 and exp_trials['rule_change'].sum() > 2:
            exp_sorted = exp_trials.sort_values(['trial_number']).copy()
            exp_sorted['period_id'] = exp_sorted['rule_change'].cumsum()
            
            trials_to_learn = []
            
            # Iterate through each rule period to find learning point
            for pid, group in exp_sorted.groupby('period_id'):
                # The learning criterion is 3 consecutive correct; the change
                # trial does not count toward it, and a trial with no response
                # is incorrect and breaks the streak. Same function as the GLM
                # labelling (scripts/wcst_rules.py). Trials to learn counts
                # from the change trial through the criterion trial.
                end = learning_end_position(group['correct'].tolist())
                if end is not None:
                    trials_to_learn.append(end + 1)
            
            if trials_to_learn:
                metrics['mean_trials_to_learn_rule'] = np.mean(trials_to_learn)
                metrics['set_shifting_efficiency'] = 1.0 / np.mean(trials_to_learn)
            
            # Categories completed: rule periods in which 3 consecutive correct
            # responses were reached. This is this study's learning criterion,
            # not the Heaton category (10 consecutive correct), so the count is
            # not comparable with Heaton-scored categories completed.
            metrics['categories_completed'] = len(trials_to_learn)
            # The count tracks how many rule periods a participant reached,
            # which depends on response speed (the number of experimental
            # trials varies by participant), so H5 uses the proportion of rule periods mastered
            # instead: categories completed / rule periods.
            n_rule_periods = int(exp_sorted['period_id'].nunique())
            metrics['rule_periods'] = n_rule_periods
            metrics['prop_rule_periods_mastered'] = len(trials_to_learn) / n_rule_periods
            metrics['trials_to_first_category'] = trials_to_learn[0] if trials_to_learn else np.nan
        
        # WORKING MEMORY MAINTENANCE METRICS
        # Definition: Maintenance starts AFTER the learning criterion (3 consecutive correct) is met.
        if len(exp_trials) > 20:
            exp_sorted = exp_trials.sort_values(['trial_number']).copy()
            exp_sorted['period_id'] = exp_sorted['rule_change'].cumsum()
            
            maintenance_trials = []
            
            for pid, group in exp_sorted.groupby('period_id'):
                # Only analyze periods with enough trials to potentially learn and maintain
                if len(group) < 4:
                    continue

                # Criterion trial: same boundary as the GLM's post_change /
                # regular split (change trial not counted).
                end = learning_end_position(group['correct'].tolist())
                criterion_met_idx = group.index[end] if end is not None else None

                if criterion_met_idx is not None:
                    # Get all trials after the criterion was met
                    criterion_trial_num = group.loc[criterion_met_idx, 'trial_number']
                    maint_data = group[group['trial_number'] > criterion_trial_num].copy()
                    
                    if len(maint_data) > 0:
                        # Normalize position relative to start of maintenance
                        maint_data['maint_position'] = range(len(maint_data))
                        maintenance_trials.append(maint_data)
            
            if maintenance_trials:
                all_maint = pd.concat(maintenance_trials)
                
                # Early Maintenance: First 3 trials of maintenance phase
                early_maint = all_maint[all_maint['maint_position'] < 3]
                # Late Maintenance: Trial 4+ of maintenance phase
                late_maint = all_maint[all_maint['maint_position'] >= 3]
                
                if len(early_maint) > 0:
                    metrics['early_rule_accuracy'] = early_maint['correct'].mean()
                    metrics['early_rule_rt'] = early_maint['response_time'].mean()
                
                if len(late_maint) > 0:
                    metrics['late_rule_accuracy'] = late_maint['correct'].mean()
                    metrics['late_rule_rt'] = late_maint['response_time'].mean()
                    
                if len(early_maint) > 0 and len(late_maint) > 0:
                    metrics['wm_maintenance_decline'] = (metrics['early_rule_accuracy'] - 
                                                       metrics['late_rule_accuracy'])
                    metrics['wm_rt_increase'] = (metrics['late_rule_rt'] - 
                                               metrics['early_rule_rt'])
                
                metrics['wm_valid_periods'] = len(maintenance_trials)
                metrics['wm_total_periods'] = exp_sorted['period_id'].nunique()
        
        # RT VARIABILITY (attentional control)
        valid_exp_trials = exp_trials.dropna(subset=['response_time'])
        valid_exp_trials = valid_exp_trials[valid_exp_trials['response_time'] > 0.1]
        correct_exp_trials = valid_exp_trials[valid_exp_trials['correct'] == 1]
        
        if len(correct_exp_trials) > 10:
            metrics['exp_correct_rt'] = correct_exp_trials['response_time'].mean()
            metrics['exp_rt_sd'] = correct_exp_trials['response_time'].std()
            metrics['exp_rt_cv'] = (correct_exp_trials['response_time'].std() / 
                                   correct_exp_trials['response_time'].mean())
        
        return metrics

    def get_visualization_data(self) -> Dict:
        """Prepare data for the timeline visualization."""
        if self.trial_data is None:
            self.process_trials()
            
        # Safely convert trials to list of dictionaries
        trials_list = self._convert_trials_to_list()
            
        viz_data = {
            'trials': trials_list,
            'controlBlocks': [
                {
                    'start': float(block['start']),
                    'end': float(block['end'])
                }
                for block in self.get_control_blocks()
            ],
            'metrics': {
                key: float(value) if isinstance(value, np.number) else value
                for key, value in self.get_performance_metrics().items()
            }
        }
        
        return viz_data

    def _convert_trials_to_list(self) -> List[Dict]:
        """Safely convert trial data to list of dictionaries."""
        if self.trial_data is None or len(self.trial_data) == 0:
            return []
        
        try:
            result = self.trial_data.apply(
                lambda row: {
                    'time': float(row['time']),
                    'blockType': str(row['block_type']),
                    'correct': int(row['correct']),
                    'ambiguityLevel': int(float(row['ambiguity_level'])) if pd.notna(row['ambiguity_level']) and str(row['ambiguity_level']).strip() != '' else 0,
                    'ruleChange': bool(int(row['rule_change'])) if pd.notna(row['rule_change']) and str(row['rule_change']).strip() != '' else False,
                    'responseTime': float(row['response_time']) if pd.notna(row['response_time']) and str(row['response_time']).strip() != '' else 0.0,
                    'trialNumber': int(float(row['trial_number'])) if pd.notna(row['trial_number']) and str(row['trial_number']).strip() != '' else 0
                },
                axis=1
            )
            # Handle both Series and DataFrame cases
            if hasattr(result, 'tolist'):
                return result.tolist()
            else:
                return list(result)
        except Exception as e:
            print(f"Error converting trials to list: {e}")
            return []
    
    def plot_timeline(self, save_path: str = None) -> None:
        """Create and save timeline visualization."""
        if self.trial_data is None:
            self.process_trials()
            
        plt.figure(figsize=(15, 6))
        
        # Plot control block backgrounds
        for block in self.get_control_blocks():
            plt.axvspan(block['start'], block['end'], 
                       color='yellow', alpha=0.2, label='Control Block')
        
        # Plot responses colored by ambiguity
        exp_data = self.trial_data[self.trial_data['block_type'] == 'experimental']
        ctrl_data = self.trial_data[self.trial_data['block_type'] == 'control']
        
        # Plot experimental trials
        plt.scatter(exp_data[exp_data['ambiguity_level'] == 1]['time'],
                   exp_data[exp_data['ambiguity_level'] == 1]['correct'],
                   c='blue', label='Low Ambiguity', alpha=0.7)
        plt.scatter(exp_data[exp_data['ambiguity_level'] == 2]['time'],
                   exp_data[exp_data['ambiguity_level'] == 2]['correct'],
                   c='orange', label='High Ambiguity', alpha=0.7)
        
        # Plot control trials
        plt.scatter(ctrl_data['time'], ctrl_data['correct'],
                   c='gray', label='Control', alpha=0.5)
        
        # Plot rule changes
        rule_changes = exp_data[exp_data['rule_change'] == 1]['time']
        for rc in rule_changes:
            plt.axvline(x=rc, color='red', linestyle='--', alpha=0.5)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Response Correct')
        plt.title('WCST Timeline')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


def main(task_name: str) -> None:
    """Main function to process WCST behavioral data."""
    # Load environment variables
    load_dotenv()
    base_dir = os.getenv('BASE_DIR')
    output_base = os.getenv('OUTPUT_DIR')
    
    if not all([base_dir, output_base]):
        raise ValueError("BASE_DIR and OUTPUT_DIR must be set in .env file")
    
    try:
        refuse_protected_tree(output_base)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    # Create output directory structure
    output_dir = os.path.join(output_base, '02_behav', task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read file info
    file_info = pd.read_csv(os.path.join(base_dir, 'data', f'{task_name}_task_info.csv'))
    
    # Process each subject
    all_results = []
    for _, row in file_info.iterrows():
        subj_id = row['subj_id']
        task_id = row['task_id']
        
        # Process events
        event_file = os.path.join(base_dir, 'data', task_name,
                                 f'NeuroLume_{subj_id}_{task_id}_task_events.tsv')
        
        # Every session in task_info must produce results; a skipped one
        # would vanish from every later stage and from group_results.json.
        if not os.path.exists(event_file):
            raise FileNotFoundError(f"Event file not found for subject {subj_id}, task {task_id}: {event_file}")
            
        print(f"\nProcessing subject {subj_id}, task {task_id}")
        try:
            events_df = pd.read_csv(event_file, sep='\t')
            # Strip whitespace from column names (file has trailing spaces)
            events_df.columns = events_df.columns.str.strip()
            # Strip whitespace from event values as well
            events_df['event'] = events_df['event'].str.strip()
            analyzer = WCSTEventAnalyzer(events_df)
            results = analyzer.get_visualization_data()
            
            # Save figures
            analyzer.plot_timeline(os.path.join(output_dir, f'{subj_id}_{task_id}_timeline.png'))
            # Add subject info to results
            results['subject'] = {'id': subj_id, 'task': task_id}
            all_results.append(results)
            
            # Save individual results
            with open(os.path.join(output_dir, f'{subj_id}_{task_id}_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error processing subject {subj_id}, task {task_id}: {str(e)}")
            raise
    
    # Save group-level results only if we have results
    if all_results:
        group_results = {
            'subjects': [r['subject'] for r in all_results],
'group_metrics': {
                metric: np.mean([r['metrics'][metric] for r in all_results 
                               if metric in r['metrics'] and not pd.isna(r['metrics'][metric])])
                for metric in set().union(*[r['metrics'].keys() for r in all_results])
                if any(metric in r['metrics'] and not pd.isna(r['metrics'][metric]) for r in all_results)
            }
        }
        
        with open(os.path.join(output_dir, 'group_results.json'), 'w') as f:
            json.dump(group_results, f, indent=2)
    else:
        print("Warning: No results were successfully processed")

if __name__ == "__main__":
    task_name = 'wcst'
    main(task_name)
