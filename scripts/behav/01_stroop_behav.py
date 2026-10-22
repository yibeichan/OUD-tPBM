import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_variant import refuse_protected_tree  # noqa: E402

class StroopEventAnalyzer:
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
            'response_outcome': 'Int64'
        }

        for col, dtype in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if dtype == 'Int64':
                    df[col] = df[col].round().astype('Int64')

        return df
        
    def process_trials(self) -> pd.DataFrame:
        """Process trial events into a clean format for visualization.

        Note: In source data, trial_type values are 'congruent' or 'mixed',
        where 'mixed' refers to incongruent trials (color-word mismatch).
        """
        # Filter for start_trial events only
        trial_events = self.events_df[self.events_df['event'] == 'start_trial'].copy()

        # Create trial data with required columns
        trial_data = pd.DataFrame({
            'time': trial_events['timestamp'],
            'block_type': trial_events['block_type'],  # 'cong_only' or 'mixed'
            'trial_type': trial_events['trial_type'],  # 'congruent' or 'mixed' (incongruent)
            'trial_number': trial_events['trial'],
            'correct': (trial_events['response_outcome'] == 1).fillna(False).astype(int),
            'response_time': trial_events['response_RT']
        })

        # Ensure numeric types for downstream comparisons
        trial_data['response_time'] = pd.to_numeric(trial_data['response_time'], errors='coerce')
        trial_data['trial_number'] = pd.to_numeric(trial_data['trial_number'], errors='coerce')
        trial_data['trial_number'] = trial_data['trial_number'].round().astype('Int64')
        
        # Add block transition markers
        trial_data['block_transition'] = (
            trial_data['block_type'] != trial_data['block_type'].shift(1)
        ).fillna(False).astype(int)
        
        self.trial_data = trial_data
        return trial_data
    
    def get_block_ranges(self) -> List[Dict[str, float]]:
        """Get time ranges for different block types."""
        if self.trial_data is None:
            self.process_trials()
            
        block_ranges = []
        current_block = None
        
        for _, row in self.trial_data.iterrows():
            if current_block is None or row['block_transition'] == 1:
                if current_block is not None:
                    current_block['end'] = row['time']
                    block_ranges.append(current_block)
                current_block = {
                    'start': row['time'],
                    'type': row['block_type']
                }
        
        # Handle last block
        if current_block is not None:
            current_block['end'] = self.trial_data['time'].iloc[-1]
            block_ranges.append(current_block)
            
        return block_ranges
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for Stroop task.

        Note: In the source data, trial_type == 'mixed' refers to INCONGRUENT trials
        (color and word mismatch), not to be confused with block_type == 'mixed'.

        Filtering strategy:
        - Accuracy metrics: Use ALL trials (no RT filtering)
        - RT metrics: Use filtered correct trials only (100ms < RT < 3s, correct responses)
        """
        if self.trial_data is None:
            self.process_trials()

        # STEP 1: Define trial subsets for ACCURACY metrics (use ALL trials, no filtering)
        cong_only_trials = self.trial_data[self.trial_data['block_type'] == 'cong_only']
        mixed_block_trials = self.trial_data[self.trial_data['block_type'] == 'mixed']
        cong_in_cong_only = cong_only_trials[cong_only_trials['trial_type'] == 'congruent']
        cong_in_mixed = mixed_block_trials[mixed_block_trials['trial_type'] == 'congruent']
        incong_in_mixed = mixed_block_trials[mixed_block_trials['trial_type'] == 'mixed']  # 'mixed' = incongruent

        # STEP 2: Filter for valid RT (100ms - 3s) and correct responses for RT metrics
        # This removes very fast (<100ms) and very slow (>3s) outlier responses
        valid_trials = self.trial_data.dropna(subset=['response_time']).copy()
        valid_trials = valid_trials[(valid_trials['response_time'] > 0.1) &
                                   (valid_trials['response_time'] < 3.0)]
        correct_trials = valid_trials[valid_trials['correct'] == 1]

        # STEP 3: Create filtered subsets for RT calculations
        correct_cong_only = correct_trials[correct_trials['block_type'] == 'cong_only']
        correct_mixed_block = correct_trials[correct_trials['block_type'] == 'mixed']
        correct_cong_in_cong_only = correct_cong_only[correct_cong_only['trial_type'] == 'congruent']
        correct_cong_in_mixed = correct_mixed_block[correct_mixed_block['trial_type'] == 'congruent']
        correct_incong_in_mixed = correct_mixed_block[correct_mixed_block['trial_type'] == 'mixed']  # 'mixed' = incongruent

        # STEP 4: Calculate metrics
        metrics = {
            # Overall performance
            'overall_accuracy': self.trial_data['correct'].mean(),  # ALL trials
            'overall_rt': correct_trials['response_time'].mean(),    # FILTERED correct trials

            # Block-level metrics
            'cong_only_accuracy': cong_only_trials['correct'].mean(),      # ALL trials
            'cong_only_rt': correct_cong_only['response_time'].mean(),     # FILTERED correct trials
            'mixed_block_accuracy': mixed_block_trials['correct'].mean(),  # ALL trials
            'mixed_block_rt': correct_mixed_block['response_time'].mean(), # FILTERED correct trials

            # Detailed trial-type metrics
            'cong_in_cong_only_accuracy': cong_in_cong_only['correct'].mean(),           # ALL trials
            'cong_in_cong_only_rt': correct_cong_in_cong_only['response_time'].mean(),   # FILTERED correct trials
            'cong_in_mixed_accuracy': cong_in_mixed['correct'].mean(),                   # ALL trials
            'cong_in_mixed_rt': correct_cong_in_mixed['response_time'].mean(),           # FILTERED correct trials
            'incong_in_mixed_accuracy': incong_in_mixed['correct'].mean(),               # ALL trials
            'incong_in_mixed_rt': correct_incong_in_mixed['response_time'].mean(),       # FILTERED correct trials

            # Interference effects (using FILTERED RT data for RT metrics)
            'mixing_cost_rt': (correct_cong_in_mixed['response_time'].mean() -
                              correct_cong_in_cong_only['response_time'].mean()),
            'mixing_cost_accuracy': (cong_in_cong_only['correct'].mean() -
                                    cong_in_mixed['correct'].mean()),
            'stroop_effect_rt': (correct_incong_in_mixed['response_time'].mean() -
                                correct_cong_in_mixed['response_time'].mean()),
            'stroop_effect_accuracy': (cong_in_mixed['correct'].mean() -
                                      incong_in_mixed['correct'].mean())
        }

        # ERROR RATES (standard reporting convention)
        # Error rate = 1 - accuracy
        metrics['overall_error_rate'] = 1 - metrics['overall_accuracy']
        metrics['cong_only_error_rate'] = 1 - metrics['cong_only_accuracy']
        metrics['mixed_block_error_rate'] = 1 - metrics['mixed_block_accuracy']
        metrics['cong_in_cong_only_error_rate'] = 1 - metrics['cong_in_cong_only_accuracy']
        metrics['cong_in_mixed_error_rate'] = 1 - metrics['cong_in_mixed_accuracy']
        metrics['incong_in_mixed_error_rate'] = 1 - metrics['incong_in_mixed_accuracy']

        # NORMALIZED INTERFERENCE INDICES
        # Accounts for individual differences in baseline RT (better for between-subject comparisons)
        baseline_cong_only_rt = metrics['cong_in_cong_only_rt']
        baseline_cong_mixed_rt = metrics['cong_in_mixed_rt']

        if baseline_cong_only_rt > 0:
            metrics['mixing_cost_normalized'] = metrics['mixing_cost_rt'] / baseline_cong_only_rt
        else:
            metrics['mixing_cost_normalized'] = np.nan

        if baseline_cong_mixed_rt > 0:
            metrics['stroop_effect_normalized'] = metrics['stroop_effect_rt'] / baseline_cong_mixed_rt
        else:
            metrics['stroop_effect_normalized'] = np.nan

        # RT VARIABILITY METRICS (key executive function markers)
        if len(correct_trials) > 5:
            metrics['overall_rt_sd'] = correct_trials['response_time'].std()
            metrics['overall_rt_cv'] = metrics['overall_rt_sd'] / correct_trials['response_time'].mean()
            
            # Condition-specific RT variability
            congruent_mixed = correct_trials[(correct_trials['block_type'] == 'mixed') &
                                           (correct_trials['trial_type'] == 'congruent')]
            incongruent_mixed = correct_trials[(correct_trials['block_type'] == 'mixed') &
                                             (correct_trials['trial_type'] == 'mixed')]  # 'mixed' = incongruent
            
            if len(congruent_mixed) > 3:
                metrics['congruent_rt_cv'] = congruent_mixed['response_time'].std() / congruent_mixed['response_time'].mean()
            if len(incongruent_mixed) > 3:
                metrics['incongruent_rt_cv'] = incongruent_mixed['response_time'].std() / incongruent_mixed['response_time'].mean()
        
        # ERROR MONITORING METRICS
        # Error clustering (consecutive errors indicate attentional lapses)
        if len(valid_trials) > 10:
            df_sorted = valid_trials.sort_values(['trial_number']).copy()
            df_sorted['error'] = (df_sorted['correct'] == 0).astype(int)
            error_runs = []
            current_run = 0

            for error in df_sorted['error']:
                if error == 1:
                    current_run += 1
                else:
                    if current_run > 0:
                        error_runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                error_runs.append(current_run)

            metrics['max_consecutive_errors'] = max(error_runs) if error_runs else 0
        
        # CONFLICT ADAPTATION (Gratton effect)
        # conflict_adaptation_cost_rt, in seconds: among correct mixed-block
        # trials in trial order, the mean RT of incongruent trials whose
        # preceding trial (the previous correct mixed-block trial) was
        # congruent, minus the mean RT of incongruent trials whose preceding
        # trial was incongruent. Positive values mean slower responses after a
        # congruent trial, i.e. less benefit from conflict history. Computed
        # only with more than 15 correct mixed-block trials, more than 8 of
        # them incongruent, and more than 2 in each group; otherwise absent.
        mixed_correct = valid_trials[(valid_trials['block_type'] == 'mixed') &
                                   (valid_trials['correct'] == 1)].sort_values(['trial_number']).copy()

        if len(mixed_correct) > 15:
            mixed_correct['prev_congruent'] = mixed_correct['trial_type'].shift(1) == 'congruent'
            curr_incong = mixed_correct[mixed_correct['trial_type'] == 'mixed']  # 'mixed' = incongruent

            if len(curr_incong) > 8:
                incong_after_cong = curr_incong[curr_incong['prev_congruent'] == True]
                incong_after_incong = curr_incong[curr_incong['prev_congruent'] == False]

                if len(incong_after_cong) > 2 and len(incong_after_incong) > 2:
                    # Positive = cost (slower after congruent trial = worse adaptation)
                    metrics['conflict_adaptation_cost_rt'] = (incong_after_cong['response_time'].mean() -
                                                             incong_after_incong['response_time'].mean())

        return metrics
    
    def plot_timeline(self, save_path: str = None) -> None:
        """Create and save timeline visualization for Stroop task."""
        if self.trial_data is None:
            self.process_trials()
            
        plt.figure(figsize=(15, 6))
        
        # Plot block backgrounds
        for block in self.get_block_ranges():
            color = 'lightblue' if block['type'] == 'cong_only' else 'lightyellow'
            plt.axvspan(block['start'], block['end'], 
                    color=color, alpha=0.3, 
                    label=f"{block['type']} block")
        
        # Get different trial types
        cong_in_cong_only = self.trial_data[
            (self.trial_data['block_type'] == 'cong_only') &
            (self.trial_data['trial_type'] == 'congruent')
        ]
        cong_in_mixed = self.trial_data[
            (self.trial_data['block_type'] == 'mixed') &
            (self.trial_data['trial_type'] == 'congruent')
        ]
        incong_in_mixed = self.trial_data[
            (self.trial_data['block_type'] == 'mixed') &
            (self.trial_data['trial_type'] == 'mixed')  # 'mixed' = incongruent
        ]

        # Plot trials with different colors
        plt.scatter(cong_in_cong_only['time'], cong_in_cong_only['correct'],
                c='blue', label='Congruent (Pure Block)', alpha=0.7)
        plt.scatter(cong_in_mixed['time'], cong_in_mixed['correct'],
                c='green', label='Congruent (Mixed Block)', alpha=0.7)
        plt.scatter(incong_in_mixed['time'], incong_in_mixed['correct'],
                c='red', label='Incongruent (Mixed Block)', alpha=0.7)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Response Correct')
        plt.title('Stroop Task Timeline')
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def get_visualization_data(self) -> Dict:
        """Prepare data for visualization and analysis."""
        if self.trial_data is None:
            self.process_trials()
            
        # Safely convert trials to list of dictionaries
        trials_list = self._convert_trials_to_list()
            
        viz_data = {
            'trials': trials_list,
            'blocks': [
                {
                    'start': float(block['start']),
                    'end': float(block['end']),
                    'type': str(block['type'])
                }
                for block in self.get_block_ranges()
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
                    'trialType': str(row['trial_type']),
                    'correct': int(row['correct']),
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


def main(task_name: str) -> None:
    """Main function to process Stroop behavioral data."""
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
            analyzer = StroopEventAnalyzer(events_df)
            results = analyzer.get_visualization_data()
            
            # Save timeline plot
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
    
    # Save group-level results
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
    task_name = 'stroop'
    main(task_name)
