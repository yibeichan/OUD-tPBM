import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from nilearn.plotting import plot_design_matrix
import mne_nirs
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import get_short_channels
from typing import Dict, Tuple, List
import logging
import psutil
import argparse
import re
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_names import rename_raw_to_canonical  # noqa: E402
from trial_onsets import match_trial_onsets, snirf_stim_onsets  # noqa: E402
from channel_quality import drop_unusable_channels  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subject_ids import task_events_filename  # noqa: E402
from analysis_variant import refuse_protected_tree  # noqa: E402
from glm_summary import write_session_summary  # noqa: E402

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_event_table(events_file: str) -> pd.DataFrame:
    """Read a task event table, tolerating whitespace-padded fields.

    Some tables carry padded column names and padded values. Column names and
    text values are stripped, so ``df['event'] == 'start_trial'`` matches the
    trial rows instead of silently matching nothing.

    Stripping alone leaves a numeric column as text: a missing
    ``response_outcome`` arrives as blanks, the column reads as object, and
    ``row['response_outcome'] == 1`` compares a string to an int and is always
    False. So blank values become missing, and a column whose every present
    value parses as a number is converted back to numeric.

    The behavioural script, ``scripts/behav/01_stroop_behav.py``, also strips
    the padding but does not restore numeric dtypes.
    """
    df = pd.read_csv(events_file, sep="\t")
    df.columns = df.columns.str.strip()
    for column in df.columns:
        if df[column].dtype != object:
            continue
        stripped = df[column].str.strip()
        # A padded blank ("    ") strips to "", which is a *value*, not a gap:
        # it keeps read_csv's untyped object column object-typed.
        stripped = stripped.mask(stripped == "")
        present = stripped.dropna()
        if present.empty:
            df[column] = stripped
            continue
        # Coerce back to numbers only when the whole column is numbers. A
        # blanket errors="coerce" would turn a genuinely textual column into
        # all-NaN just as quietly as the object dtype breaks "== 1".
        as_number = pd.to_numeric(present, errors="coerce")
        df[column] = pd.to_numeric(stripped, errors="coerce") \
            if as_number.notna().all() else stripped
    return df


class StroopfNIRSAnalysis:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.raw = None
        self.events_df = None
        self.design_matrix = None
        self.glm_estimates = None
        self.basic_contrasts = None
        self.contrasts = None
        self.contrast_maps = None
        self.design_matrix_rank = None
        self.used_noise_model = None
        self.onset_offset = None
        self.dropped_channels = []
        
    def process_subject(self, subj_id: str, task_id: str, filename: str, stim_dur: float) -> None:
        """
        Process a single subject's data.
        
        Parameters
        ----------
        subj_id : str
            Subject ID
        task_id : str
            Task ID
        filename : str
            SNIRF filename
        stim_dur : float
            Stimulus duration for HRF modeling (default: 1.0 seconds).
            Based on actual trial durations from task_events.tsv where
            start_trial events have ~1s duration for stimulus presentation.
        """
        # Construct file paths
        snirf_file = os.path.join(os.getenv("BASE_DIR"), "data", self.task_name, filename)
        events_file = os.path.join(os.getenv("BASE_DIR"), "data", self.task_name,
                                 task_events_filename(subj_id, task_id))
        
        # Create output directories
        output_base = os.getenv("OUTPUT_DIR")
        figure_dir = os.path.join(output_base, "figures", "GLM", self.task_name)
        glm_dir = os.path.join(output_base, "03_glm", self.task_name)
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(glm_dir, exist_ok=True)
        
        # Monitor memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        try:
            # Load and process data
            self.load_data(snirf_file, events_file)
            events, condition_mapping = self.process_events()
            self.create_annotations(events, condition_mapping)
            self.create_design_matrix(stim_dur=stim_dur)
            self.run_glm_analysis()
            
            # Save design matrix plot
            design_matrix_path = os.path.join(figure_dir, 
                                            f"{subj_id}_{task_id}_designmatrix.png")
            self.plot_design_matrix(design_matrix_path)
            
            # Save events plot
            events_path = os.path.join(figure_dir, 
                                     f"{subj_id}_{task_id}_events.png")
            self.plot_events(events_path)
            
            # Add this line to create contrast plots
            contrast_path = os.path.join(figure_dir, f"{subj_id}_{task_id}_contrasts.png")
            self.plot_contrasts(contrast_path)
            
            # Save GLM results
            glm_output_path = os.path.join(glm_dir, 
                                         f"{subj_id}_{task_id}_glm.pkl")
            self.save_glm_results(glm_output_path)
            self.save_session_record(
                subj_id, task_id, os.path.join(glm_dir, f"{subj_id}_{task_id}_glm_session.json"))
            
            # Log memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            logger.info(f"Memory used for processing: {memory_used:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error processing subject {subj_id}, task {task_id}: {str(e)}")
            raise
            
    def load_data(self, snirf_file: str, events_file: str, optode_frame: str = "mri") -> None:
        """Load SNIRF and events data."""
        logger.info(f"Loading SNIRF data from {snirf_file}")
        try:
            self.raw = mne.io.read_raw_snirf(snirf_file, optode_frame=optode_frame)
            self.raw.load_data()

            # Name channels by the full probe's optode order so a channel means
            # the same optode in every session (see channel_names.py).
            n_renamed = rename_raw_to_canonical(self.raw, snirf_file)
            if n_renamed:
                logger.info(f"Renamed {n_renamed} channels to canonical names")

            # A constant or non-finite channel has no estimable beta and would
            # force the whole session's AR(1) fit to OLS. It is dropped
            # here, before the short-channel regressors and the GLM, so it is
            # absent from the outputs rather than written as a zero.
            self.dropped_channels = drop_unusable_channels(self.raw)
            if self.dropped_channels:
                logger.warning(
                    f"Dropped {len(self.dropped_channels)} zero-variance or non-finite "
                    f"channel(s) before the GLM: {self.dropped_channels}"
                )

            logger.info(f"Loading events data from {events_file}")
            self.events_df = read_event_table(events_file)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def process_events(self) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Process events and create event codes for different conditions.
        
        Note: In source data, trial_type values are 'congruent' or 'mixed',
        where 'mixed' refers to INCONGRUENT trials (color-word mismatch).
        
        Block structure:
        - cong_only: Contains only congruent trials
        - mixed: Contains both congruent trials and incongruent trials (labeled as 'mixed' in raw data)
        
        Returns
        -------
        events_new : np.ndarray
            Processed events array
        condition_mapping : dict
            Mapping between event codes and condition names
        """
        events_new = []
        sfreq = self.raw.info['sfreq']
        
        # Filter for start_trial events
        event_df_filtered = self.events_df[
            self.events_df['event'] == 'start_trial'
        ].sort_values('timestamp', kind='stable')
        
        logger.info(f"Processing {len(event_df_filtered)} trials...")

        # Onsets come from the SNIRF StartTrial stims, not the event table's
        # timestamp, which runs on a different clock. The match
        # raises if the counts differ, if the offset is not constant to one
        # sample, or if an onset falls outside the recording, so no trial is
        # ever dropped here.
        onsets, self.onset_offset = match_trial_onsets(
            event_df_filtered['timestamp'].to_numpy(dtype=float),
            snirf_stim_onsets(self.raw),
            sfreq,
            n_times=len(self.raw.times),
        )
        logger.info(
            f"Event-table minus SNIRF onset offset: {self.onset_offset:.3f} s "
            f"({self.onset_offset * sfreq:.2f} samples); onsets taken from the SNIRF stims"
        )
        
        # Create events with block context
        for position, (idx, row) in enumerate(event_df_filtered.iterrows()):
            if pd.isna(row['trial_type']):
                continue
            
            sample = int(onsets[position] * sfreq)
            
            # Assign event codes based on trial and block type
            if row['trial_type'] == 'congruent':
                event_code = 1 if row['block_type'] == 'cong_only' else 2
            else:  # trial_type == 'mixed' indicates INCONGRUENT trials
                event_code = 3
            
            events_new.append([sample, 0, event_code])
        
        events_new = np.array(events_new)
        
        condition_mapping = {
            1: 'congruent_pure',     # Congruent trials in pure blocks
            2: 'congruent_mixed',    # Congruent trials in mixed blocks
            3: 'incongruent'         # Incongruent trials (only occur in mixed blocks)
        }
        
        logger.info(f"Created {len(events_new)} events.")
        return events_new, condition_mapping
    
    def create_annotations(self, events: np.ndarray, condition_mapping: Dict[int, str]) -> None:
        """
        Create and set annotations for the raw data.
        
        Parameters
        ----------
        events : np.ndarray
            Processed events array
        condition_mapping : dict
            Mapping between event codes and condition names
        """
        annot_from_events = mne.annotations_from_events(
            events=events,
            event_desc=condition_mapping,
            sfreq=self.raw.info['sfreq'],
            orig_time=self.raw.info['meas_date']
        )
        self.raw.set_annotations(annot_from_events)
    
    def create_design_matrix(
        self,
        stim_dur: float,
        drift_model: str = "cosine",
        high_pass: float = 0.01,
        hrf_model: str = "glover",
    ) -> None:
        """
        Create design matrix for GLM analysis.
        
        Parameters
        ----------
        stim_dur : float
            Duration of the stimulus in seconds
        drift_model : str
            Type of drift model to use
        high_pass : float
            High pass filter cutoff
        hrf_model : str
            HRF model to use
        """
        logger.info("Creating design matrix...")
        try:
            self.design_matrix = make_first_level_design_matrix(
                self.raw,
                drift_model=drift_model,
                high_pass=high_pass,
                hrf_model=hrf_model,
                stim_dur=stim_dur,
            )
            
            # Add short channel regression. The ShortHbO and ShortHbR
            # regressors are in every model, so a session without them stops
            # here instead of being fit without them.
            logger.info("Adding short channel regressors...")
            # get_short_channels returns a Raw object with only short-separation channels
            # (channels with source-detector distance < max_dist, default 10mm)
            raw_short = get_short_channels(self.raw)  # Returns Raw object, not list
            n_short_channels = len(raw_short.ch_names)
            if n_short_channels == 0:
                raise ValueError("No short channels found in data; cannot add short-channel regressors.")
            logger.info(f"Found {n_short_channels} short channels.")
            for chromophore, column in (("hbo", "ShortHbO"), ("hbr", "ShortHbR")):
                short_data = raw_short.copy().pick(picks=chromophore).get_data()
                if short_data.size == 0:
                    raise ValueError(f"No short {chromophore.upper()} channels found to create the {column} regressor.")
                self.design_matrix[column] = np.mean(short_data, axis=0)
                logger.info(f"Added {column} regressor from {short_data.shape[0]} channels.")

            # Check the rank of the matrix that is actually fitted, after the
            # short-channel regressors are added.
            rank = np.linalg.matrix_rank(self.design_matrix)
            n_cols = self.design_matrix.shape[1]
            self.design_matrix_rank = rank  # Store for summary
            if rank < n_cols:
                raise ValueError(f"Design matrix is rank deficient: rank={rank}, n_cols={n_cols}")
            logger.info(f"Design matrix has full rank: {rank}/{n_cols}")

            # Create basic contrasts from design matrix
            n_columns = len(self.design_matrix.columns)
            self.basic_contrasts = {}
            
            # Create basic contrast vectors (one-hot encoded)
            # Ensure these conditions actually exist in the design matrix after make_first_level_design_matrix
            # Every reported contrast needs all three conditions, so a
            # session missing one stops here rather than being written with
            # fewer contrasts.
            condition_columns = ['congruent_pure', 'congruent_mixed', 'incongruent']
            missing = [c for c in condition_columns if c not in self.design_matrix.columns]
            if missing:
                raise ValueError(f"Condition column(s) {missing} not found in the design matrix.")
                 
            for col in condition_columns:
                contrast_vector = np.zeros(n_columns)
                col_idx = self.design_matrix.columns.get_loc(col)
                contrast_vector[col_idx] = 1
                self.basic_contrasts[col] = contrast_vector
                
            # Define contrasts of interest for Stroop
            # Note: 'congruent_mixed' = congruent trials in mixed blocks
            #       'congruent_pure' = congruent trials in pure blocks
            #       'incongruent' = incongruent trials (only in mixed blocks)
            self.contrasts = {}
            
            # Classic Stroop interference effect (within mixed blocks)
            if 'incongruent' in self.basic_contrasts and 'congruent_mixed' in self.basic_contrasts:
                self.contrasts['stroop_effect'] = self.basic_contrasts["incongruent"] - self.basic_contrasts["congruent_mixed"]
            else:
                logger.warning("Could not create 'stroop_effect' contrast due to missing basic contrasts.")
            
            # Mixing cost: cost of being in mixed block for congruent trials
            if 'congruent_mixed' in self.basic_contrasts and 'congruent_pure' in self.basic_contrasts:
                self.contrasts['mixing_cost'] = self.basic_contrasts["congruent_mixed"] - self.basic_contrasts["congruent_pure"]
            else:
                logger.warning("Could not create 'mixing_cost' contrast due to missing basic contrasts.")
                
            # Overall incongruent vs pure congruent (maximum contrast)
            if 'incongruent' in self.basic_contrasts and 'congruent_pure' in self.basic_contrasts:
                self.contrasts['incongruent_vs_pure'] = self.basic_contrasts["incongruent"] - self.basic_contrasts["congruent_pure"]
            else:
                logger.warning("Could not create 'incongruent_vs_pure' contrast due to missing basic contrasts.")

            if not self.contrasts:
                 logger.warning("No contrasts of interest could be defined.")
            
            # Log contrast information
            logger.info("Created contrasts with shapes:")
            for name, contrast in self.contrasts.items():
                logger.info(f"{name}: {contrast.shape}")
                
        except Exception as e:
            logger.error(f"Error creating design matrix: {str(e)}")
            logger.error("Full error:", exc_info=True)
            raise
    
    def run_glm_analysis(self) -> None:
        """Run GLM analysis and compute contrasts using mne-nirs tools."""
        logger.info("Running GLM analysis...")
        
        # Try different noise models in order of preference
        noise_models = ['ar1', 'ols']
        glm_success = False
        used_noise_model = None
        
        for noise_model in noise_models:
            try:
                logger.info(f"Attempting GLM with noise_model='{noise_model}'")
                
                # Run GLM (returns a single RegressionResults object for all channels)
                self.glm_estimates = run_glm(
                    self.raw, 
                    self.design_matrix, 
                    noise_model=noise_model
                )
                
                # If we get here, GLM succeeded
                glm_success = True
                used_noise_model = noise_model
                self.used_noise_model = noise_model  # Store for summary
                
                if noise_model == 'ols':
                    logger.warning("WARNING: Using OLS noise model due to AR(1) failure - interpret p-values conservatively!")
                else:
                    logger.info(f"Successfully completed GLM with {noise_model} noise model")
                
                break  # Exit the loop on success
                
            except Exception as e:
                logger.error(f"GLM with {noise_model} failed: {str(e)}")
                if noise_model == noise_models[-1]:  # Last attempt
                    logger.error("All GLM attempts failed")
                    raise
        
        if not glm_success:
            self.glm_estimates = None
            self.contrast_maps = None
            return
        
        # Extract contrast results
        logger.info("Computing and extracting contrast results...")
        self.contrast_maps = {} # Store results per contrast
        n_regressors = self.design_matrix.shape[1]

        # Loop through each defined contrast
        for contrast_name, contrast_vector in self.contrasts.items():
                 logger.debug(f"Computing contrast: {contrast_name}")
                 # Ensure contrast vector has the correct length
                 if len(contrast_vector) != n_regressors:
                     raise ValueError(f"Contrast '{contrast_name}' has incorrect length. Expected {n_regressors}, got {len(contrast_vector)}.")
                     
                 try:
                     # Compute the contrast across all channels at once
                     contrast_obj = self.glm_estimates.compute_contrast(contrast_vector, contrast_type='t') # MNE-NIRS ContrastResults
                     logger.info(f"Successfully created MNE-NIRS contrast object for {contrast_name}: {type(contrast_obj)}")
                     
                     # Access the underlying Nilearn contrast object
                     nilearn_contrast = contrast_obj.data 
                     logger.info(f"Accessed underlying Nilearn contrast object: {type(nilearn_contrast)}")

                     # Initialize storage for this contrast map
                     contrast_results_for_map = {}
                     
                     # Attempt to extract stats directly from the Nilearn object
                     # Assuming these return arrays ordered by channel
                     effects = nilearn_contrast.effect_size() # Confirmed by Nilearn docs
                     stats = nilearn_contrast.stat()          # Confirmed by Nilearn docs
                     p_values = nilearn_contrast.p_value()      # Confirmed by Nilearn docs
                     variances = nilearn_contrast.effect_variance() # Confirmed by Nilearn docs

                     for i, ch_name in enumerate(self.raw.ch_names):
                         # Calculate SE from variance, handle potential negative variance for safety
                         se = np.sqrt(variances[i]) if variances[i] >= 0 else None
                         contrast_results_for_map[ch_name] = {
                             'effect': effects[i],
                             'stat': stats[i],       
                             'p_value': p_values[i],  
                             'se': se, 
                             'df': nilearn_contrast.dof, # Confirmed by Nilearn docs
                             'ch_name': ch_name,      
                         }
                         
                     # Store the results for this contrast
                     self.contrast_maps[contrast_name] = contrast_results_for_map
                     logger.debug(f"Successfully computed and stored contrast: {contrast_name}")

                 except Exception as e:
                     # A session written with fewer contrasts than the model
                     # defines would still be reported as a success.
                     logger.error(f"Could not compute contrast '{contrast_name}'. Error: {e}")
                     raise

        logger.info("Successfully computed and extracted all contrast results.")
    
    def plot_design_matrix(self, design_matrix_path: str = None) -> None:
        """
        Plot the design matrix.
        
        Parameters
        ----------
        design_matrix_path : str, optional
            Path to save the design matrix plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_design_matrix(self.design_matrix, ax=ax)
        
        # Add labels for better clarity
        ax.set_xlabel('Time (scans)')
        ax.set_ylabel('Regressors')
        
        if design_matrix_path:
            plt.savefig(design_matrix_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved design matrix plot to {design_matrix_path}")
        plt.close()

    def plot_contrasts(self, contrast_path: str = None) -> None:
        """Plot contrast maps for each defined contrast with enhanced visualization."""
        try:
            # Check if we have any contrast maps
            if not hasattr(self, 'contrast_maps') or not self.contrast_maps:
                logger.warning("No contrast maps available to plot")
                return

            n_contrasts = len(self.contrast_maps)
            if n_contrasts == 0:
                logger.warning("No contrasts to plot")
                return

            # Create figure with 1 column and n_contrasts * 2 rows (stack HbO/HbR)
            # Increase width, adjust height per pair
            fig, axes = plt.subplots(n_contrasts * 2, 1, figsize=(18, 6 * n_contrasts), squeeze=False) 
            axes = axes.flatten() # Flatten the axes array for easier indexing

            for row, (contrast_name, contrast_values) in enumerate(self.contrast_maps.items()):
                try:
                    # Convert the dictionary of results for this contrast into a DataFrame
                    contrast_df = pd.DataFrame.from_dict(contrast_values, orient='index')
                    if 'ch_name' not in contrast_df.columns:
                         logger.error(f"DataFrame for contrast '{contrast_name}' is missing 'ch_name' column. Columns: {contrast_df.columns.tolist()}. Skipping plot.")
                         continue

                    # Filter the DataFrame for HbO and HbR
                    hbo_data = contrast_df[contrast_df['ch_name'].str.contains('hbo', case=False, na=False)]
                    hbr_data = contrast_df[contrast_df['ch_name'].str.contains('hbr', case=False, na=False)]
                    
                    # Calculate axes indices for this contrast's HbO and HbR
                    hbo_ax_idx = row * 2
                    hbr_ax_idx = row * 2 + 1
                    
                    # Pass the filtered DataFrames to the plotting helper
                    if not hbo_data.empty:
                        self._plot_contrast_subset(axes[hbo_ax_idx], contrast_name, hbo_data, 'hbo', 'r')
                    else:
                        logger.warning(f"No HbO data found for contrast {contrast_name}")
                        axes[hbo_ax_idx].set_title(f"{contrast_name} - HbO (No Data)") # Add placeholder title
                    
                    if not hbr_data.empty:
                        self._plot_contrast_subset(axes[hbr_ax_idx], contrast_name, hbr_data, 'hbr', 'b')
                    else:
                        logger.warning(f"No HbR data found for contrast {contrast_name}")
                        axes[hbr_ax_idx].set_title(f"{contrast_name} - HbR (No Data)") # Add placeholder title
                        
                except Exception as e:
                    logger.error(f"Error preparing/plotting contrast {contrast_name}: {str(e)}")
                    logger.error("Full error:", exc_info=True) # Log traceback for plotting errors
                    # Ensure axes are cleared or marked if error occurs mid-plot
                    hbo_ax_idx = row * 2
                    hbr_ax_idx = row * 2 + 1
                    if hbo_ax_idx < len(axes): axes[hbo_ax_idx].set_title(f"{contrast_name} - HbO (Error)")
                    if hbr_ax_idx < len(axes): axes[hbr_ax_idx].set_title(f"{contrast_name} - HbR (Error)")
                    continue

            plt.tight_layout(h_pad=2.0) # Add some vertical padding
            if contrast_path:
                plt.savefig(contrast_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved enhanced contrast plots to {contrast_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Error in plot_contrasts: {str(e)}")
            plt.close()  # Ensure figure is closed even if there's an error

    def _plot_contrast_subset(self, ax, contrast_name, data, hb_type, color):
        """Plot one contrast's channel effects for one chromophore; ``data`` is that chromophore's rows."""
        try:
            if data.empty:
                # This check is now redundant as it's done in plot_contrasts, but keep for safety
                logger.warning(f"No data passed to _plot_contrast_subset for {contrast_name} - {hb_type}")
                ax.set_title(f"{contrast_name} - {hb_type.upper()} (No Data)")
                return
            
            # Use .loc to create the new column and avoid SettingWithCopyWarning
            data_copy = data.copy() # Work on a copy explicitly
            data_copy.loc[:, 'channel'] = data_copy['ch_name']
            # Sort numerically if possible, otherwise alphabetically
            try:
                 # Ensure extracted parts are treated as integers for correct sorting
                 sort_keys = data_copy['ch_name'].str.extract(r'S(\d+)_D(\d+)').astype(int)
                 # Create a tuple for multi-level sort, handling potential NaN if pattern fails
                 data_copy.loc[:, 'sort_key'] = list(zip(sort_keys[0], sort_keys[1])) 
                 data_copy = data_copy.sort_values('sort_key')
            except Exception as sort_e:
                 logger.warning(f"Could not sort numerically for {contrast_name} {hb_type}, falling back to string sort. Error: {sort_e}")
                 data_copy = data_copy.sort_values('channel') # Fallback sort

            # Create bar plot using the sorted copy
            bars = ax.bar(range(len(data_copy)), data_copy['effect'], color=color, alpha=0.6)
            
            # Add error bars (if 'se' is available and meaningful)
            if 'se' in data_copy.columns and pd.notna(data_copy['se']).all():
                 ax.errorbar(range(len(data_copy)), data_copy['effect'], yerr=data_copy['se'],
                            fmt='none', color='black', capsize=3)
            
            # Add significance markers based on p_value
            for i, row in enumerate(data_copy.itertuples()):
                 # Check if p_value exists and is not None
                 if hasattr(row, 'p_value') and pd.notna(row.p_value):
                      if row.p_value < 0.05:
                          marker = '*' * sum([row.p_value < cutoff for cutoff in [0.05, 0.01, 0.001]])
                          # Adjust text position based on effect sign and SE (if available)
                          text_y = row.effect
                          if hasattr(row, 'se') and pd.notna(row.se):
                               text_y += row.se if row.effect >= 0 else -row.se
                          # Add small offset relative to axis limits if SE is zero or missing
                          y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                          offset = 0.02 * y_range * (1 if row.effect >= 0 else -1)
                          text_y += offset
                          
                          ax.text(i, text_y, marker, ha='center', va='bottom' if row.effect >=0 else 'top')

            # Customize plot
            ax.set_xlabel('Channel')
            ax.set_ylabel('Effect Size') # Or 'Contrast Estimate'
            ax.set_title(f"{contrast_name} - {hb_type.upper()}")
            ax.set_xticks(range(len(data_copy)))
            ax.set_xticklabels(data_copy['channel'], rotation=45, ha='right', fontsize=8) # Reduce fontsize
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        except Exception as e:
            logger.error(f"Error in _plot_contrast_subset for {contrast_name} - {hb_type}: {str(e)}")
            logger.error("Full error:", exc_info=True)
            raise
        
    def plot_events(self, event_path: str = None) -> None:
        """
        Plot the events.
        
        Parameters
        ----------
        event_path : str, optional
            Path to save the events plot
        """
        try:
            events, event_dict = mne.events_from_annotations(self.raw)
            
            # Ensure color_dict keys match event_dict keys (likely strings)
            color_dict = {}
            for event_name, event_id in event_dict.items(): # Iterate through event_dict items
                if 'congruent_pure' in event_name:
                    color_dict[event_id] = 'blue'
                elif 'congruent_mixed' in event_name:
                    color_dict[event_id] = 'green'
                elif 'incongruent' in event_name:
                    color_dict[event_id] = 'red'
                else:
                    color_dict[event_id] = 'gray'
            
            fig = mne.viz.plot_events(
                events,
                event_id=event_dict,
                sfreq=self.raw.info["sfreq"],
                color=color_dict
            )
            
            if event_path:
                fig.savefig(event_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved events plot to {event_path}")
            plt.close()
        
        except Exception as e:
            logger.error(f"Error in plot_events: {str(e)}")
            logger.error("Full error details:", exc_info=True) # Add traceback
            plt.close()
    
    def session_record(self, subj_id: str, task_id: str) -> dict:
        """What this session's fit used: onset offset, dropped channels, noise model, rank."""
        return {
            'subject': str(subj_id),
            'session_id': str(task_id),
            'task': self.task_name,
            'onset_source': 'snirf_StartTrial',
            'onset_offset_s': None if self.onset_offset is None else round(float(self.onset_offset), 4),
            'dropped_channels': list(self.dropped_channels),
            'n_channels_fit': len(self.raw.ch_names),
            'noise_model': self.used_noise_model,
            'design_matrix_rank': None if self.design_matrix_rank is None else int(self.design_matrix_rank),
            'design_matrix_columns': [str(c) for c in self.design_matrix.columns],
        }

    def save_session_record(self, subj_id: str, task_id: str, output_path: str) -> None:
        """Write :meth:`session_record` as JSON next to the session's contrasts."""
        with open(output_path, 'w') as f:
            json.dump(self.session_record(subj_id, task_id), f, indent=2)
        logger.info(f"Saved session record to {output_path}")

    def save_glm_results(self, output_path: str) -> None:
        """Save GLM results and contrasts to files."""
        logger.info(f"Saving GLM results to {output_path}")
        
        try:
            # Combine all contrasts into a single DataFrame for easier analysis later
            all_contrasts_list = []
            for contrast_name, ch_data_dict in self.contrast_maps.items():
                 contrast_df = pd.DataFrame.from_dict(ch_data_dict, orient='index')
                 contrast_df['contrast'] = contrast_name
                 # Extract chromophore type
                 contrast_df['chromophore'] = contrast_df['ch_name'].str.extract(r'(hb[or])$', expand=False) 
                 # Extract source and detector
                 sd_match = contrast_df['ch_name'].str.extract(r'S(\d+)_D(\d+)', expand=True)
                 if not sd_match.empty: # Check if extraction worked
                      contrast_df['Source'] = sd_match[0]
                      contrast_df['Detector'] = sd_match[1]
                 all_contrasts_list.append(contrast_df.reset_index(drop=True)) # Reset index before concat

            if all_contrasts_list: # Check if list is not empty
                 all_contrasts_df = pd.concat(all_contrasts_list, ignore_index=True)
             
                 # Save combined contrasts DataFrame
                 contrast_pkl_path = output_path.replace('.pkl', '_contrasts.pkl')
                 all_contrasts_df.to_pickle(contrast_pkl_path)
                 logger.info(f"Saved combined contrast results (DataFrame) to {contrast_pkl_path}")

                 # Save CSV version for quick inspection
                 contrast_csv_path = output_path.replace('.pkl', '_contrasts.csv')
                 all_contrasts_df.to_csv(contrast_csv_path, index=False)
                 logger.info(f"Saved combined contrast results (CSV) to {contrast_csv_path}")
            else:
                 raise ValueError("No contrast dataframes generated to save.")

            logger.info(f"GLM result saving process complete.")
            
        except Exception as e:
            logger.error(f"Error in save_glm_results: {str(e)}")
            logger.error("Full error:", exc_info=True)
            raise

def setup_file_logging(subj_id, task_id, task_name):
    """Set up file logging for this subject."""
    output_dir = os.getenv("OUTPUT_DIR")
    log_dir = os.path.join(output_dir, "logs", "GLM", task_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a file handler for this specific subject
    log_file = os.path.join(log_dir, f"{subj_id}_{task_id}_glm_analysis.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return log_file

def main():
    parser = argparse.ArgumentParser(description='Process fNIRS data with GLM analysis')
    parser.add_argument('--task_name', required=True, 
                       help='Name of the task (e.g., "stroop")')
    parser.add_argument('--file_info', required=True, 
                       help='Path to CSV containing file information')
    parser.add_argument('--array_idx', required=True, type=int,
                       help='Slurm array index (0-based)')
    parser.add_argument('--stim_dur', type=float, default=1.0,
                       help='Stimulus duration for HRF modeling (default: 1.0 seconds, based on actual trial durations in data)')
    args = parser.parse_args()

    # Refuse to write into a protected run tree (analysis_variant.py).
    refuse_protected_tree(os.getenv("OUTPUT_DIR", ""))
    
    # Convert file_info path to absolute path if it's not already
    file_info_path = os.path.abspath(args.file_info)
    logger.info(f"Reading file info from: {file_info_path}")
    
    # Check if file exists
    if not os.path.exists(file_info_path):
        raise FileNotFoundError(f"File not found: {file_info_path}")
    
    # Read file information
    file_info = pd.read_csv(file_info_path)
    
    # Get the row corresponding to the array index
    if args.array_idx >= len(file_info):
        raise ValueError(f"Array index {args.array_idx} out of bounds for file_info with {len(file_info)} rows")
    
    # log the row filename
    logger.info(f"Processing row: {file_info.iloc[args.array_idx]['filename']}")
    row = file_info.iloc[args.array_idx]
    
    # Set up file logging for this subject
    log_file = setup_file_logging(row['subj_id'], row['task_id'], args.task_name)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Processing subject: {row['subj_id']}, task: {row['task_id']}, file: {row['filename']}")
    
    try:
        # Initialize analysis and process the single subject
        analysis = StroopfNIRSAnalysis(args.task_name)
        analysis.process_subject(row['subj_id'], row['task_id'], row['filename'], args.stim_dur)
        
        logger.info("Analysis completed successfully")
        
        # Create a summary log entry
        summary = {
            'subject_id': str(row['subj_id']),
            'task_id': str(row['task_id']),
            'filename': str(row['filename']),
            'status': 'SUCCESS',
            'glm_model': str(getattr(analysis, 'used_noise_model', 'unknown')),
            'design_matrix_rank': int(getattr(analysis, 'design_matrix_rank', 0)) if getattr(analysis, 'design_matrix_rank', None) is not None else 'unknown',
            'contrasts_computed': int(len(analysis.contrast_maps)) if analysis.contrast_maps else 0,
            'errors': []
        }
        
    except Exception as e:
        logger.error(f"Analysis failed for subject {row['subj_id']}: {str(e)}")
        logger.error("Full error:", exc_info=True)
        
        # Create error summary
        summary = {
            'subject_id': str(row['subj_id']),
            'task_id': str(row['task_id']), 
            'filename': str(row['filename']),
            'status': 'FAILED',
            'error': str(e),
            'error_type': str(type(e).__name__)
        }
        
        raise
    
    finally:
        # One summary file per session; the shared analysis_summary.json is
        # rebuilt from them (scripts/brain/glm_summary.py).
        summary_file = write_session_summary(
            os.path.join(os.getenv("OUTPUT_DIR"), "logs", "GLM", args.task_name), summary)
        if summary_file is not None:
            logger.info(f"Summary written to: {summary_file}")

if __name__ == "__main__":
    main()
