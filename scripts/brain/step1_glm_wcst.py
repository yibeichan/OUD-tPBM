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
from scipy import stats
import re
import json
from nilearn.glm.first_level import glover_hrf, spm_hrf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_names import rename_raw_to_canonical  # noqa: E402
from trial_onsets import match_trial_onsets, snirf_stim_onsets  # noqa: E402
from channel_quality import drop_unusable_channels  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subject_ids import task_events_filename  # noqa: E402
from analysis_variant import refuse_protected_tree  # noqa: E402
from glm_summary import write_session_summary  # noqa: E402
from wcst_rules import actual_rule_changes, check_trial_order, learning_end_position  # noqa: E402

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

    The behavioural script, ``scripts/behav/01_wcst_behav.py``, also strips
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


def trial_durations_from_table(event_df: pd.DataFrame) -> list[float]:
    """Return each trial's duration as recorded in the event table.

    Every WCST table carries a per-trial ``duration``, so the modulator reads
    the measured value rather than deriving one. The gap to the next trial's
    onset is not a duration: it includes the inter-trial interval and, at a
    block's final trial, the inter-block pause, and the modulator is
    mean-centred, so such values would be extreme leverage points.

    Positionally aligned with ``event_df``'s rows: the caller looks durations
    up by position, not by label.

    Raises
    ------
    ValueError
        If any trial's duration is missing or non-positive. The column is
        populated in every table, so an absence means the input changed; no
        default is substituted.
    """
    durations = pd.to_numeric(event_df["duration"], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(durations) | (durations <= 0)
    if bad.any():
        positions = np.flatnonzero(bad).tolist()
        raise ValueError(
            f"{len(positions)} trial(s) have a missing or non-positive duration, "
            f"at positions {positions[:10]}; the duration column must be populated "
            "for every trial"
        )
    return [float(d) for d in durations]


def duration_hrf_kernel(hrf_model: str, sfreq: float) -> np.ndarray:
    """Return the HRF kernel for the duration modulators, one value per sample.

    nilearn's ``glover_hrf``/``spm_hrf`` take ``t_r``, the sampling interval in
    seconds, so the interval is ``1 / sfreq``. Passing ``sfreq`` itself would
    sample the kernel at the wrong spacing (3.76 s instead of 0.27 s at this
    montage's rate), placing the peak about 0.5 s after onset instead of about
    5 s. With the interval, the modulators are convolved with the same
    HRF model as the condition regressors.
    """
    t_r = 1.0 / sfreq
    if hrf_model == 'glover':
        return glover_hrf(t_r, oversampling=1)
    if hrf_model == 'spm':
        return spm_hrf(t_r, oversampling=1)
    raise ValueError(
        f"HRF model '{hrf_model}' is not supported for the duration modulators; "
        "use 'glover' or 'spm'"
    )


class WCSTfNIRSAnalysis:
    def __init__(self, task_name: str, learning_criterion: int = 3):
        """
        Initialize WCST fNIRS Analysis.
        
        Parameters
        ----------
        task_name : str
            Name of the task (e.g., "wcst")
        learning_criterion : int
            Number of consecutive correct responses required to consider rule learned.
            Default is 3. This is a design choice for short fNIRS blocks, not
            the 10-correct category of clinical WCST scoring, and it was not
            tested by a sensitivity analysis in this repository.
        """
        self.task_name = task_name
        self.learning_criterion = learning_criterion
        self.raw = None
        self.events_df = None
        self.design_matrix = None
        self.glm_estimates = None
        self.durations = None
        self.contrasts = None
        self.design_matrix_rank = None
        self.used_noise_model = None
        self.contrast_maps = None
        self.onset_offset = None
        self.dropped_channels = []
        
    def load_data(self, snirf_file: str, events_file: str, 
                  optode_frame: str = "mri") -> None:
        """Load SNIRF and events data."""
        logger.info(f"Loading SNIRF data from {snirf_file}")
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
    
    def process_events(self) -> Tuple[np.ndarray, Dict[str, str], List[float]]:
        """Process events and create event codes for WCST conditions."""
        # Store processed events and mapping for later use
        self.processed_events = None
        self.condition_mapping = None
        self.durations = None # Will store final aligned durations

        events_new = []
        
        # Filter for start_trial events
        event_df_filtered = self.events_df[
            self.events_df['event'] == 'start_trial'
        ].sort_values('timestamp', kind='stable')
        
        # Focus on experimental block and calculate rule changes
        exp_data = event_df_filtered[
            event_df_filtered['block_type'] == 'experimental'
        ].copy()
        
        check_trial_order(exp_data)

        # A rule change is a trial whose sorting rule differs from the previous
        # experimental trial's. The table's rule_change
        # flag marks every block start, including starts that keep the rule;
        # those need no set shift and continue the current rule period. The
        # behavioural stage uses the same function (scripts/wcst_rules.py).
        is_change = actual_rule_changes(exp_data['trial_type'])
        n_flagged = int((exp_data['rule_change'] == 'yes').sum()) \
            if 'rule_change' in exp_data.columns else None
        logger.info(
            f"Detected {int(is_change.sum())} rule change events "
            f"({n_flagged} block starts flagged in the event table)"
        )

        exp_data['block_id'] = is_change.cumsum()
        exp_data['trial_position'] = 'regular' # Default to maintenance/regular
        
        # Classify trials by the learning criterion (3 consecutive correct),
        # through the function behaviour also uses (scripts/wcst_rules.py):
        # change = the first trial of a rule period; post (learning) = the
        # trials after it up to and including the criterion trial, or the rest
        # of the period if the criterion is never met; regular (maintenance)
        # = the trials after the criterion trial. The change trial does not
        # count toward the criterion. A trial with no response (missing
        # response_outcome) counts as incorrect: it breaks the streak and,
        # in maintenance, is regular_incorrect.
        for _, block in exp_data.groupby('block_id'):
            end = learning_end_position(
                (block['response_outcome'] == 1).tolist(), self.learning_criterion)
            labels = ['change'] + ['post'] * (len(block) - 1)
            if end is not None:
                labels[end + 1:] = ['regular'] * (len(block) - end - 1)
            exp_data.loc[block.index, 'trial_position'] = labels

        durations = trial_durations_from_table(event_df_filtered)
        # Durations are looked up by position in event_df_filtered, so the two
        # must stay the same length (trial_durations_from_table guarantees it;
        # this guards a later change to that contract).
        if len(durations) != len(event_df_filtered):
            raise ValueError(
                f"{len(event_df_filtered)} trial(s) but {len(durations)} duration(s): trials "
                "could not be matched to a recorded duration; durations must stay "
                "positionally aligned with the filtered event table"
            )

        # Onsets come from the SNIRF StartTrial stims, not the event table's
        # timestamp, which runs on a different clock. The match
        # raises if the counts differ, if the offset is not constant to one
        # sample, or if an onset falls outside the recording. So every trial
        # lies inside the recording and none is dropped: an onset past the end
        # means the stims and the data disagree, and stops the session rather
        # than being dropped with a warning.
        sfreq = self.raw.info['sfreq']
        n_times = len(self.raw.times) if hasattr(self.raw, "times") else None
        onsets, self.onset_offset = match_trial_onsets(
            event_df_filtered['timestamp'].to_numpy(dtype=float),
            snirf_stim_onsets(self.raw),
            sfreq,
            n_times=n_times,
        )
        logger.info(
            f"Event-table minus SNIRF onset offset: {self.onset_offset:.3f} s "
            f"({self.onset_offset * sfreq:.2f} samples); onsets taken from the SNIRF stims"
        )

        # Simplified event coding scheme:
        # 11: Regular trials (correct)
        # 12: Regular trials (incorrect)
        # 21: Change trials (all)
        # 31: Post-change trials (all)
        final_durations = []
        for position, (idx, row) in enumerate(event_df_filtered.iterrows()):
            if idx not in exp_data.index:
                continue

            sample = int(onsets[position] * sfreq)
            position_label = exp_data.loc[idx, 'trial_position']
            # No response (missing outcome) counts as incorrect.
            is_correct = row['response_outcome'] == 1

            if position_label == 'regular':
                event_code = 11 if is_correct else 12
            elif position_label == 'change':
                event_code = 21
            elif position_label == 'post':
                event_code = 31

            events_new.append([sample, 0, event_code])
            final_durations.append(durations[position])

        events_new = np.array(events_new)

        # Simplified condition mapping
        condition_mapping = {
            11: 'regular_correct',
            12: 'regular_incorrect',
            21: 'change',
            31: 'post_change'
        }
        
        logger.info(f"Created {len(events_new)} events.")

        # Store the processed events, mapping, and aligned durations
        self.processed_events = events_new
        self.condition_mapping = condition_mapping
        self.durations = final_durations # Use the aligned durations

        # Add debug information
        logger.info("\nTrial position distribution:")
        logger.info(exp_data['trial_position'].value_counts())
        
        logger.info("\nDuration statistics:")
        logger.info(f"Mean duration: {np.mean(durations):.2f}s")
        logger.info(f"Std duration: {np.std(durations):.2f}s")
        logger.info(f"Min duration: {np.min(durations):.2f}s")
        logger.info(f"Max duration: {np.max(durations):.2f}s")

        # Return values might not be strictly needed if stored in self, but keep for compatibility?
        # Let's return them for now, but note that self attributes are the source of truth now.
        return self.processed_events, self.condition_mapping, self.durations
    
    def create_design_matrix(
        self,
        drift_model: str = "cosine",
        high_pass: float = 0.01,
        hrf_model: str = "glover"
    ) -> None:
        """Create design matrix with parametric modulation for durations."""
        logger.info("Creating design matrix...")
        
        # Ensure required data is available from previous steps
        if self.processed_events is None or self.condition_mapping is None or self.durations is None:
            raise ValueError("Events, condition mapping, or durations not processed. Run process_events first.")
            
        sfreq = self.raw.info['sfreq']
        n_timepoints = self.raw.n_times
        
        # Calculate mean duration and mean-center the actual durations
        if not self.durations:
             raise ValueError("Durations list is empty.")
        mean_duration = np.mean(self.durations)
        durations_centered = np.array(self.durations) - mean_duration
        logger.info(f"Using mean duration of {mean_duration:.2f}s for base regressors.")
        logger.info(f"Mean-centered durations range: {np.min(durations_centered):.2f} to {np.max(durations_centered):.2f}s")
        
        # Create basic design matrix with mean duration for all events
        # This handles the main effects of conditions and drift
        self.design_matrix = make_first_level_design_matrix(
            self.raw,
            drift_model=drift_model,
            high_pass=high_pass,
            hrf_model=hrf_model,
            stim_dur=mean_duration # Use average duration for base regressors
        )
        
        # --- Create Parametric Modulators for Duration --- 
        logger.info("Creating parametric modulators for duration...")
        
        # 1. Choose and create the HRF kernel, sampled at the data interval
        hrf_kernel = duration_hrf_kernel(hrf_model, sfreq)

        # 2. Invert condition mapping for easy lookup
        event_id_to_name = {v: k for k, v in self.condition_mapping.items()}

        # 3. Create and convolve modulator time series for each condition
        condition_columns_to_modulate = list(self.condition_mapping.values()) # Modulate all defined conditions
        
        # Map event array indices to duration array indices
        # Assuming self.processed_events and self.durations are aligned row-by-row
        if len(self.processed_events) != len(self.durations):
            raise ValueError(f"Mismatch between number of processed events ({len(self.processed_events)}) and durations ({len(self.durations)}). Cannot create modulators.")

        for condition_name in condition_columns_to_modulate:
            logger.debug(f"Creating duration modulator for: {condition_name}")
            # Find event codes for this condition name
            event_codes = [code for code, name in self.condition_mapping.items() if name == condition_name]
            if not event_codes:
                logger.warning(f"No event code found for condition '{condition_name}'. Skipping duration modulator.")
                continue

            # Create the modulator stimulus train (zeros, with centered durations at event onsets)
            modulator_train = np.zeros(n_timepoints)
            event_indices_for_condition = np.where(np.isin(self.processed_events[:, 2], event_codes))[0]
            
            if len(event_indices_for_condition) == 0:
                 logger.debug(f"No events found for condition '{condition_name}' to modulate.")
                 continue # Skip if no events for this condition
                 
            event_samples = self.processed_events[event_indices_for_condition, 0]
            corresponding_durations = durations_centered[event_indices_for_condition]
            
            # Ensure samples are within bounds
            valid_samples_mask = (event_samples >= 0) & (event_samples < n_timepoints)
            valid_event_samples = event_samples[valid_samples_mask]
            valid_durations = corresponding_durations[valid_samples_mask]
            
            if len(valid_event_samples) != len(event_samples):
                logger.warning(f"Some event samples for '{condition_name}' were out of bounds (0, {n_timepoints}).")

            modulator_train[valid_event_samples] = valid_durations

            # Convolve with HRF
            # Use 'full' convolution and then trim to ensure alignment
            convolved_modulator = np.convolve(modulator_train, hrf_kernel, mode='full')[:n_timepoints]
            
            # Add the convolved modulator to the design matrix
            modulator_col_name = f"{condition_name}_duration"
            if modulator_col_name in self.design_matrix.columns:
                 logger.warning(f"Column '{modulator_col_name}' already exists. Overwriting.")
            self.design_matrix[modulator_col_name] = convolved_modulator
            logger.debug(f"Added modulator column: {modulator_col_name}")

        # --- Add Short Channel Regressors --- 
        logger.info("Adding short channel regressors...")
        
        # The ShortHbO and ShortHbR regressors are in every model, so a
        # session without them stops here instead of being fit without them.
        # get_short_channels returns a Raw object with only short-separation channels
        # (channels with source-detector distance < max_dist, default 10mm)
        raw_short = get_short_channels(self.raw)  # Returns Raw object, not DataFrame
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

        # --- Check Design Matrix Rank ---
        logger.info("Checking design matrix rank for collinearity issues...")
        rank = np.linalg.matrix_rank(self.design_matrix)
        n_cols = self.design_matrix.shape[1]
        self.design_matrix_rank = rank  # Store for summary
        if rank < n_cols:
            raise ValueError(f"Design matrix is rank deficient: rank={rank}, n_cols={n_cols}")
        logger.info(f"Design matrix has full rank: {rank}/{n_cols}")

        # --- Define Contrasts --- 
        # Must be done AFTER all columns are added
        logger.info("Defining contrasts...")
        n_columns = len(self.design_matrix.columns)
        self.contrasts = {}
        
        # A session with no regular-incorrect trials has no regular_incorrect
        # column. Its correct_vs_incorrect, change_vs_regular and learning are
        # still written, but they are not the contrasts their names say:
        # downstream stages treat them as missing through
        # session_info.session_lacks_regular_incorrect (variant flag
        # mask_wcst_without_regular_incorrect). Any other missing column is
        # an error.
        MAINTENANCE_CONTRASTS = ("correct_vs_incorrect", "change_vs_regular", "learning")
        if 'regular_incorrect' not in self.design_matrix.columns:
            logger.warning(
                "No regular_incorrect trials in this session: the written "
                f"{', '.join(MAINTENANCE_CONTRASTS)} contrasts omit that condition "
                "and must be masked downstream."
            )

        def get_contrast_vector(col_name):
            if col_name in self.design_matrix.columns:
                contrast_vector = np.zeros(n_columns)
                col_idx = self.design_matrix.columns.get_loc(col_name)
                contrast_vector[col_idx] = 1
                return contrast_vector
            if col_name == 'regular_incorrect':
                return np.zeros(n_columns)
            raise ValueError(f"Column '{col_name}' not found in design matrix. Cannot create contrast component.")

        # Define the contrasts using the helper function
        self.contrasts['correct_vs_incorrect'] = (
            get_contrast_vector('regular_correct') - get_contrast_vector('regular_incorrect')
        )
        self.contrasts['change_vs_regular'] = (
            get_contrast_vector('change') - 
            (get_contrast_vector('regular_correct') + get_contrast_vector('regular_incorrect')) / 2
        )
        self.contrasts['learning'] = (
            get_contrast_vector('post_change') - 
            (get_contrast_vector('regular_correct') + get_contrast_vector('regular_incorrect')) / 2
        )
        
        # Duration modulator contrast (reported as duration_effect_correct)
        self.contrasts['duration_effect_correct'] = get_contrast_vector('regular_correct_duration')

        # Log contrast information
        logger.info("Created contrasts (non-zero elements shown as column names):")

        for name, contrast in self.contrasts.items():
            logger.info(f"{name}: {contrast.shape}")

    def create_annotations(self, events: np.ndarray, condition_mapping: Dict[int, str]) -> None:
        """Create and set annotations for the raw data."""
        annot_from_events = mne.annotations_from_events(
            events=events,
            event_desc=condition_mapping,
            sfreq=self.raw.info['sfreq'],
            orig_time=self.raw.info['meas_date']
        )
        self.raw.set_annotations(annot_from_events)
    
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
        """Plot the design matrix with duration modulators."""
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_design_matrix(self.design_matrix, ax=ax)
        
        # Add labels for duration modulators
        ax.set_xlabel('Time (scans)')
        ax.set_ylabel('Regressors')
        if design_matrix_path:
            plt.savefig(design_matrix_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved design matrix plot to {design_matrix_path}")
        plt.close()
    
    def plot_events(self, event_path: str = None) -> None:
        """Plot the events with WCST-specific color coding."""
        try:
            events, event_dict = mne.events_from_annotations(self.raw)
            
            # Colours for the four modelled event codes (see condition_mapping)
            color_dict = {11: 'blue', 12: 'gray', 21: 'red', 31: 'orange'}
            color_dict = {k: v for k, v in color_dict.items() if k in event_dict.values()}

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
                        self._plot_contrast_subset(axes[hbo_ax_idx], contrast_name, 'hbo', 'r')
                    else:
                        logger.warning(f"No HbO data found for contrast {contrast_name}")
                        axes[hbo_ax_idx].set_title(f"{contrast_name} - HbO (No Data)") # Add placeholder title
                    
                    if not hbr_data.empty:
                        self._plot_contrast_subset(axes[hbr_ax_idx], contrast_name, 'hbr', 'b')
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

    def _plot_contrast_subset(self, ax, contrast_name, hb_type, color):
        """Plot one contrast's channel effects for one chromophore (filtered from self.contrast_maps)."""
        try:
            # Get contrast data
            contrast_df = pd.DataFrame.from_dict(self.contrast_maps[contrast_name], orient='index')
            
            # Filter for chromophore
            mask = contrast_df['ch_name'].str.contains(hb_type)
            data = contrast_df[mask]
            
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
                 # Extract chromophore type (adjust regex if needed)
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
    
    def process_subject(self, subj_id: str, task_id: str, filename: str) -> None:
        """Process a single subject's WCST data."""
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
        
        try:
            # Load and process data
            self.load_data(snirf_file, events_file)
            events, condition_mapping, self.durations = self.process_events()
            self.create_annotations(events, condition_mapping)
            self.create_design_matrix()
            self.run_glm_analysis()
            
            # Save plots
            self.plot_design_matrix(os.path.join(figure_dir, f"{subj_id}_{task_id}_designmatrix.png"))
            self.plot_events(os.path.join(figure_dir, f"{subj_id}_{task_id}_events.png"))
            self.plot_contrasts(os.path.join(figure_dir, f"{subj_id}_{task_id}_contrasts.png"))
            
            # Save results
            self.save_glm_results(os.path.join(glm_dir, f"{subj_id}_{task_id}_glm.pkl"))
            self.save_session_record(
                subj_id, task_id, os.path.join(glm_dir, f"{subj_id}_{task_id}_glm_session.json"))
            
        except Exception as e:
            logger.error(f"Error processing subject {subj_id}, task {task_id}: {str(e)}")
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
    parser = argparse.ArgumentParser(description='Process WCST fNIRS data with GLM analysis')
    parser.add_argument('--task_name', required=True, help='Name of the task (e.g., "wcst")')
    parser.add_argument('--file_info', required=True, help='Path to CSV containing file information')
    parser.add_argument('--array_idx', required=True, type=int, help='Slurm array index (0-based)')
    args = parser.parse_args()

    # Refuse to write into a protected run tree (analysis_variant.py).
    refuse_protected_tree(os.getenv("OUTPUT_DIR", ""))

    # Convert file_info path to absolute path
    file_info_path = os.path.abspath(args.file_info)
    logger.info(f"Reading file info from: {file_info_path}")
    
    # Check if file exists
    if not os.path.exists(file_info_path):
        raise FileNotFoundError(f"File not found: {file_info_path}")
    
    # Read file information
    file_info = pd.read_csv(file_info_path)
    
    # Get the row corresponding to the array index
    if args.array_idx >= len(file_info):
        raise ValueError(f"Array index {args.array_idx} out of bounds")
    
    # log the row filename
    logger.info(f"Processing row: {file_info.iloc[args.array_idx]['filename']}")
    row = file_info.iloc[args.array_idx]
    
    # Set up file logging for this subject
    log_file = setup_file_logging(row['subj_id'], row['task_id'], args.task_name)
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Processing subject: {row['subj_id']}, task: {row['task_id']}, file: {row['filename']}")
    
    try:
        # Initialize analysis and process the subject
        analysis = WCSTfNIRSAnalysis(args.task_name)
        analysis.process_subject(row['subj_id'], row['task_id'], row['filename'])
        
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
