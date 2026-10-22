#!/usr/bin/env python3
"""
Per-session channel-to-parcellation mapping (Step 2a).

For every SNIRF file under ``BASE_DIR/data/{stroop,wcst}`` this writes one
``mapping_{task}_{subject}_{session}.json`` under
``OUTPUT_DIR/04_channel_mappings/`` (plus ``session_index.json``), converting
between:
1. GLM flat indexing (S1_D1 format)
2. SNIRF labels (M000S00 format - uppercase)
3. Parcellation labels (m000s00 format - lowercase)

Every later brain stage uses these per-session files, because each session
has its own channel set. Run this once before step2b.

Pipeline:
    Step 1: GLM analysis (step1_glm_stroop.py, step1_glm_wcst.py)
    Step 2a: per-session GLM to parcellation mapping (this script)
    Step 2b: per-session parcellation to ROI assignment (step2b_parcellation_to_roi.py)

Usage:
    python step2a_glm_to_parcellation.py

The default run needs BASE_DIR and OUTPUT_DIR. The remaining flags are
diagnostics that write no mapping files: --validate-all and --match-sessions
compare a mapping against the GLM outputs, --check-consistency compares the
optode layouts across SNIRF files, --verbose adds detail and --snirf-file
picks the SNIRF used for labels in those diagnostics.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import h5py
import mne
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_variant import refuse_protected_tree  # noqa: E402
import channel_names  # noqa: E402
from roi_utils import extract_session_id, extract_subject_id  # noqa: E402


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

if not BASE_DIR or not OUTPUT_DIR:
    logger.error("BASE_DIR and OUTPUT_DIR must be set in .env")
    sys.exit(1)


class ChannelMapper:
    """Handle channel mapping between different formats.
    
    This class converts between three channel naming conventions:
    - GLM: S{source}_D{detector} with 1-based indices
    - SNIRF: M{module}S{source} / M{module}D{detector} (uppercase)
    - Parcellation: m{module}s{source} / m{module}d{detector} (lowercase)
    """
    
    def __init__(self, snirf_file: str):
        """Initialize mapper with SNIRF file to extract labels.
        
        Args:
            snirf_file: Path to a representative SNIRF file
        """
        self.snirf_file = snirf_file
        self.source_labels: List[str] = []
        self.detector_labels: List[str] = []
        self.measurement_pairs: List[Tuple[int, int]] = []
        self._load_snirf_labels()
        # Native optode index -> index in the full probe's label order. Parcels
        # are still resolved through this recording's own labels; only the
        # key written out is canonical (see channel_names.py).
        self._src_map, self._det_map = channel_names.index_maps(
            self.source_labels, self.detector_labels)
    
    @staticmethod
    def _decode_label(label) -> str:
        """Decode bytes to string if necessary."""
        return label.decode("utf-8") if isinstance(label, (bytes, bytearray)) else str(label)
    
    @staticmethod
    def _find_first_nirs_group(h5_file: h5py.File) -> h5py.Group:
        """Find the first nirs group in SNIRF file.
        
        SNIRF files can have nirs, nirs1, nirs2, etc.
        Returns the first one found (sorted alphabetically).
        """
        nirs_keys = sorted([k for k in h5_file.keys() if k.startswith("nirs")])
        if not nirs_keys:
            raise ValueError("No nirs group found in SNIRF file")
        return h5_file[nirs_keys[0]]
    
    def _load_snirf_labels(self):
        """Load source/detector labels and measurement list from SNIRF file."""
        logger.info(f"Loading labels from: {Path(self.snirf_file).name}")
        
        with h5py.File(self.snirf_file, 'r') as f:
            nirs_group = self._find_first_nirs_group(f)
            nirs_name = nirs_group.name.split('/')[-1]
            logger.debug(f"Using NIRS group: {nirs_name}")
            
            if "probe" not in nirs_group:
                raise ValueError(f"SNIRF file missing probe group in {nirs_name}")
            
            probe_group = nirs_group["probe"]
            
            # Load source labels
            src_labels_ds = probe_group.get("sourceLabels")
            if src_labels_ds is not None:
                for i in range(src_labels_ds.shape[0]):
                    self.source_labels.append(self._decode_label(src_labels_ds[i]))
            else:
                raise ValueError("SNIRF file missing sourceLabels")
            
            # Load detector labels
            det_labels_ds = probe_group.get("detectorLabels")
            if det_labels_ds is not None:
                for i in range(det_labels_ds.shape[0]):
                    self.detector_labels.append(self._decode_label(det_labels_ds[i]))
            else:
                raise ValueError("SNIRF file missing detectorLabels")
            
            # Load measurement pairs from measurement list
            self._load_measurement_pairs(nirs_group)
        
        logger.info(f"Loaded {len(self.source_labels)} sources, {len(self.detector_labels)} detectors")
        if self.measurement_pairs:
            logger.info(f"Found {len(self.measurement_pairs)} unique S-D pairs from measurement list")
        else:
            logger.warning("No measurement list found - will use all possible S-D combinations")
    
    def _load_measurement_pairs(self, nirs_group: h5py.Group):
        """Extract unique source-detector pairs from measurement list.
        
        The measurement list contains entries for each wavelength and chromophore,
        so we need to get unique S-D pairs.
        """
        # Try data1, data2, etc.
        data_keys = sorted([k for k in nirs_group.keys() if k.startswith("data")])
        
        if not data_keys:
            logger.warning("No data group found in SNIRF file")
            return
        
        data_group = nirs_group[data_keys[0]]
        
        # Collect measurement list entries
        ml_entries = []
        for key in sorted(data_group.keys()):
            if key.startswith("measurementList"):
                ml_group = data_group[key]
                try:
                    src_idx = int(ml_group["sourceIndex"][()])
                    det_idx = int(ml_group["detectorIndex"][()])
                    ml_entries.append((src_idx, det_idx))
                except (KeyError, ValueError) as e:
                    logger.debug(f"Skipping {key}: {e}")
        
        if ml_entries:
            # Get unique pairs, preserving measurement list order
            seen = set()
            unique_pairs = []
            for pair in ml_entries:
                if pair not in seen:
                    seen.add(pair)
                    unique_pairs.append(pair)
            self.measurement_pairs = unique_pairs
            logger.debug(f"Extracted {len(self.measurement_pairs)} unique S-D pairs")
    
    def glm_to_snirf_labels(self, source_idx: int, detector_idx: int) -> Tuple[str, str]:
        """Convert GLM indices (1-based) to SNIRF labels.
        
        Args:
            source_idx: Source index from GLM (1-based)
            detector_idx: Detector index from GLM (1-based)
            
        Returns:
            Tuple of (source_label, detector_label) in SNIRF format
            
        Raises:
            ValueError: If indices are out of range
        """
        # Validate indices
        if source_idx < 1 or detector_idx < 1:
            raise ValueError(f"Indices must be >= 1: S{source_idx}_D{detector_idx}")
        
        # Convert to 0-based indexing
        src_idx_0 = source_idx - 1
        det_idx_0 = detector_idx - 1
        
        # Get labels from SNIRF
        if src_idx_0 >= len(self.source_labels):
            raise ValueError(f"Source index {source_idx} out of range (max: {len(self.source_labels)})")
        if det_idx_0 >= len(self.detector_labels):
            raise ValueError(f"Detector index {detector_idx} out of range (max: {len(self.detector_labels)})")
        
        return self.source_labels[src_idx_0], self.detector_labels[det_idx_0]
    
    def snirf_to_parcellation_labels(self, snirf_src: str, snirf_det: str) -> Tuple[str, str]:
        """Convert SNIRF labels to parcellation format.
        
        SNIRF format: M000S00 (uppercase module M, source/detector S/D)
        Parcellation format: m000s00 (lowercase)
        
        Args:
            snirf_src: SNIRF source label (e.g., 'M000S00')
            snirf_det: SNIRF detector label (e.g., 'M000D00')
            
        Returns:
            Tuple of (source_label, detector_label) in parcellation format
        """
        def _parse(label: str, target_char: str) -> Tuple[str, str]:
            """Parse module and index from label."""
            label_lower = label.lower()
            if 'm' not in label_lower or target_char not in label_lower:
                raise ValueError(f"Unexpected label format: {label} (expected M...{target_char.upper()}...)")
            module_part = label_lower.split('m', 1)[1].split(target_char, 1)[0]
            index_part = label_lower.split(target_char, 1)[1]
            return module_part, index_part
        
        src_module, src_index = _parse(snirf_src, 's')
        det_module, det_index = _parse(snirf_det, 'd')
        
        return f"m{src_module}s{src_index}", f"m{det_module}d{det_index}"
    
    def glm_to_parcellation_labels(self, source_idx: int, detector_idx: int) -> Tuple[str, str]:
        """Convert GLM indices directly to parcellation labels.
        
        Args:
            source_idx: Source index from GLM (1-based)
            detector_idx: Detector index from GLM (1-based)
            
        Returns:
            Tuple of (source_label, detector_label) in parcellation format
        """
        snirf_src, snirf_det = self.glm_to_snirf_labels(source_idx, detector_idx)
        return self.snirf_to_parcellation_labels(snirf_src, snirf_det)
    
    def create_glm_to_parcellation_lookup(self) -> Dict[str, int]:
        """Create a lookup table from GLM channel names to parcellation indices.
        
        Returns:
            Dictionary mapping 'S{src}_D{det}' to parcellation channel index
        """
        # Load parcellation to get channel order
        parcellation_path = Path(BASE_DIR) / "data" / "atlas" / "flow2_absorption_Schaefer2018_100Parcels_7Networks.npz"
        
        if not parcellation_path.exists():
            raise FileNotFoundError(f"Parcellation file not found: {parcellation_path}")
        
        logger.info(f"Loading parcellation from: {parcellation_path.name}")
        data = np.load(parcellation_path, allow_pickle=True)
        
        parc_source_labels = [self._decode_label(v) for v in data['source_labels']]
        parc_detector_labels = [self._decode_label(v) for v in data['detector_labels']]
        
        logger.info(f"Parcellation has {len(parc_source_labels)} channels")
        
        # Create lookup from parcellation labels to indices
        parc_label_to_idx = {}
        for idx, (src, det) in enumerate(zip(parc_source_labels, parc_detector_labels)):
            parc_label_to_idx[f"{src}_{det}"] = idx
        
        # Determine which S-D pairs to map
        if self.measurement_pairs:
            candidate_pairs = self.measurement_pairs
            logger.info(f"Using {len(candidate_pairs)} measured S-D pairs")
        else:
            # Fallback: all possible combinations (less efficient)
            candidate_pairs = [
                (src_idx, det_idx)
                for src_idx in range(1, len(self.source_labels) + 1)
                for det_idx in range(1, len(self.detector_labels) + 1)
            ]
            logger.warning(f"Using all {len(candidate_pairs)} possible S-D combinations")
        
        # Create GLM to parcellation lookup
        glm_to_parc_idx = {}
        skipped_channels = []
        
        for src_idx, det_idx in candidate_pairs:
            # Native key until the canonical one resolves, so an out-of-range
            # native index is still reported under a name in skipped_channels.
            glm_key = f"S{src_idx}_D{det_idx}"
            
            try:
                glm_key = f"S{self._src_map[src_idx]}_D{self._det_map[det_idx]}"
                # Native indices: parcels resolve through this recording's labels.
                parc_src, parc_det = self.glm_to_parcellation_labels(src_idx, det_idx)
                parc_key = f"{parc_src}_{parc_det}"
                
                if parc_key in parc_label_to_idx:
                    glm_to_parc_idx[glm_key] = parc_label_to_idx[parc_key]
                else:
                    skipped_channels.append((glm_key, parc_key, "not in parcellation"))
            except (KeyError, ValueError) as exc:
                skipped_channels.append((glm_key, None, str(exc)))
        
        logger.info(f"Created lookup with {len(glm_to_parc_idx)} channel mappings")
        
        if skipped_channels:
            logger.debug(f"Skipped {len(skipped_channels)} channels (likely short channels)")
            if len(skipped_channels) <= 10:
                for glm_key, parc_key, reason in skipped_channels:
                    logger.debug(f"  {glm_key} -> {parc_key}: {reason}")
        
        return glm_to_parc_idx
    
    def get_stats(self) -> Dict:
        """Get statistics about the mapper configuration."""
        return {
            'n_sources': len(self.source_labels),
            'n_detectors': len(self.detector_labels),
            'n_measurement_pairs': len(self.measurement_pairs),
            'snirf_file': str(self.snirf_file),
            'first_source_label': self.source_labels[0] if self.source_labels else None,
            'first_detector_label': self.detector_labels[0] if self.detector_labels else None,
        }


def find_matching_snirf_for_glm(glm_file: Path, task: str = 'stroop') -> Optional[Path]:
    """Find the SNIRF file that matches a GLM file by session ID.
    
    Args:
        glm_file: Path to GLM contrasts CSV file
        task: Task type ('stroop' or 'wcst')
        
    Returns:
        Path to matching SNIRF file or None
    """
    glm_session_id = extract_session_id(glm_file.name)
    
    task_dir = Path(BASE_DIR) / "data" / task
    if not task_dir.exists():
        return None
    
    for snirf_file in task_dir.glob("*.snirf"):
        snirf_session_id = extract_session_id(snirf_file.name)
        if snirf_session_id == glm_session_id:
            return snirf_file
    
    return None


def validate_mapping_with_glm_detailed(mapper: ChannelMapper, glm_file: Path, 
                                       lookup: Dict[str, int], verbose: bool = False,
                                       check_session_match: bool = True) -> Dict:
    """Enhanced validation with detailed diagnostics.
    
    Args:
        mapper: ChannelMapper instance
        glm_file: Path to GLM contrasts CSV file
        lookup: GLM to parcellation lookup table
        verbose: Print detailed diagnostics
        check_session_match: Check if SNIRF and GLM are from same session
        
    Returns:
        Dictionary with detailed validation statistics
    """
    # Check session match
    if check_session_match and verbose:
        mapper_session = extract_session_id(Path(mapper.snirf_file).name)
        glm_session = extract_session_id(glm_file.name)
        
        if mapper_session != glm_session:
            logger.warning(f"\n⚠️ SESSION MISMATCH:")
            logger.warning(f"  SNIRF session: {mapper_session} ({Path(mapper.snirf_file).name})")
            logger.warning(f"  GLM session:   {glm_session} ({glm_file.name})")
            logger.warning(f"  These should be from the same recording!")
    
    glm_data = pd.read_csv(glm_file)
    hbo_data = glm_data[glm_data['chromophore'] == 'hbo'].copy()
    
    # Get unique channels from one contrast
    contrasts = hbo_data['contrast'].unique()
    if len(contrasts) == 0:
        return {'error': 'No contrasts found', 'file': str(glm_file)}
    
    contrast_data = hbo_data[hbo_data['contrast'] == contrasts[0]]
    
    # Get channel sets
    glm_channels = set(
        f"S{int(row['Source'])}_D{int(row['Detector'])}"
        for _, row in contrast_data.iterrows()
    )
    mapped_channels = set(lookup.keys())
    
    # Analyze overlaps
    mapped_and_in_glm = glm_channels & mapped_channels
    mapped_not_in_glm = mapped_channels - glm_channels  # In mapping but not GLM
    in_glm_not_mapped = glm_channels - mapped_channels  # In GLM but not mapping
    
    result = {
        'file': glm_file.name,
        'total_glm_channels': len(glm_channels),
        'total_mapping_entries': len(mapped_channels),
        'successfully_validated': len(mapped_and_in_glm),
        'in_mapping_not_glm': len(mapped_not_in_glm),
        'in_glm_not_mapped': len(in_glm_not_mapped),
        'validation_rate': len(mapped_and_in_glm) / len(glm_channels) if len(glm_channels) > 0 else 0,
        'sample_unmapped_in_glm': list(in_glm_not_mapped)[:10]
    }
    
    if verbose:
        logger.info(f"\n{'='*70}")
        logger.info(f"DETAILED VALIDATION: {glm_file.name}")
        logger.info(f"{'='*70}")
        logger.info(f"Total mapping entries:        {len(mapped_channels)}")
        logger.info(f"Total GLM channels:           {len(glm_channels)}")
        logger.info(f"Successfully validated:       {len(mapped_and_in_glm)}")
        logger.info(f"In mapping but not GLM:       {len(mapped_not_in_glm)} ⚠️" if mapped_not_in_glm else "In mapping but not GLM:       0")
        logger.info(f"In GLM but not mapped:        {len(in_glm_not_mapped)}")
        
        if mapped_not_in_glm:
            logger.warning(f"\n⚠️ {len(mapped_not_in_glm)} channels in mapping don't appear in GLM:")
            logger.warning(f"Sample: {list(mapped_not_in_glm)[:10]}")
            logger.warning("Possible causes:")
            logger.warning("  - GLM from different session/subject")
            logger.warning("  - Channels marked as bad during preprocessing")
            logger.warning("  - Channels excluded due to quality metrics")
        
        if in_glm_not_mapped:
            logger.info(f"\n{len(in_glm_not_mapped)} GLM channels not in mapping (likely short channels):")
            logger.info(f"Sample: {list(in_glm_not_mapped)[:10]}")
    
    return result


def validate_mapping_with_glm(mapper: ChannelMapper, glm_file: Path, 
                              lookup: Dict[str, int]) -> Dict:
    """Validate the mapping against a single GLM file (legacy compatibility).
    
    Args:
        mapper: ChannelMapper instance
        glm_file: Path to GLM contrasts CSV file
        lookup: GLM to parcellation lookup table
        
    Returns:
        Dictionary with validation statistics
    """
    result = validate_mapping_with_glm_detailed(mapper, glm_file, lookup, verbose=False)
    
    # Convert to legacy format for compatibility
    return {
        'file': result.get('file', glm_file.name),
        'total_channels': result.get('total_glm_channels', 0),
        'mapped_channels': result.get('successfully_validated', 0),
        'unmapped_channels': result.get('in_glm_not_mapped', 0),
        'mapping_rate': result.get('validation_rate', 0),
        'unmapped_examples': result.get('sample_unmapped_in_glm', [])
    }


def check_optode_consistency(base_dir: Path, task: str = 'stroop', sample_size: int = 5) -> Dict:
    """Check if optode layouts are consistent across subjects.
    
    Args:
        base_dir: Base data directory
        task: Task to check ('stroop' or 'wcst')
        sample_size: Number of files to sample
        
    Returns:
        Dictionary with consistency analysis
    """
    task_dir = Path(base_dir) / "data" / task
    if not task_dir.exists():
        return {'error': f'Task directory not found: {task_dir}'}
    
    snirf_files = list(task_dir.glob("*.snirf"))[:sample_size]
    if not snirf_files:
        return {'error': 'No SNIRF files found'}
    
    configs = []
    for snirf_file in snirf_files:
        try:
            mapper = ChannelMapper(str(snirf_file))
            config = {
                'file': snirf_file.name,
                'n_sources': len(mapper.source_labels),
                'n_detectors': len(mapper.detector_labels),
                'n_pairs': len(mapper.measurement_pairs),
                'first_source': mapper.source_labels[0] if mapper.source_labels else None,
                'first_detector': mapper.detector_labels[0] if mapper.detector_labels else None
            }
            configs.append(config)
        except Exception as e:
            configs.append({'file': snirf_file.name, 'error': str(e)})
    
    # Check consistency
    valid_configs = [c for c in configs if 'error' not in c]
    if not valid_configs:
        return {'error': 'No valid configurations found', 'configs': configs}
    
    config_tuples = [(c['n_sources'], c['n_detectors'], c['n_pairs']) for c in valid_configs]
    is_consistent = len(set(config_tuples)) == 1
    
    result = {
        'files_checked': len(configs),
        'is_consistent': is_consistent,
        'configs': configs,
        'unique_configurations': len(set(config_tuples))
    }
    
    if not is_consistent:
        logger.warning("\n⚠️ WARNING: Optode configurations vary across subjects!")
        logger.warning("Consider creating subject-specific mappings.")
    else:
        logger.info("\n✓ All subjects have identical optode configurations.")
    
    return result


def validate_mapping_all_subjects(mapper: ChannelMapper, lookup: Dict[str, int]) -> Dict:
    """Validate mapping against all available GLM files.
    
    Args:
        mapper: ChannelMapper instance
        lookup: GLM to parcellation lookup table
        
    Returns:
        Dictionary with aggregate validation statistics
    """
    logger.info("\nValidating mapping against all GLM files...")
    
    results = []
    
    for task in ['stroop', 'wcst']:
        glm_dir = Path(OUTPUT_DIR) / "03_glm" / task
        if not glm_dir.exists():
            logger.warning(f"GLM directory not found: {glm_dir}")
            continue
        
        glm_files = list(glm_dir.glob("*_glm_contrasts.csv"))
        logger.info(f"Found {len(glm_files)} GLM files for {task}")
        
        for glm_file in glm_files:
            result = validate_mapping_with_glm(mapper, glm_file, lookup)
            result['task'] = task
            results.append(result)
    
    if not results:
        return {'error': 'No GLM files found for validation'}
    
    # Aggregate statistics
    total_files = len(results)
    total_channels = sum(r.get('total_channels', 0) for r in results)
    total_mapped = sum(r.get('mapped_channels', 0) for r in results)
    
    # Check consistency across files
    all_unmapped = set()
    for r in results:
        all_unmapped.update(r.get('unmapped_examples', []))
    
    return {
        'total_files_validated': total_files,
        'total_channels': total_channels,
        'total_mapped': total_mapped,
        'overall_mapping_rate': total_mapped / total_channels if total_channels > 0 else 0,
        'unmapped_channels_sample': list(all_unmapped)[:10],
        'per_file_results': results
    }


def find_representative_snirf() -> Path:
    """Find a representative SNIRF file for extracting labels.
    
    Returns the first available SNIRF file from stroop or wcst data.
    """
    for task in ['stroop', 'wcst']:
        task_dir = Path(BASE_DIR) / "data" / task
        if task_dir.exists():
            snirf_files = list(task_dir.glob("*.snirf"))
            if snirf_files:
                logger.info(f"Using SNIRF file from {task}: {snirf_files[0].name}")
                return snirf_files[0]
    
    raise FileNotFoundError("No SNIRF files found in data/stroop or data/wcst")


def check_consistency_report(args) -> None:
    """Log whether optode layouts agree across SNIRF files."""
    logger.info(f"\n{'='*70}")
    logger.info("CHECKING OPTODE CONFIGURATION CONSISTENCY")
    logger.info(f"{'='*70}")
    
    for task in ['stroop', 'wcst']:
        logger.info(f"\nChecking {task.upper()} data:")
        consistency_result = check_optode_consistency(Path(BASE_DIR), task=task, sample_size=10)
        
        if 'error' in consistency_result:
            logger.error(f"  Error: {consistency_result['error']}")
        else:
            logger.info(f"  Files checked: {consistency_result['files_checked']}")
            logger.info(f"  Unique configurations: {consistency_result['unique_configurations']}")
            logger.info(f"  Consistent: {'✓ Yes' if consistency_result['is_consistent'] else '⚠️ No'}")
            
            if args.verbose:
                logger.info("\n  Configuration details:")
                for cfg in consistency_result['configs']:
                    if 'error' not in cfg:
                        logger.info(f"    {cfg['file']}: {cfg['n_sources']} sources, "
                                   f"{cfg['n_detectors']} detectors, {cfg['n_pairs']} pairs")
    logger.info("")


def run_per_session() -> None:
    """Write one mapping file per SNIRF session plus an index."""
    logger.info(f"\n{'='*70}")
    logger.info("GENERATING PER-SESSION MAPPING FILES")
    logger.info(f"{'='*70}")
    
    # Create output directory for mappings
    mapping_dir = Path(OUTPUT_DIR) / "04_channel_mappings"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create an index file to list all sessions
    session_index = {
        'description': 'Index of per-session channel mappings',
        'sessions': []
    }
    
    total_created = 0
    total_failed = 0
    
    for task in ['stroop', 'wcst']:
        task_dir = Path(BASE_DIR) / "data" / task
        if not task_dir.exists():
            logger.warning(f"Task directory not found: {task_dir}")
            continue
    
        snirf_files = list(task_dir.glob("*.snirf"))
        logger.info(f"\nProcessing {len(snirf_files)} SNIRF files for {task.upper()}...")
    
        for snirf_file in snirf_files:
            session_id = extract_session_id(snirf_file.name)
        
            subject_id = extract_subject_id(snirf_file.name)

            try:
                # Create mapper for this session
                session_mapper = ChannelMapper(str(snirf_file))
                session_lookup = session_mapper.create_glm_to_parcellation_lookup()
                session_stats = session_mapper.get_stats()
            
                # Save session-specific mapping file
                session_output_file = mapping_dir / f"mapping_{task}_{subject_id}_{session_id}.json"
            
                # Convert to serializable format
                lookup_list = []
                for glm_key, parc_idx in sorted(session_lookup.items()):
                    src, det = glm_key.split('_')
                    lookup_list.append({
                        'glm_channel': glm_key,
                        'source': int(src[1:]),
                        'detector': int(det[1:]),
                        'parcellation_index': int(parc_idx)
                    })
            
                output_data = {
                    'description': f'GLM to parcellation channel mapping for session {session_id}',
                    'session_id': session_id,
                    'subject_id': subject_id,
                    'task': task,
                    'total_channels': len(lookup_list),
                    'snirf_file': snirf_file.name,
                    'mapper_config': {
                        'n_sources': session_stats['n_sources'],
                        'n_detectors': session_stats['n_detectors'],
                        'n_measurement_pairs': session_stats['n_measurement_pairs']
                    },
                    'mappings': lookup_list
                }
            
                with open(session_output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
            
                # Add to index
                session_index['sessions'].append({
                    'session_id': session_id,
                    'subject_id': subject_id,
                    'task': task,
                    'mapping_file': session_output_file.name,
                    'snirf_file': snirf_file.name,
                    'n_channels': len(lookup_list)
                })
            
                total_created += 1
                logger.info(f"  ✓ {session_id} ({subject_id}): {len(lookup_list)} channels → {session_output_file.name}")
            
            except Exception as e:
                total_failed += 1
                logger.error(f"  ✗ {snirf_file.name}: {e}")
    
    # Save index file
    index_file = mapping_dir / "session_index.json"
    with open(index_file, 'w') as f:
        json.dump(session_index, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("PER-SESSION MAPPING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Mapping files created:    {total_created}")
    logger.info(f"Failed:                   {total_failed}")
    logger.info(f"Output directory:         {mapping_dir}")
    logger.info(f"Index file:               {index_file}")

    # A session without a mapping would silently drop out of every later
    # stage, so a failed session fails the stage.
    if total_failed:
        logger.error(f"{total_failed} session(s) failed; see the errors above.")
        sys.exit(1)
    
    # Statistics across sessions
    if session_index['sessions']:
        channel_counts = [s['n_channels'] for s in session_index['sessions']]
        logger.info(f"\nChannel counts across sessions:")
        logger.info(f"  Min:  {min(channel_counts)}")
        logger.info(f"  Max:  {max(channel_counts)}")
        logger.info(f"  Mean: {sum(channel_counts)/len(channel_counts):.0f}")


def main():
    """Main entry point for channel mapping."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create channel mapping between GLM and parcellation'
    )
    parser.add_argument(
        '--validate-all', 
        action='store_true',
        help='Validate mapping against all GLM files (slower but thorough)'
    )
    parser.add_argument(
        '--match-sessions',
        action='store_true',
        help='Match SNIRF and GLM files by session ID for accurate validation'
    )
    parser.add_argument(
        '--per-session',
        action='store_true',
        help='Generate per-session mapping files (the default; accepted for compatibility)'
    )
    parser.add_argument(
        '--check-consistency',
        action='store_true',
        help='Check if optode configurations are consistent across subjects'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed validation diagnostics'
    )
    parser.add_argument(
        '--snirf-file',
        type=str,
        default=None,
        help='Specific SNIRF file to use for labels (default: auto-detect)'
    )
    args = parser.parse_args()
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    
    # Default: per-session mapping files. Diagnostics run only when asked for.
    diagnostics = args.match_sessions or args.validate_all
    if not diagnostics:
        if args.check_consistency:
            check_consistency_report(args)
        run_per_session()
        return

    # Diagnostics need one SNIRF for labels
    if args.snirf_file:
        snirf_file = Path(args.snirf_file)
    else:
        snirf_file = find_representative_snirf()
    
    if not snirf_file.exists():
        logger.error(f"SNIRF file not found: {snirf_file}")
        sys.exit(1)
    
    if args.check_consistency:
        check_consistency_report(args)

    # Create mapper
    logger.info(f"\n{'='*70}")
    logger.info("CHANNEL MAPPING: GLM → Parcellation")
    logger.info(f"{'='*70}")
    
    mapper = ChannelMapper(str(snirf_file))
    mapper_stats = mapper.get_stats()
    
    # Test a few conversions
    logger.info(f"\nTest conversions:")
    test_cases = [(1, 1), (1, 7), (10, 20)]
    for src, det in test_cases:
        try:
            snirf_src, snirf_det = mapper.glm_to_snirf_labels(src, det)
            parc_src, parc_det = mapper.glm_to_parcellation_labels(src, det)
            logger.info(f"  S{src}_D{det} → {snirf_src}_{snirf_det} → {parc_src}_{parc_det}")
        except ValueError as e:
            logger.warning(f"  S{src}_D{det} → Error: {e}")
    
    # Create lookup table
    lookup = mapper.create_glm_to_parcellation_lookup()
    
    # Diagnostics (write no mapping files)
    if args.match_sessions:
        # Validate each GLM with its matching SNIRF file
        logger.info(f"\n{'='*70}")
        logger.info("SESSION-MATCHED VALIDATION")
        logger.info(f"{'='*70}")
        logger.info("Validating each GLM file against its matching SNIRF file...")
        
        matched_results = []
        unmatched_glm = []
        
        for task in ['stroop', 'wcst']:
            glm_dir = Path(OUTPUT_DIR) / "03_glm" / task
            if not glm_dir.exists():
                continue
            
            glm_files = list(glm_dir.glob("*_glm_contrasts.csv"))
            logger.info(f"\nProcessing {len(glm_files)} GLM files for {task.upper()}...")
            
            for glm_file in glm_files:
                matching_snirf = find_matching_snirf_for_glm(glm_file, task)
                
                if matching_snirf:
                    # Create mapper for this specific SNIRF file
                    session_mapper = ChannelMapper(str(matching_snirf))
                    session_lookup = session_mapper.create_glm_to_parcellation_lookup()
                    
                    # Validate
                    result = validate_mapping_with_glm_detailed(
                        session_mapper, glm_file, session_lookup, 
                        verbose=args.verbose, check_session_match=True
                    )
                    result['task'] = task
                    result['matched_snirf'] = matching_snirf.name
                    result['session_id'] = extract_session_id(glm_file.name)
                    matched_results.append(result)
                    
                    if not args.verbose:
                        rate_pct = result['validation_rate'] * 100
                        status = "✓" if result['in_mapping_not_glm'] == 0 else "⚠️"
                        logger.info(f"  {status} {glm_file.name}: {rate_pct:.1f}% (unmapped: {result['in_glm_not_mapped']})")
                else:
                    unmatched_glm.append(glm_file.name)
                    logger.warning(f"  ✗ {glm_file.name}: No matching SNIRF found")
        
        # Summary
        if matched_results:
            avg_rate = sum(r['validation_rate'] for r in matched_results) / len(matched_results)
            perfect_matches = sum(1 for r in matched_results if r['in_mapping_not_glm'] == 0)
            
            logger.info(f"\n{'='*70}")
            logger.info("SESSION-MATCHED VALIDATION SUMMARY")
            logger.info(f"{'='*70}")
            logger.info(f"Total GLM files validated:    {len(matched_results)}")
            logger.info(f"Unmatched GLM files:          {len(unmatched_glm)}")
            logger.info(f"Perfect session matches:      {perfect_matches}/{len(matched_results)}")
            logger.info(f"Average mapping rate:         {avg_rate:.1%}")
            
            # Show worst matches (likely still mismatched)
            problem_results = [r for r in matched_results if r['in_mapping_not_glm'] > 0]
            if problem_results:
                logger.warning(f"\n⚠️ {len(problem_results)} files still have channels in mapping but not in GLM:")
                for r in sorted(problem_results, key=lambda x: x['in_mapping_not_glm'], reverse=True)[:5]:
                    logger.warning(f"  {r['file']}: {r['in_mapping_not_glm']} channels in mapping not in GLM")
        
        validation_stats = {
            'total_files_validated': len(matched_results),
            'unmatched_files': len(unmatched_glm),
            'overall_mapping_rate': avg_rate if matched_results else 0,
            'per_file_results': matched_results
        }
    
    elif args.validate_all:
        validation_stats = validate_mapping_all_subjects(mapper, lookup)
        
        logger.info(f"\nValidation Summary:")
        logger.info(f"  Files validated: {validation_stats['total_files_validated']}")
        logger.info(f"  Overall mapping rate: {validation_stats['overall_mapping_rate']:.1%}")
        
        if validation_stats.get('unmapped_channels_sample'):
            logger.info(f"  Sample unmapped channels: {validation_stats['unmapped_channels_sample'][:5]}")
        
        # Run detailed validation on first file if verbose
        if args.verbose:
            logger.info("\n--- DETAILED VALIDATION DIAGNOSTICS ---")
            for task in ['stroop', 'wcst']:
                glm_dir = Path(OUTPUT_DIR) / "03_glm" / task
                if glm_dir.exists():
                    glm_files = list(glm_dir.glob("*_glm_contrasts.csv"))
                    if glm_files:
                        validate_mapping_with_glm_detailed(mapper, glm_files[0], lookup, verbose=True)
                        break


if __name__ == "__main__":
    main()
