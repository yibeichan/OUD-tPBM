#!/usr/bin/env python3
"""
Per-session channel-to-ROI mapping for fNIRS GLM results (Step 2b).

Reads the per-session mapping files written by step2a
(``OUTPUT_DIR/04_channel_mappings/mapping_*.json``) and assigns each channel
to a Schaefer-atlas network/ROI, writing ``roi_mapping_*.json`` beside them.

Usage:
    python step2b_parcellation_to_roi.py

Pass ``--mapping-file FILE`` to convert one aggregate GLM mapping file into
``channel_roi_mapping.json`` instead (a diagnostic; no later stage reads it).

Pipeline:
    Step 1: GLM analysis (step1_glm_stroop.py, step1_glm_wcst.py)
    Step 2a: per-session GLM to parcellation mapping (step2a_glm_to_parcellation.py)
    Step 2b: per-session parcellation to ROI assignment (this script)
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_variant import refuse_protected_tree  # noqa: E402

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
    logger.error("BASE_DIR and OUTPUT_DIR must be set in .env file")
    sys.exit(1)


def load_schaefer_labels(label_file: Path) -> Dict:
    """Load Schaefer parcellation labels.
    
    Args:
        label_file: Path to Schaefer label file
        
    Returns:
        Dictionary mapping parcel index to ROI info
    """
    parcel_info = {}
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                idx = int(parts[0])
                label = parts[1]
                
                # Parse label: e.g., '7Networks_LH_Vis_1'
                network_parts = label.split('_')
                if len(network_parts) >= 4:
                    hemisphere = network_parts[1]
                    network = network_parts[2]
                    roi_num = network_parts[3]
                    
                    parcel_info[idx] = {
                        'label': label,
                        'hemisphere': hemisphere,
                        'network': network,
                        'network_roi': f"{network}_{roi_num}",
                        'full_roi': f"{hemisphere}_{network}_{roi_num}"
                    }
    
    logger.info(f"Loaded {len(parcel_info)} Schaefer parcels")
    return parcel_info


def create_channel_roi_mapping(glm_mapping: Dict, membership: np.ndarray, 
                               parcel_info: Dict) -> Dict:
    """Create channel to ROI mapping using GLM mapping and parcellation data.
    
    Args:
        glm_mapping: GLM to parcellation mapping from step2a
        membership: Channel-to-parcel membership matrix
        parcel_info: Schaefer parcel information
        
    Returns:
        Complete ROI mapping dictionary
    """
    mapping = {
        'channel_to_network': {},
        'channel_to_network_roi': {},
        'channel_to_full_roi': {},
        'channel_to_parcel_idx': {},
        'network_to_channels': {},
        'network_roi_to_channels': {},
        'full_roi_to_channels': {},
        'roi_statistics': {}
    }
    
    # Process each GLM channel
    mapped_count = 0
    network_counts = {}
    
    for channel_map in glm_mapping['mappings']:
        glm_channel = channel_map['glm_channel']
        parc_idx = channel_map['parcellation_index']
        
        # Get parcel membership for this channel
        channel_membership = membership[parc_idx]
        
        # Find parcel with highest membership (excluding background at index 0)
        parcel_weights = channel_membership[1:]  # Skip background
        if np.any(parcel_weights > 0):
            best_parcel = np.argmax(parcel_weights) + 1
            
            if best_parcel in parcel_info:
                mapped_count += 1
                info = parcel_info[best_parcel]
                
                # Channel to ROI mappings
                mapping['channel_to_network'][glm_channel] = info['network']
                mapping['channel_to_network_roi'][glm_channel] = info['network_roi']
                mapping['channel_to_full_roi'][glm_channel] = info['full_roi']
                mapping['channel_to_parcel_idx'][glm_channel] = int(best_parcel)
                
                # Track network counts
                network_counts[info['network']] = network_counts.get(info['network'], 0) + 1
                
                # ROI to channels mappings
                if info['network'] not in mapping['network_to_channels']:
                    mapping['network_to_channels'][info['network']] = []
                mapping['network_to_channels'][info['network']].append(glm_channel)
                
                if info['network_roi'] not in mapping['network_roi_to_channels']:
                    mapping['network_roi_to_channels'][info['network_roi']] = []
                mapping['network_roi_to_channels'][info['network_roi']].append(glm_channel)
                
                if info['full_roi'] not in mapping['full_roi_to_channels']:
                    mapping['full_roi_to_channels'][info['full_roi']] = []
                mapping['full_roi_to_channels'][info['full_roi']].append(glm_channel)
    
    # Calculate statistics
    total_channels = glm_mapping['total_channels']
    mapping['roi_statistics'] = {
        'total_channels': total_channels,
        'mapped_channels': mapped_count,
        'mapping_percentage': 100.0 * mapped_count / total_channels if total_channels > 0 else 0,
        'unique_networks': len(mapping['network_to_channels']),
        'unique_network_rois': len(mapping['network_roi_to_channels']),
        'unique_full_rois': len(mapping['full_roi_to_channels']),
        'network_distribution': network_counts
    }
    
    return mapping


def process_single_mapping(glm_mapping_file: Path, membership: np.ndarray, 
                          parcel_info: Dict, output_file: Path) -> Dict:
    """Process a single GLM mapping file and create ROI mapping.
    
    Args:
        glm_mapping_file: Path to GLM mapping JSON
        membership: Channel-to-parcel membership matrix
        parcel_info: Schaefer parcel information
        output_file: Path to save ROI mapping
        
    Returns:
        ROI mapping dictionary
    """
    # Load GLM mapping
    with open(glm_mapping_file, 'r') as f:
        glm_mapping = json.load(f)
    
    logger.info(f"Processing {glm_mapping_file.name}: {glm_mapping['total_channels']} channels")
    
    # Create ROI mapping
    mapping = create_channel_roi_mapping(glm_mapping, membership, parcel_info)
    
    # Add session info if available
    if 'session_id' in glm_mapping:
        mapping['session_id'] = glm_mapping['session_id']
    if 'subject_id' in glm_mapping:
        mapping['subject_id'] = glm_mapping['subject_id']
    if 'task' in glm_mapping:
        mapping['task'] = glm_mapping['task']
    if 'snirf_file' in glm_mapping:
        mapping['snirf_file'] = glm_mapping['snirf_file']
    
    # Save mapping
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    return mapping


def process_per_session_mappings(mapping_dir: Path, membership: np.ndarray,
                                 parcel_info: Dict) -> Dict:
    """Process all per-session mapping files.
    
    Args:
        mapping_dir: Directory containing per-session mappings
        membership: Channel-to-parcel membership matrix
        parcel_info: Schaefer parcel information
        
    Returns:
        Summary statistics
    """
    # Find all mapping files (not the index file)
    mapping_files = [f for f in mapping_dir.glob("mapping_*.json")]
    
    if not mapping_files:
        logger.error(f"No mapping files found in {mapping_dir}")
        return {}
    
    logger.info(f"Found {len(mapping_files)} session mapping files")
    
    # Process each session
    results = []
    failed = []
    roi_output_dir = mapping_dir  # Save ROI mappings in same directory
    
    for glm_mapping_file in sorted(mapping_files):
        # Create corresponding ROI mapping filename
        roi_output_file = roi_output_dir / glm_mapping_file.name.replace('mapping_', 'roi_mapping_')
        
        try:
            mapping = process_single_mapping(
                glm_mapping_file, membership, parcel_info, roi_output_file
            )
            
            stats = mapping['roi_statistics']

            # step2a writes session_id, subject_id and task into every
            # per-session mapping file. Without them the session-index match
            # below cannot succeed, so a mapping file that lacks them is
            # unusable and fails here; the enclosing except logs it per file
            # and the remaining sessions still process. No placeholder such
            # as 'unknown' is substituted.
            missing = [
                key for key in ('session_id', 'subject_id', 'task')
                if not mapping.get(key)
            ]
            if missing:
                raise ValueError(
                    f"missing required field(s) {missing}; regenerate with "
                    f"step2a_glm_to_parcellation.py"
                )

            results.append({
                'file': glm_mapping_file.name,
                'output': roi_output_file.name,
                'session_id': mapping['session_id'],
                'subject_id': mapping['subject_id'],
                'task': mapping['task'],
                'total_channels': stats['total_channels'],
                'mapped_channels': stats['mapped_channels'],
                'mapping_percentage': stats['mapping_percentage'],
                'unique_networks': stats['unique_networks']
            })
            
            logger.info(f"  ✓ {roi_output_file.name}: {stats['mapped_channels']}/{stats['total_channels']} "
                       f"({stats['mapping_percentage']:.1f}%) mapped to {stats['unique_networks']} networks")
            
        except Exception as e:
            logger.error(f"  ✗ {glm_mapping_file.name}: {e}")
            failed.append(glm_mapping_file.name)
    
    # Update session index with ROI mapping info
    index_file = mapping_dir / "session_index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            session_index = json.load(f)
        
        # Add ROI mapping column
        for session in session_index.get('sessions', []):
            session_id = session['session_id']
            matching_result = next((r for r in results if r['session_id'] == session_id), None)
            if matching_result:
                session['roi_mapping_file'] = matching_result['output']
                session['roi_mapped_channels'] = matching_result['mapped_channels']
                session['roi_mapping_percentage'] = matching_result['mapping_percentage']
            else:
                logger.error(
                    f"  No ROI mapping result for session {session_id}; its "
                    f"ROI columns in {index_file.name} are left unset. "
                    f"Check the per-file errors above."
                )
                failed.append(session_id)
        
        session_index['roi_mappings_generated'] = True
        
        with open(index_file, 'w') as f:
            json.dump(session_index, f, indent=2)
        
        logger.info(f"Updated session index: {index_file}")

    # A session without an ROI mapping would silently drop out of every later
    # stage, so a failed session fails the stage (after the index is written,
    # so the successful sessions can be inspected).
    if failed:
        raise RuntimeError(f"{len(failed)} session ROI mapping(s) failed: {sorted(set(failed))}")
    
    # Summary statistics
    summary = {
        'total_sessions': len(results),
        'total_channels_all_sessions': sum(r['total_channels'] for r in results),
        'total_mapped_all_sessions': sum(r['mapped_channels'] for r in results),
        'avg_mapping_percentage': sum(r['mapping_percentage'] for r in results) / len(results) if results else 0,
        'per_session_results': results
    }
    
    return summary


def main():
    """Run channel to ROI mapping."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create channel-to-ROI mapping from parcellation'
    )
    parser.add_argument(
        '--per-session',
        action='store_true',
        help='Process per-session mapping files (the default; accepted for compatibility)'
    )
    parser.add_argument(
        '--mapping-file',
        type=str,
        default=None,
        help='Convert this one aggregate GLM mapping file to channel_roi_mapping.json instead of the per-session run'
    )
    args = parser.parse_args()
    try:
        refuse_protected_tree(OUTPUT_DIR)
    except RuntimeError as exc:
        sys.exit(f"ABORT: {exc}")
    
    # Load parcellation data
    parc_file = Path(BASE_DIR) / "data" / "atlas" / "flow2_absorption_Schaefer2018_100Parcels_7Networks.npz"
    if not parc_file.exists():
        logger.error(f"Parcellation file not found: {parc_file}")
        sys.exit(1)
    
    parc_data = np.load(parc_file)
    membership = parc_data['channel_to_parcel_membership']
    logger.info(f"Loaded parcellation: {membership.shape[0]} channels, {membership.shape[1]} parcels")
    
    # Load Schaefer labels
    label_file = Path(BASE_DIR) / "data" / "atlas" / "Schaefer2018_100Parcels_7Networks_order.txt"
    if not label_file.exists():
        logger.error(f"Label file not found: {label_file}")
        sys.exit(1)
    
    parcel_info = load_schaefer_labels(label_file)
    
    if not args.mapping_file:
        # Process per-session mappings
        logger.info(f"\n{'='*70}")
        logger.info("PROCESSING PER-SESSION ROI MAPPINGS")
        logger.info(f"{'='*70}")
        
        mapping_dir = Path(OUTPUT_DIR) / "04_channel_mappings"
        if not mapping_dir.exists():
            logger.error(f"Mapping directory not found: {mapping_dir}")
            logger.error("Run step2a_glm_to_parcellation.py first")
            sys.exit(1)
        
        summary = process_per_session_mappings(mapping_dir, membership, parcel_info)
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("PER-SESSION ROI MAPPING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Sessions processed:       {summary['total_sessions']}")
        logger.info(f"Total channels:           {summary['total_channels_all_sessions']}")
        logger.info(f"Total mapped:             {summary['total_mapped_all_sessions']}")
        logger.info(f"Average mapping rate:     {summary['avg_mapping_percentage']:.1f}%")
        logger.info(f"\nROI mapping files saved to: {mapping_dir}")
    
    else:
        # Convert one aggregate mapping file
        logger.info(f"\n{'='*70}")
        logger.info("PROCESSING SINGLE ROI MAPPING")
        logger.info(f"{'='*70}")
        
        glm_mapping_file = Path(args.mapping_file)
        
        if not glm_mapping_file.exists():
            logger.error(f"GLM mapping file not found: {glm_mapping_file}")
            logger.error("Run step2a_glm_to_parcellation.py first")
            sys.exit(1)
        
        output_file = Path(OUTPUT_DIR) / "04_channel_mappings" / "channel_roi_mapping.json"
        mapping = process_single_mapping(glm_mapping_file, membership, parcel_info, output_file)
        
        # Print summary
        stats = mapping['roi_statistics']
        logger.info(f"\nMapping Summary:")
        logger.info(f"- Total channels:   {stats['total_channels']}")
        logger.info(f"- Mapped channels:  {stats['mapped_channels']} ({stats['mapping_percentage']:.1f}%)")
        logger.info(f"- Unique networks:  {stats['unique_networks']}")
        logger.info(f"- Unique ROIs:      {stats['unique_full_rois']}")
        
        if 'network_distribution' in stats:
            logger.info(f"\nNetwork distribution:")
            for network, count in sorted(stats['network_distribution'].items()):
                logger.info(f"  {network}: {count} channels")
        
        logger.info(f"\nMapping saved to: {output_file}")


if __name__ == "__main__":
    main()
