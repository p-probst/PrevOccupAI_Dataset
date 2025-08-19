"""
Functions for creating a dataset from mBAN files where each sub-activity is saved into its own file.

Available Functions
-------------------
[Public]
generate_mban_segmented_dataset(...): Generates a dataset from mBAN files in which all (sub)activities are segmented.
------------------
[Private]
_process_mban_subject_folder(...): Process a single subject's folder containing mBAN data.
_save_mban_segmented_tasks(...): Saves the segmented mBAN tasks into individual files.
_skip_mvc_folder(...): Helper to skip mvc folders.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Optional, Tuple
import json
import pandas as pd
from tqdm import tqdm

# internal imports
from constants import VALID_ACTIVITIES, ACTIVITY_MAP, WALK, STAIRS, CABINETS, STAND, SIT, \
     MBAN_FS, MBAN_Y_ACC, SEGMENTED_DATA_FOLDER

from .segment_activities import segment_activities, crop_segments
from .load_sensor_data import get_mban_accelerometer_data
from file_utils import create_dir

# Configure logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def generate_mban_segmented_dataset(raw_data_path: str, output_path: str,
                                   fs: int = MBAN_FS, crop_n_seconds: int = 5,
                                   plot_segment_lines: bool = True, plot_cropped_tasks: bool = False) -> None:
    """
    Generates a dataset from mBAN files in which all (sub)activities are segmented into their own respective files.
    
    This function processes mBAN OpenSignals files organized in subject folders and activity subfolders.
    It extracts accelerometer data, segments activities, and saves the results as numpy files.
    
    :param raw_data_path: Path to the folder containing subject data
    :param output_path: Path where segmented data will be saved
    :param fs: Sampling frequency for processing (default: MBAN_FS = 1000 Hz)
    :param crop_n_seconds: Number of seconds to crop from beginning and end of segments
    :param plot_segment_lines: Whether to plot segmentation lines
    :param plot_cropped_tasks: Whether to plot cropped tasks
    """
    
    print("=== Starting mBAN Dataset Generation ===")
    print(f"Raw data path: {raw_data_path}")
    print(f"Output path: {output_path}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Crop seconds: {crop_n_seconds}")
    
    # Create output directory
    create_dir(output_path, SEGMENTED_DATA_FOLDER)
    segmented_output_path = os.path.join(output_path, SEGMENTED_DATA_FOLDER)
    
    # Check if raw data path exists
    if not os.path.exists(raw_data_path):
        raise ValueError(f"Raw data path does not exist: {raw_data_path}")
    
    # Look for acquisitions folder
    acquisitions_path = os.path.join(raw_data_path, 'acquisitions')
    if not os.path.exists(acquisitions_path):
        # Try using raw_data_path directly if it contains subject folders
        acquisitions_path = raw_data_path
    
    # Get all subject folders
    subject_folders = []
    for item in os.listdir(acquisitions_path):
        item_path = os.path.join(acquisitions_path, item)
        if os.path.isdir(item_path) and item.startswith('P'):  # Subject folders typically start with P
            subject_folders.append(item)
    
    if not subject_folders:
        raise ValueError(f"No subject folders found in {acquisitions_path}")
    
    subject_folders.sort()
    print(f"Found {len(subject_folders)} subjects: {subject_folders}")
    
    # Process each subject
    total_segments = 0
    for subject_id in tqdm(subject_folders, desc="Processing subjects"):
        subject_path = os.path.join(acquisitions_path, subject_id)
        # Skip mvc folder
        if subject_id.lower() == 'mvc':
            print(f"Skipping mvc folder for {subject_id}")
            continue
        print(f"\nProcessing subject: {subject_id}")
        try:
            segments_count = _process_mban_subject_folder(
                subject_path, subject_id, segmented_output_path,
                fs, crop_n_seconds, plot_segment_lines, plot_cropped_tasks
            )
            total_segments += segments_count
            print(f"Subject {subject_id}: {segments_count} segments generated")
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {str(e)}")
            continue
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Total segments generated: {total_segments}")
    print(f"Output saved to: {segmented_output_path}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _get_subject_specific_params(subject_id: str, activity_name: str) -> dict:
    """
    Get subject-specific parameters for segmentation based on visual analysis optimization.
    These parameters were fine-tuned through systematic testing of problematic files
    using visual feedback methodology to achieve optimal segmentation results.
    
    :param subject_id: Subject identifier (e.g., 'P003')
    :param activity_name: Activity name (e.g., 'stairs', 'walking', 'standing')
    :return: Dictionary of parameters to pass to segment_activities
    """
    
    # OPTIMIZED PARAMETERS based on systematic visual analysis and testing
    # Each parameter set was validated through iterative testing and plot analysis
    
    # P002: Clear 8-segment pattern in stairs, needs ultra-sensitive detection
    if subject_id == 'P002' and activity_name == 'stairs':
        return {
            'onset_threshold': 0.00002,  # Ultra-low for maximum sensitivity
            'envelope_param': 4,         # Minimal smoothing
            'min_segment_length_seconds': 3
        }
    
    # P003: Multi-activity optimizations
    elif subject_id == 'P003' and activity_name == 'stairs':
        return {
            'onset_threshold': 0.00005,  # Very sensitive for cut-off detection
            'envelope_param': 5,
            'min_segment_length_seconds': 3
        }
    elif subject_id == 'P003' and activity_name == 'walking':
        return {
            'onset_threshold': 0.00015,  # Balanced sensitivity for walking
            'envelope_param': 15,
            'min_segment_length_seconds': 5
        }
    elif subject_id == 'P003' and activity_name == 'standing':
        return {
            'peak_height': 0.4,          # Fine-tuned to detect exactly 2 jumps (3 segments)
            'peak_dist_seconds': 150     # Optimal spacing to avoid false positives
        }
    
    # P004: Clear 8-segment stairs pattern, maximum sensitivity needed
    elif subject_id == 'P004' and activity_name == 'stairs':
        return {
            'onset_threshold': 0.00001,  # Maximum sensitivity
            'envelope_param': 3,         # Minimal smoothing
            'min_segment_length_seconds': 2
        }
    
    # P006: Undersegmentation resolved with moderate sensitivity
    elif subject_id == 'P006' and activity_name == 'stairs':
        return {
            'onset_threshold': 0.0005,
            'envelope_param': 50,
            'min_segment_length_seconds': 10
        }
    
    # P007: Complex stairs patterns requiring fine-tuned sensitivity per file
    elif subject_id == 'P007' and activity_name == 'stairs':
        return {
            'onset_threshold': 0.00003,  # Balanced ultra-sensitivity
            'envelope_param': 5,
            'min_segment_length_seconds': 3
        }
    elif subject_id == 'P007' and activity_name == 'walking':
        return {
            'onset_threshold': 0.00002,  # Ultra-sensitive for walking undersegmentation
            'envelope_param': 3,
            'min_segment_length_seconds': 2
        }
    
    # P008: Successfully optimized with moderate parameters
    elif subject_id == 'P008' and activity_name == 'walking':
        return {
            'onset_threshold': 0.0002,
            'envelope_param': 20,
            'min_segment_length_seconds': 6
        }
    
    # P010: Successfully optimized with moderate parameters
    elif subject_id == 'P010' and activity_name == 'stairs':
        return {
            'onset_threshold': 0.0002,
            'envelope_param': 25,
            'min_segment_length_seconds': 6
        }
    
    # P011: Successfully optimized with moderate parameters
    elif subject_id == 'P011' and activity_name in ['stairs', 'walking']:
        return {
            'onset_threshold': 0.0005,
            'envelope_param': 50,
            'min_segment_length_seconds': 10
        }
    
    # P012: Extreme undersegmentation case, maximum sensitivity required
    elif subject_id == 'P012' and activity_name == 'stairs':
        return {
            'onset_threshold': 0.00001,  # Maximum sensitivity
            'envelope_param': 2,         # Minimal smoothing
            'min_segment_length_seconds': 2
        }
    
    # P013: Successfully optimized with moderate parameters
    elif subject_id == 'P013' and activity_name == 'walking':
        return {
            'onset_threshold': 0.0004,
            'envelope_param': 40,
            'min_segment_length_seconds': 9
        }
    
    # P015: Oversegmentation resolved with heavy smoothing
    elif subject_id == 'P015' and activity_name == 'walking':
        return {
            'envelope_param': 300  # Heavy smoothing for oversegmentation
        }
    
    # P016: Erratic end resolved with heavy smoothing
    elif subject_id == 'P016' and activity_name == 'walking':
        return {
            'envelope_param': 300  # Heavy smoothing for erratic signals
        }
    
    # P017: Erratic end resolved with heavy smoothing
    elif subject_id == 'P017' and activity_name == 'walking':
        return {
            'envelope_param': 300  # Heavy smoothing for erratic signals
        }
    
    # P019: Successfully optimized with file-specific parameters
    elif subject_id == 'P019' and activity_name == 'stairs':
        return {
            'onset_threshold': 0.0004,
            'envelope_param': 40,
            'min_segment_length_seconds': 9
        }
    elif subject_id == 'P019' and activity_name in ['cabinets', 'standing']:
        return {
            'peak_height': 0.35,
            'peak_dist_seconds': 1
        }
    
    # P020: Oversegmentation resolved with heavy smoothing
    elif subject_id == 'P020' and activity_name == 'walking':
        return {
            'envelope_param': 300  # Heavy smoothing for oversegmentation
        }
    
    # Legacy parameters for other subjects
    elif subject_id == 'P009' and activity_name == 'cabinets':
        return {'peak_height': 0.5}
    
    # Default parameters (no special handling)
    return {}


def _get_expected_segments(subject_id: str, activity_name: str) -> int:
    """
    Get the expected number of segments for a subject-activity combination
    based on protocol requirements and manual review findings.
    
    :param subject_id: Subject identifier (e.g., 'P003')
    :param activity_name: Activity name (e.g., 'stairs', 'walking', 'standing')
    :return: Expected number of segments
    """
    
    # Mapping based on protocol and manual review of problematic files
    expected_segments_map = {
        # 8-segment stairs patterns (complex cases)
        ('P002', 'stairs'): 8,
        ('P004', 'stairs'): 8,
        ('P007', 'stairs'): 8,
        
        # 4-segment stairs patterns (standard protocol)
        ('P003', 'stairs'): 4,
        ('P006', 'stairs'): 4,
        ('P010', 'stairs'): 4,
        ('P011', 'stairs'): 4,
        ('P012', 'stairs'): 4,
        ('P019', 'stairs'): 4,
        
        # 3-segment walking patterns (standard protocol)
        ('P003', 'walking'): 3,
        ('P007', 'walking'): 3,
        ('P008', 'walking'): 3,
        ('P011', 'walking'): 3,
        ('P013', 'walking'): 3,
        ('P015', 'walking'): 3,
        ('P016', 'walking'): 3,
        ('P017', 'walking'): 3,
        ('P020', 'walking'): 3,
        
        # 3-segment standing/cabinet patterns
        ('P003', 'standing'): 3,
        ('P019', 'standing'): 3,
        ('P019', 'cabinets'): 3,
    }
    
    # Get expected segments or default based on activity
    expected = expected_segments_map.get((subject_id, activity_name))
    
    if expected is None:
        # Default expectations based on activity type
        if activity_name == 'stairs':
            return 4  # Standard stairs protocol
        elif activity_name == 'walking':
            return 3  # Standard walking protocol  
        elif activity_name in ['standing', 'cabinets']:
            return 3  # Standard standing/cabinet protocol
        else:
            return 1  # Unknown activity, conservative default
    
    return expected


def _apply_post_processing(segmented_tasks: List[pd.DataFrame], subject_id: str, 
                          activity_name: str, file_id: str) -> List[pd.DataFrame]:
    """
    Apply post-processing rules for subjects that need trimming of oversegmented results.
    Based on systematic analysis of problematic files with oversegmentation and erratic endings.
    
    :param segmented_tasks: List of segmented DataFrames
    :param subject_id: Subject identifier
    :param activity_name: Activity name
    :param file_id: File identifier for logging
    :return: Post-processed list of segments
    """
    
    expected_segments = _get_expected_segments(subject_id, activity_name)
    original_count = len(segmented_tasks)
    
    # Handle oversegmentation and erratic end segments for specific subjects
    # These subjects showed consistent oversegmentation patterns that need trimming
    if subject_id in ['P015', 'P016', 'P017', 'P020'] and activity_name == 'walking':
        if original_count > expected_segments:
            # Trim to expected number of segments (keep first N segments)
            segmented_tasks = segmented_tasks[:expected_segments]
            print(f"  📏 Post-processing {file_id}: Trimmed from {original_count} to {expected_segments} segments")
    
    # Handle other specific cases that might need post-processing
    # (Can be extended based on future analysis)
    
    return segmented_tasks


def _process_mban_subject_folder(subject_path: str, subject_id: str, output_path: str,
                                fs: int=1000, crop_n_seconds: int=5, 
                                plot_segment_lines: bool=True, plot_cropped_tasks: bool=False) -> int:
    """
    Now processes ALL mBAN files for each activity independently.
    
    :param subject_path: Path to subject folder
    :param subject_id: Subject identifier (e.g., 'P001')
    :param output_path: Output path for segmented data
    :param fs: Sampling frequency
    :param crop_n_seconds: Seconds to crop from each segment
    :param plot_segment_lines: Whether to plot segmentation lines
    :param plot_cropped_tasks: Whether to plot cropped tasks
    :return: Number of segments generated for this subject
    """
    
    segments_count = 0
    
    # Get all activity folders in subject directory
    activity_folders = []
    for item in os.listdir(subject_path):
        item_path = os.path.join(subject_path, item)
        if os.path.isdir(item_path):
            activity_folders.append(item)
    
    if not activity_folders:
        print(f"No activity folders found for subject {subject_id}")
        return 0
    
    print(f"\nProcessing subject {subject_id} with activities: {activity_folders}")
    
    # Process each activity
    for activity_folder in activity_folders:
        activity_path = os.path.join(subject_path, activity_folder)
        # Map folder name to standard activity name
        activity_name = _map_folder_to_activity(activity_folder) 
        if activity_name is None:
            print(f"Unknown activity folder: {activity_folder}, skipping...")
            continue
        try:
            print(f"Processing {subject_id} - {activity_name} ({activity_folder})")
            # Load ALL mBAN data files for this activity
            mban_data_list = get_mban_accelerometer_data(activity_path) # output is a list of DataFrames
            if not mban_data_list:
                print(f"No mBAN data found for {subject_id} - {activity_name}")
                continue
            print(f"Found {len(mban_data_list)} mBAN files for {activity_name}")
            # Process each mBAN file independently
            for file_idx, sensor_data_df in enumerate(mban_data_list, 1):
                print(f"\n  Processing mBAN file {file_idx}/{len(mban_data_list)} for {activity_name}")
                print(f"  File shape: {sensor_data_df.shape}")
                if sensor_data_df.empty:
                    print(f"  Empty data in file {file_idx}, skipping...")
                    continue
                # Skip problematic files entirely based on systematic analysis
                # These files were identified during comprehensive testing of 23 problematic cases
                skip_file = False
                
                # Files excluded during systematic testing (couldn't load or severely corrupted)
                # Based on testing results that showed these files consistently fail to load
                if (subject_id == 'P013' and activity_name == 'stairs' and file_idx == 3) or \
                   (subject_id == 'P020' and activity_name == 'standing' and file_idx == 2):
                    skip_file = True
                    print(f"  🚫 Skipping known problematic file from testing: {subject_id}/{activity_name}/file{file_idx:02d}")
                
                # Additional exclusions can be added here if found during production runs
                # These should be files that consistently fail to load or cause crashes
                
                if skip_file:
                    continue
                
                # Get subject-specific parameters
                segment_params = _get_subject_specific_params(subject_id, activity_name)
                if segment_params:
                    print(f"  ⚙️  Applying optimized parameters: {segment_params}")
                else:
                    print(f"  ⚙️  Using default parameters")
                
                # Segment this file into sub-activities
                segmented_tasks = segment_activities(
                    sensor_data_df, 
                    activity_name, 
                    file_id=f"{subject_id}_file{file_idx}",
                    plot_segments=plot_segment_lines,  # Use the parameter from main
                    **segment_params  # Apply subject-specific parameters
                )
                if not segmented_tasks:
                    print(f"  ❌ No segments found in file {file_idx}")
                    continue
                
                initial_count = len(segmented_tasks)
                print(f"  🎯 Initial segmentation detected: {initial_count} segments")
                
                # Apply post-processing
                file_id = f"{subject_id}_{activity_name}_file{file_idx}"
                segmented_tasks = _apply_post_processing(segmented_tasks, subject_id, activity_name, file_id)
                
                final_count = len(segmented_tasks)
                if final_count != initial_count:
                    print(f"  ✅ Post-processing complete: {final_count} segments retained")
                else:
                    print(f"  ✅ Final result: {final_count} segments")
                # Crop the segments
                if crop_n_seconds > 0:
                    segmented_tasks = crop_segments(segmented_tasks, n_seconds=crop_n_seconds, fs=fs)
                    print(f"  Cropped segments to {crop_n_seconds} seconds each")
                # Optionally plot cropped tasks
                if plot_cropped_tasks:
                    _plot_cropped_tasks(segmented_tasks, f"{subject_id}_file{file_idx}", activity_name)
                # Save the segmented tasks with file identifier, passing the original folder name
                saved_count = _save_mban_segmented_tasks(
                    segmented_tasks, subject_id, activity_name, output_path, file_idx, activity_folder
                )
                segments_count += saved_count
                print(f"  Saved {saved_count} segments from file {file_idx}")
        except Exception as e:
            print(f"Error processing {subject_id} - {activity_folder}: {str(e)}")
            continue
    
    return segments_count

def _map_folder_to_activity(folder_name: str) -> Optional[str]:
    """
    Map folder names to standard activity names.
    
    :param folder_name: Name of the activity folder
    :return: Standard activity name or None if not recognized
    """
    folder_lower = folder_name.lower()
    
    # Direct mapping
    activity_mapping = {
        'sitting': SIT,
        'sitting_2': SIT,
        'sitting_3': SIT,
        'standing': STAND,
        'standing_2': STAND,
        'standing_3': STAND,
        'walking': WALK,
        'walking_2': WALK,
        'walking_3': WALK,
        'stairs': STAIRS,
        'cabinets': CABINETS,
        'cabinets_2': CABINETS,
        'cabinets_3': CABINETS,
    }
    
    return activity_mapping.get(folder_lower)


def _save_mban_segmented_tasks(segmented_tasks: List[pd.DataFrame], subject_id: str, 
                              activity_name: str, output_path: str, file_idx: int = 1, activity_folder: Optional[str] = None) -> int:
    """
    Save segmented mBAN tasks to individual numpy files.
    Now creates subject-specific folders and includes file identifier in filename.
    
    :param segmented_tasks: List of segmented DataFrames
    :param subject_id: Subject identifier
    :param activity_name: Activity name
    :param output_path: Base output directory path
    :param file_idx: File index for multiple mBAN files (default: 1)
    :return: Number of files saved
    """
    
    saved_count = 0
    
    # Create subject-specific folder
    subject_folder = os.path.join(output_path, subject_id)
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)
        print(f"    Created subject folder: {subject_folder}")
    
    # Assign subactivity names based on activity and segment order, matching segment_activities logic
    walk_names = ["walk_slow", "walk_medium", "walk_fast"]
    stairs_names_4 = ["stairs_up_1", "stairs_down_1", "stairs_up_2", "stairs_down_2"]
    stairs_names_8 = ["stairs_up_1", "stairs_down_1", "stairs_up_2", "stairs_down_2", 
                     "stairs_up_3", "stairs_down_3", "stairs_up_4", "stairs_down_4"]
    cabinets_names = ["drink_coffee", "moving_objects"]
    stand_names = ["stand_still_1", "stand_conversing", "stand_still_2"]
    sitting_names = ["sitting"]

    for i, task_df in enumerate(segmented_tasks):
        if task_df.empty:
            continue
        # Determine subactivity name based on activity and segment order
        if activity_name == WALK:
            if len(segmented_tasks) == 3:
                subactivity = walk_names[i]
            else:
                subactivity = f"walk_segment_{i+1}"
        elif activity_name == STAIRS:
            if len(segmented_tasks) == 4:
                subactivity = stairs_names_4[i]
            elif len(segmented_tasks) == 8:
                subactivity = stairs_names_8[i]
            else:
                subactivity = f"stairs_segment_{i+1}"
        elif activity_name == CABINETS:
            if len(segmented_tasks) == 2:
                subactivity = cabinets_names[i]
            else:
                subactivity = f"cabinets_segment_{i+1}"
        elif activity_name == STAND:
            if len(segmented_tasks) == 3:
                subactivity = stand_names[i]
            else:
                subactivity = f"stand_segment_{i+1}"
        elif activity_name == SIT:
            subactivity = sitting_names[0]
        else:
            subactivity = f"segment_{i+1}"
        # Clean up for filename
        subactivity = subactivity.replace("/", "_").replace(" ", "_")
        # Use the original folder name (e.g., cabinets_2) for the filename if provided
        activity_folder_str = activity_folder if activity_folder is not None else activity_name
        # Remove underscores for _2, _3, etc. (e.g., cabinets_2 -> cabinets2)
        if activity_folder_str.endswith(('_2', '_3')):
            activity_folder_str = activity_folder_str.replace('_', '')
        filename = f"{subject_id}_{activity_folder_str}_file{file_idx:02d}_{subactivity}.npy"
        filepath = os.path.join(subject_folder, filename)
        # Convert to numpy array and include nSeq for missing data detection
        if 'nSeq' in task_df.columns:
            data_to_save = task_df[['nSeq', 'x_ACC', 'y_ACC', 'z_ACC']].values
        elif 'time' in task_df.columns:
            data_to_save = task_df[['x_ACC', 'y_ACC', 'z_ACC']].values
        else:
            data_to_save = task_df.values
        np.save(filepath, data_to_save)
        saved_count += 1
        print(f"    Saved: {subject_id}/{filename} (shape: {data_to_save.shape})")
    
    return saved_count


def _plot_cropped_tasks(segmented_tasks: List[pd.DataFrame], subject_id: str, activity_name: str) -> None:
    """
    Plot the cropped segmented tasks for visualization.
    
    :param segmented_tasks: List of segmented DataFrames
    :param subject_id: Subject identifier
    :param activity_name: Activity name
    """
    
    if not segmented_tasks:
        return
    
    fig, axes = plt.subplots(len(segmented_tasks), 1, figsize=(12, 3 * len(segmented_tasks)))
    if len(segmented_tasks) == 1:
        axes = [axes]
    
    fig.suptitle(f'{subject_id} - {activity_name} - Cropped Segments', fontsize=14)
    
    for i, task_df in enumerate(segmented_tasks):
        if task_df.empty:
            continue
        ax = axes[i]
        # Plot accelerometer data by column name only
        if 'time' in task_df.columns:
            time_data = task_df['time'].values
        else:
            time_data = np.arange(len(task_df))
        for axis in ['x_ACC', 'y_ACC', 'z_ACC']:
            if axis in task_df.columns:
                ax.plot(time_data, task_df[axis].values, label=axis.upper(), alpha=0.7)
            else:
                print(f"Warning: {axis} not found in segment {i+1}, skipping plot for this axis.")
        ax.set_title(f'Segment {i+1} ({len(task_df)} samples)')
        ax.set_xlabel('Time' if 'time' in task_df.columns else 'Sample')
        ax.set_ylabel('Acceleration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    import matplotlib
    print(f"[DEBUG] matplotlib backend: {matplotlib.get_backend()}")
    plt.show(block=True)
    plt.close('all')  # Free memory
    print(f"[DEBUG] plt.show() called for {subject_id} - {activity_name}")


def create_mban_dataset_summary(output_path: str) -> None:
    """
    Create a summary of the generated mBAN dataset.
    Now handles subject-specific folder structure.
    
    :param output_path: Path to the segmented data folder
    """
    
    segmented_data_path = os.path.join(output_path, SEGMENTED_DATA_FOLDER)
    
    if not os.path.exists(segmented_data_path):
        print("No segmented data found")
        return
    
    # Get all subject folders
    subject_folders = [d for d in os.listdir(segmented_data_path) 
                      if os.path.isdir(os.path.join(segmented_data_path, d))]
    
    if not subject_folders:
        print("No subject folders found in segmented data folder")
        return
    
    # Analyze files in each subject folder
    summary = {}
    file_summary = {}  # Track files per activity
    total_files = 0
    
    for subject_id in sorted(subject_folders):
        subject_path = os.path.join(segmented_data_path, subject_id)
        numpy_files = [f for f in os.listdir(subject_path) if f.endswith('.npy')]
        
        if not numpy_files:
            continue
            
        summary[subject_id] = {}
        file_summary[subject_id] = {}
        
        for file in numpy_files:
            parts = file.replace('.npy', '').split('_')
            if len(parts) >= 5:  # subject_activity_file_filenum_segment_segnum
                activity = parts[1]
                file_num = parts[3]  # file01, file02, etc.
                
                if activity not in summary[subject_id]:
                    summary[subject_id][activity] = 0
                    file_summary[subject_id][activity] = set()
                
                summary[subject_id][activity] += 1
                file_summary[subject_id][activity].add(file_num)
        
        total_files += len(numpy_files)
    
    # Print summary
    print("\n=== mBAN Dataset Summary ===")
    total_segments = 0
    total_mban_files = 0
    
    for subject_id in sorted(summary.keys()):
        subject_path = os.path.join(segmented_data_path, subject_id)
        print(f"\n{subject_id}/ (folder: {subject_path})")
        subject_segments = 0
        subject_files = 0
        
        for activity in sorted(summary[subject_id].keys()):
            segment_count = summary[subject_id][activity]
            mban_file_count = len(file_summary[subject_id][activity])
            
            print(f"  {activity}: {segment_count} segments from {mban_file_count} mBAN files")
            subject_segments += segment_count
            subject_files += mban_file_count
        
        print(f"  Total: {subject_segments} segments from {subject_files} mBAN files")
        total_segments += subject_segments
        total_mban_files += subject_files
    
    print(f"\nOverall Total:")
    print(f"  Segments: {total_segments}")
    print(f"  mBAN files processed: {total_mban_files}")
    print(f"  Subjects: {len(summary)}")
    print(f"  Activities: {set(act for subj in summary.values() for act in subj.keys())}")
    
    # Show example folder structure
    print(f"\nFolder structure:")
    for subject_id in sorted(list(summary.keys())[:2]):  # Show first 2 subjects
        subject_path = os.path.join(segmented_data_path, subject_id)
        numpy_files = [f for f in os.listdir(subject_path) if f.endswith('.npy')]
        print(f"  {subject_id}/")
        for file in sorted(numpy_files[:3]):  # Show first 3 files
            print(f"    ├── {file}")
        if len(numpy_files) > 3:
            print(f"    └── ... and {len(numpy_files) - 3} more files")
