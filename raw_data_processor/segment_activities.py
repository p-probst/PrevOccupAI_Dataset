"""
Functions for segmenting the sub-activities (e.g., walk_slow, stairs_up, etc.) from the sensdef segment_activities(sensor_data_df: pd.DataFrame, activity: str, plot_after_sync_jump: bool = False, fs: int=1000,  envelope_type: str = RMS,
                       envelope_param: int = ENVELOPE_PARAM, min_segment_length_seconds: int = MIN_SEGMENT_LENGTH_SECONDS, peak_height: float = 0.3,
                       peak_dist_seconds: int = 2*MINUTE, plot_segments: bool = True, onset_threshold: Optional[float] = None,
                       convert_adc: bool = True, adc_bits: int = 16, adc_g_range: int = 8, file_id: Optional[str] = None) -> List[pd.DataFrame]:ata using the y-axis of
the accelerometer.

Each acquisition was acquired using a pre-defined protocol that allows for easy task segmentation.
Walking recordings:
(1) walking on a plane surface (walking_slow, walking_medium, walking_fast)
(2) walking stairs (stairs_up, stairs_down)
--> synchronization jump at the beginning
--> between each sub-activity the subject stood still for 10 seconds

Standing recordings:
(1) standing while making coffee (cabinets_coffee) and moving/retrieving objects in cabinet (cabinets_folders)
(2) standing still (standing_still) and standing while conversion (standing_talk)
--> synchronization jump at the beginning
--> between each sub-activity the subject stood still for 5 seconds, jumped, and stood still for 5 seconds

Sitting recording:
sitting while working on a computer
--> synchronization jump at the beginning

synchronization jump at the beginning: after all sensors are connected
(1) stand still for ten seconds
(2) ten jumps
(3) stand still for ten seconds
Available Functions
-------------------
[Public]
segment_activities(...): Segments the data contained in sensor_data_df into its sub-activities.
crop_segments(...): Crops the beginning and the end of each signal by n_seconds to remove potential transitions between segments
------------------
[Private]
_remove_synchronization_jump(...): Identifies a synchronization jump in the acceleration signal and removes it from the data.
_walking_onset_detection(...): gets the indices of where the walking tasks start and end based on the y-axis of the phone's accelerometer.
_get_task_indices_onset(...): gets the indices for when each walking task starts and stops.
_remove_short_segments(...): removes segments that are shorter than the set minimum segment length.
_jump_peak_detection(...): gets the indices of the jumps perfromed between standing/cabinets sub-activities.
_get_task_indices_peaks(...): generates the task indices for each performed task.
------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# internal imports
from constants import WALK, STAND, SIT, CABINETS, STAIRS
from .pre_process import pre_process_inertial_data
from .filters import get_envelope, RMS, MOVING_AVERAGE, LOW_PASS
from adc_a import adc_to_acceleration

Z_ACC = 'z_ACC'

# --- Tuned parameters for robust segmentation ---
SECONDS_AFTER_SYNC_JUMP = 15  
SECONDS_AFTER_SYNC_JUMP_SIT = 25  
SECONDS_STILL = 5
NUM_JUMPS_STAND = 3
NUM_JUMPS_CABINETS = 2
MINUTE = 60

# Lowered threshold for more sensitive onset detection
ONSET_THRESHOLD = 0.001
# Reduced minimum segment length to capture shorter segments
MIN_SEGMENT_LENGTH_SECONDS = 15  # Reduced from 30 to 15
# Smaller envelope window for better temporal resolution
ENVELOPE_PARAM = 100  # Reduced from 200 to 100

# --- Export utility ---
import os
def export_segment_to_numpy(segment_df, subactivity_name, activity, file_id=None, export_dir="exported_segments"):
    """
    Export the segment DataFrame to a numpy file in the subject's folder, named as '{activity}_{subactivity}.npy'.
    file_id should be the subject identifier (e.g., 'P003').
    """
    # Infer subject from file_id if possible
    subject = str(file_id) if file_id is not None else "unknown_subject"
    # Clean up subactivity name for filename
    safe_subactivity = subactivity_name.replace("/", "_").replace(" ", "_")
    safe_activity = activity.replace("/", "_").replace(" ", "_")
    # Build subject-specific export directory
    subject_dir = os.path.join(export_dir, subject)
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    # Save as '{activity}_{subactivity}.npy'
    npy_path = os.path.join(subject_dir, f"{safe_activity}_{safe_subactivity}.npy")
    np.save(npy_path, segment_df.to_numpy())
    print(f"[EXPORT] Saved segment '{safe_activity}_{safe_subactivity}' to {npy_path}")

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def segment_activities(sensor_data_df: pd.DataFrame, activity: str, plot_after_sync_jump: bool = False, fs: int=1000,  envelope_type: str = RMS,
                       envelope_param: int = ENVELOPE_PARAM, min_segment_length_seconds: int = MIN_SEGMENT_LENGTH_SECONDS, peak_height: float = 0.3,
                       peak_dist_seconds: int = 2*MINUTE, plot_segments: bool = True, onset_threshold: Optional[float] = None,
                       convert_adc: bool = True, adc_bits: int = 16, adc_g_range: int = 8, file_id: Optional[str] = None) -> List[pd.DataFrame]:
    """
    Segments the data contained in sensor_data_df into its sub-activities. For example: walking is segmented into
    walking_slow, walking_medium, and walking_fast.
    The following steps are performed:
    (1) remove the synchronization jump
    (2) segment the data into its sub-activities
    (a) for recordings containing walking and stairs: use onset-based activity segmentation.
    (b) for recordings containing standing and cabinets: use peak-based activity segmentation.
    (c) for the recording containing sitting: only (1) is performed.

    :param sensor_data_df: pandas.DataFrame containing the data for the entire recording.
    :param plot_after_sync_jump: boolean indicating whether to plot the z_ACC signal after removing the synchronization jump.
    :param file_id: the identifier of the file, e.g., 'P003'. Used for logging and exporting segments.
    :param activity: the name of the activity as a string.
    :param fs: the sampling frequency of the recording (in Hz). Default: 1000 (Hz)
    :param envelope_type: the type of envelope used for onset detection. The following types
                          are available:
                          'lowpass': uses a lowpass filter
                          'ma': uses a moving average filter
                          'rms': uses a root-mean-square filter
                           Default: 'rms'
    :param envelope_param: the parameter for the envelope_type. The following options are available
                           'lowpass': type_param is the cutoff frequency of the lowpass filter
                           'ma': type_param is the window size in samples
                           'rms': type_param is the window size in samples
    :param min_segment_length_seconds: the minimum length a task should be. This can be used to filter out wrongly
                                       detected (short) segments when applying onset-detection. Default: 30 (seconds)

    :param peak_height: the peak height for when applying peak-based segmentation. Default: 0.4
    :param peak_dist_seconds: the distance between peaks to avoid detecting wrong peaks. Default: 120 (seconds)
    :param plot_segments: boolean that indicates whether a plot should be shown in which the obtained segmentation
                          indexes are plotted superimposed on the raw data signal. Default: False
    :param convert_adc: boolean indicating whether to convert raw ADC values to acceleration units. Default: True
    :param adc_bits: number of ADC bits for conversion (e.g., 16 for 16-bit ADC). Default: 16
    :param adc_g_range: accelerometer range in ±g (e.g., 8 for ±8g range). Default: 8
    :return: List of pandas.DataFrames containing the segmented sub-activities.
    """

    # Print/log which file/activity is being segmented
    if file_id is not None:
        print(f"\n--- Segmenting file: {file_id} | Activity: {activity} ---")
    else:
        print(f"\n--- Segmenting Activity: {activity} ---")
    
    # Skip files shorter than 1 minute (60*fs samples)
    min_samples = 60 * fs
    if len(sensor_data_df) < min_samples:
        print(f"[SKIP] File too short (<1 min): {len(sensor_data_df)/fs:.2f} seconds, skipping.")
        return []
    
    # check whether the dataFrame contains the z-axis of the ACC
    if Z_ACC not in sensor_data_df.columns:
        raise ValueError(f"To perform task segmentation the {Z_ACC} sensor is needed. The provided data does not "
                         f"contain the needed sensor. The following sensors were provided: {sensor_data_df.columns}")

    # check whether a supported envelope type was utilized
    if envelope_type not in [RMS, LOW_PASS, MOVING_AVERAGE]:

        print(f"The envelope type you chose is not supported. Chosen envelope type: {envelope_type}."
              f"\nSetting envelope_type to default: {RMS}.")

        # set envelope_type to default
        envelope_type = RMS

    # Convert all axes to acceleration (m/s²) for mBAN data
    for axis in ['x_ACC', 'y_ACC', 'z_ACC']:
        if axis in sensor_data_df.columns:
            acc = sensor_data_df[axis].to_numpy()
            if convert_adc:
                acc = adc_to_acceleration(acc, n_bits=adc_bits, g_range=adc_g_range)
                sensor_data_df[axis] = acc
                print(f"    Converted {axis}: min={acc.min():.3f}, max={acc.max():.3f}, mean={acc.mean():.3f} m/s²")
            else:
                print(f"    Raw {axis}: min={acc.min():.0f}, max={acc.max():.0f}, mean={acc.mean():.0f}")

    # get the z-axis of the ACC (now in m/s²)
    z_acc = sensor_data_df[Z_ACC].to_numpy()

    # Remove initial artifact/spike if present
    z_acc, sensor_data_df = _remove_initial_artifact(z_acc, sensor_data_df, fs=fs, plot=False, file_id=file_id or activity)

    # Disable raw signal plot for faster processing
    # Always plot the raw signal before segmentation for visual inspection
    # fig, ax = plt.subplots(figsize=(10, 3))
    # ax.plot(np.arange(len(z_acc)) / fs, z_acc, color='tab:blue', label=f'{file_id or activity} (raw)')
    # ax.set_title(f"Raw z_ACC signal: {file_id or activity}")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Acceleration (m/s²)")
    # plt.legend()
    # plt.show()

    # remove synchronization jump from data
    print("--> removing synchronization jump")

    if activity == SIT:
        z_acc, sensor_data_df = _remove_synchronization_jump(z_acc, sensor_data_df,
                                                             jump_offset_seconds=SECONDS_AFTER_SYNC_JUMP_SIT, fs=fs,
                                                             plot=False)
        # Disable post-sync plot for faster processing
        # if plot_after_sync_jump:
        #     fig, ax = plt.subplots(figsize=(10, 3))
        #     ax.plot(np.arange(len(z_acc)) / fs, z_acc, color='tab:orange', label=f'{file_id or activity} (sync removed)')
        #     ax.set_title(f"z_ACC after sync jump removal: {file_id or activity}")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Acceleration (m/s²)")
        #     plt.legend()
        #     plt.show()
        # no further segmentation needed for sitting activity
        return [sensor_data_df]
    else:
        z_acc, sensor_data_df = _remove_synchronization_jump(z_acc, sensor_data_df,
                                                             jump_offset_seconds=SECONDS_AFTER_SYNC_JUMP, fs=fs,
                                                             plot=False)
        # Disable post-sync plot for faster processing
        # if plot_after_sync_jump:
        #     fig, ax = plt.subplots(figsize=(10, 3))
        #     ax.plot(np.arange(len(z_acc)) / fs, z_acc, color='tab:orange', label=f'{file_id or activity} (sync removed)')
        #     ax.set_title(f"z_ACC after sync jump removal: {file_id or activity}")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Acceleration (m/s²)")
        #     plt.legend()
        #     plt.show()

    # list to store the segmented tasks
    segmented_tasks = []

    # pre-process the z_acc to facilitate task segmentation
    z_acc = pre_process_inertial_data(z_acc, is_acc=True, fs=fs, normalize=True)

    # removing the impulse response of the filter (it disappears after around 2 seconds)
    z_acc = z_acc[2*fs:]
    sensor_data_df = sensor_data_df.iloc[2*fs:, :].reset_index(drop=True)

    # Determine which onset threshold to use
    if onset_threshold is not None:
        effective_onset_threshold = onset_threshold
    else:
        effective_onset_threshold = ONSET_THRESHOLD

    # Determine subactivity names based on activity and number of segments
    if activity == WALK:
        print('--> performing onset-based task segmentation')
        task_indices = _walking_onset_detection(z_acc, effective_onset_threshold, fs=fs, envelope_type=envelope_type,
                                                envelope_param=envelope_param,
                                                min_segment_length_seconds=min_segment_length_seconds)
        walk_names = ["walk_slow", "walk_medium", "walk_fast"]
        if len(task_indices) == 3:
            subactivities = walk_names
        else:
            subactivities = [f"walk_segment_{i+1}" for i in range(len(task_indices))]
    elif activity == STAIRS:
        print('--> performing onset-based task segmentation')
        task_indices = _walking_onset_detection(z_acc, effective_onset_threshold, fs=fs, envelope_type=envelope_type,
                                                envelope_param=envelope_param,
                                                min_segment_length_seconds=min_segment_length_seconds)
        stairs_names_4 = ["stairs_up_1", "stairs_down_1", "stairs_up_2", "stairs_down_2"]
        stairs_names_8 = ["stairs_up_1", "stairs_down_1", "stairs_up_2", "stairs_down_2", 
                         "stairs_up_3", "stairs_down_3", "stairs_up_4", "stairs_down_4"]
        if len(task_indices) == 4:
            subactivities = stairs_names_4
        elif len(task_indices) == 8:
            subactivities = stairs_names_8
        else:
            subactivities = [f"stairs_segment_{i+1}" for i in range(len(task_indices))]
    elif activity == CABINETS:
        print('--> performing peak-based task segmentation')
        task_indices = _jump_peak_detection(z_acc, activity, peak_height=peak_height,
                                            peak_dist_seconds=peak_dist_seconds, fs=fs)
        print(f"[DEBUG] Detected {len(task_indices)} segments for CABINETS activity.")
        cabinets_names = ["make/drink coffee", "moving objects"]
        if len(task_indices) == 2:
            subactivities = cabinets_names
        else:
            subactivities = [f"cabinets_segment_{i+1}" for i in range(len(task_indices))]
    elif activity == STAND:
        print('--> performing peak-based task segmentation')
        task_indices = _jump_peak_detection(z_acc, activity, peak_height=peak_height,
                                            peak_dist_seconds=peak_dist_seconds, fs=fs)
        print(f"[DEBUG] Detected {len(task_indices)} segments for STAND activity.")
        stand_names = ["stand still", "stand conversing", "stand still"]
        if len(task_indices) == 3:
            subactivities = stand_names
        else:
            subactivities = [f"stand_segment_{i+1}" for i in range(len(task_indices))]
    else:
        task_indices = []
        subactivities = []

    # Plot segmentation lines on the signal
    if plot_segments and len(task_indices) > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.arange(len(sensor_data_df[Z_ACC])) / fs, sensor_data_df[Z_ACC], color='teal', label=f'{file_id or activity} (segmented)')
        for idx, (start_idx, stop_idx) in enumerate(task_indices):
            color = f'C{idx%10}'
            ax.axvline(x=start_idx/fs, color=color, linestyle='--', label=f'Segment {idx+1}: {subactivities[idx] if idx < len(subactivities) else "subactivity"}')
            ax.axvline(x=stop_idx/fs, color=color, linestyle=':')
        ax.set_title(f"Segmentation: {file_id or activity}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Use non-blocking plotting for production runs
        if file_id and 'test' not in str(file_id).lower():
            # Production mode: save plot and don't block
            plt.savefig(f'segmentation_{file_id}.png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # Test mode: show plot for inspection
            plt.show()
            plt.close('all')  # Free memory after segmentation plot

    # cut the tasks out of the DataFrame
    for idx, (start_idx, stop_idx) in enumerate(task_indices):
        seg = sensor_data_df.iloc[start_idx:stop_idx, :]
        seg_name = subactivities[idx] if idx < len(subactivities) else f"segment_{idx+1}"
        seg.name = seg_name
        segmented_tasks.append(seg)

    return segmented_tasks


def crop_segments(segmented_tasks, fs=1000, n_seconds=2):
    """
    Crop the segments to remove dead space between activities.
    
    Args:
        segmented_tasks (list): List of segmented task dataframes
        fs (int): Sampling frequency (default: 1000 Hz for mBAN)
        n_seconds (float): Number of seconds to crop from each end (default: 2s)
        
    Returns:
        list: Cropped segments with reduced dead space
    """

    # calculate how many samples need to be cropped
    crop_samples = n_seconds * fs

    # cycle over the segmented task
    cropped_segments = []
    for task_pos, task in enumerate(segmented_tasks):
        if len(task) > 2 * crop_samples:
            cropped = task.iloc[crop_samples:-crop_samples, :]
            cropped_segments.append(cropped)
        else:
            print(f"Warning: Segment {task_pos+1} too short to crop ({len(task)} samples), skipping cropping.")
            cropped_segments.append(task)
    return cropped_segments


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _remove_initial_artifact(z_acc: np.ndarray, sensor_data_df: pd.DataFrame, fs: int = 1000, plot: bool = False, file_id: str = "") -> tuple:
    """
    Detects and removes an initial artifact/spike at the very start of the signal, before the true synchronization jump.
    The function scans the first 1-2 seconds for a sharp, isolated spike that is much larger than the typical signal.
    If found, removes all samples up to and including the spike.
    """
    window_sec = 2  
    window_samples = int(window_sec * fs)
    threshold_factor = 6 # increased sensitivity
    if len(z_acc) < window_samples:
        return z_acc, sensor_data_df
    # Compute median and MAD in the window
    window = z_acc[:window_samples]
    median = np.median(window)
    mad = np.median(np.abs(window - median))
    # Find spikes
    spike_indices = np.where(np.abs(window - median) > threshold_factor * mad)[0]
    if len(spike_indices) > 0:
        cut_pos = spike_indices[-1] + 1  # remove up to and including last spike
        print(f"[Artifact removal] Initial artifact detected at sample(s) {spike_indices}, cutting at {cut_pos}.")
        if plot:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(np.arange(len(z_acc)) / fs, z_acc, label='z_ACC')
            for idx in spike_indices:
                ax.axvline(x=idx/fs, color='red', linestyle='--', label='Artifact')
            ax.axvline(x=cut_pos/fs, color='purple', linestyle=':', label='Cut position')
            ax.set_title(f"Initial artifact removal: {file_id}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration (m/s²)")
            ax.legend()
            # Use non-blocking plotting
            if file_id and 'test' not in str(file_id).lower():
                plt.savefig(f'artifact_removal_{file_id}.png', dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        return z_acc[cut_pos:], sensor_data_df.iloc[cut_pos:, :].reset_index(drop=True)
    else:
        return z_acc, sensor_data_df

def _remove_synchronization_jump(y_acc: np.ndarray, sensor_data_df: pd.DataFrame, jump_offset_seconds: int,
                                 fs: int = 1000, plot: bool = False, jump_threshold: float = 20.0) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Identifies a synchronization jump in the acceleration signal and determines the position to cut the signal.
    If no jump is detected (i.e., max value below threshold), no cut is performed.
    """
    # get the position of the maximum (this should catch one of the synchronization jumps)
    jump_window = y_acc[: 3 * MINUTE * fs]
    jump_pos = np.argmax(jump_window)
    jump_val = jump_window[jump_pos]

    if jump_val < jump_threshold:
        print(f"[SYNC JUMP] No synchronization jump detected (max={jump_val:.2f} < threshold={jump_threshold}). No cut performed.")
        return y_acc, sensor_data_df

    cut_pos = jump_pos + jump_offset_seconds * fs

    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(y_acc)) / fs, y_acc, label='z_ACC')
        ax.axvline(x=float(cut_pos) / fs, color='darkgreen', label='Cut position')
        ax.axvline(x=float(jump_pos) / fs, color='red', label='Jump position')
        ax.set_title("Synchronization jump removal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration")
        ax.legend()
        # Use non-blocking plotting - save instead of show for production
        plt.savefig(f'sync_jump_removal.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SYNC JUMP] Jump detected at {jump_pos/fs:.1f}s (val={jump_val:.1f}), cutting at {cut_pos/fs:.1f}s")
    return y_acc[cut_pos:], sensor_data_df.iloc[cut_pos:, :].reset_index(drop=True)


def _walking_onset_detection(y_acc: np.ndarray, threshold: float, fs: int = 1000, envelope_type: str ='rms',
                             envelope_param: int = 100, min_segment_length_seconds: int = 30 
                             ) -> List[Tuple[int, int]]:
    """
    gets the indices of where the walking tasks start and end based on the y-axis of the phone's accelerometer.
    :param y_acc: y-axis of the phone's accelerometer signal.
    :param threshold: the threshold used to detect the onset. Should be between [0, 1]. It is best to visualize the envelope
                      of the normalized signal in order to set this onset.
    :param fs: sampling frequency
    :param envelope_type: the type of filter that should be used for getting the envelope of the signal. The following types
                          are available:
                          'lowpass': uses a lowpass filter
                          'ma': uses a moving average filter
                          'rms': uses a root-mean-square filter
    :param envelope_param: the parameter for the envelope_type. The following options are available, the original value is 100
                           'lowpass': type_param is the cutoff frequency of the lowpass filter
                           'ma': type_param is the window size in samples
                           'rms': type_param is the window size in samples
    :return: the envelope, the start indices of the onsets, and the stop indices of the onsets
    """

    # check if threshold is between 0 and 1
    if not 0 <= threshold <= 1:
        raise IOError(f"The threshold has to be between 0 and 1. Provided value: {threshold}")

    # get the abolute of the signal
    y_acc = np.abs(y_acc)

    # get the envelope of the signal
    acc_env = get_envelope(y_acc, envelope_type=envelope_type, type_param=envelope_param, fs=fs)

    # binarize the signal
    binary_onset = (acc_env >= threshold).astype(int)

    # get the start and stop indices of each walking segment
    task_indices = _get_task_indices_onset(binary_onset)

    # remove short segments
    task_indices = _remove_short_segments(task_indices, min_segment_length_seconds*fs)

    return task_indices


def _get_task_indices_onset(binary_onset: np.ndarray) -> List[Tuple[int, int]]:
    """
    gets the indices for when each walking task starts and stops.
    :param binary_onset: the binarized envelope of the signal
    :return: the start and stop indices of each performed task in a list of tuples.
    """

    # get the start and stopps of each task
    # (1) calculate the difference
    diff_sig = np.diff(binary_onset)

    # (2) get the task starts and end
    task_start = np.where(diff_sig == 1)[0]
    task_end = np.where(diff_sig == -1)[0]

    # (3) add start at the beginning and end if the onset is 1 at the beginning or the end
    if binary_onset[0] == 1:
        task_start = np.insert(task_start, 0, 0)

    if binary_onset[-1] == 1:
        task_end = np.append(task_end, len(binary_onset)-1)

    return list(zip(task_start, task_end))


def _remove_short_segments(task_indices: List[Tuple[int, int]], min_length_samples: int) -> List[Tuple[int, int]]:
    """
    removes segments that are shorter than the set minimum segment length.

    :param task_indices: list of tuples containing the (start, stops) indices of each segment.
    :param min_length_samples: the minimum segment length in samples
    :return: list of tuples containing the (start, stops) indices of each segment that are longer than the indicated
             minimum segment length.
    """

    # list for holding the corrected values
    corrected_indices = []

    # cycle over the list
    for start_idx, stop_idx in task_indices:

        # calculate the length of the segment (in samples)
        segment_length_samples = stop_idx - start_idx

        if segment_length_samples >= min_length_samples:

            corrected_indices.append((start_idx, stop_idx))

    return corrected_indices


def _jump_peak_detection(y_acc: np.ndarray, activity: str, peak_height: float = 0.3, 
                         peak_dist_seconds: int = 2*MINUTE, fs: int = 1000) -> List[Tuple[int, int]]:
    """
    gets the indices of the jumps performed between standing/cabinets sub-activities.
    :param y_acc: y-axis of the phone's accelerometer signal.
    :param activity: the name of the activity as a string
    :param peak_height: the peak height for when applying peak-based segmentation. Default: 0.4
    :param peak_dist_seconds: the distance between peaks to avoid detecting wrong peaks in samles. Default: 120 (seconds)
    :param fs: the sampling frequency of the signal.
    :return: the start and stop indices of each performed task in a list of tuples.
    """

    # calculate the peak_distance in samples
    peak_dist = peak_dist_seconds * fs

    print(f"[DEBUG] Jump detection params: peak_height={peak_height}, peak_dist_seconds={peak_dist_seconds} ({peak_dist} samples)")
    print(f"[DEBUG] Signal range: min={y_acc.min():.3f}, max={y_acc.max():.3f}")

    # find the jumping peaks
    jump_indices, _ = find_peaks(y_acc, height=peak_height, distance=peak_dist)
    
    print(f"[DEBUG] Found {len(jump_indices)} peaks with current parameters")

    # adjust peaks in case there is more/less found than the expected amount
    # (this correction is only needed for subject P001/standing and P019/cabinets)
    if activity == STAND and len(jump_indices) < NUM_JUMPS_STAND:

        # print("less then the expected amount of jumps found. Adding peak at the end of signal.")
        # add a jump at the end of the signal
        jump_indices = np.append(jump_indices, len(y_acc)-1)

    if activity == CABINETS and len(jump_indices) > NUM_JUMPS_CABINETS:

        # print("more than then expected amount of jumps found. Just considering the first and the last jump.")
        # consider only the first and the last jump
        jump_indices = jump_indices[[0, -1]]

    # get the task start and stops
    task_indices = _get_task_indices_peaks(jump_indices, fs)

    return task_indices


def _get_task_indices_peaks(jump_indices: np.ndarray, fs: int = 1000) -> List[Tuple[int, int]]:
    """
    generates the task indices for each performed task.
    :param jump_indices: the indices of the jumps.
    :param fs: the sampling frequency.
    :return: list of tuples containing the (start, stops) indices of each task/segment.
    """

    # list for holding the starts and the stops
    task_start = [0] # initializing with zero to get the beginning of the signal
    task_end = []

    # get the amount of jumps
    num_jumps = len(jump_indices)

    # cycle over the jump indices
    for num, jump_idx in enumerate(jump_indices, start=1):

        # check if last jump has not been reached yet
        if num != num_jumps:

            # append the indices for both start and end
            task_start.append(jump_idx + SECONDS_STILL * fs)
            task_end.append(jump_idx - SECONDS_STILL * fs)
        else:

            # append only end index
            task_end.append(jump_idx - SECONDS_STILL * fs)

    return list(zip(task_start, task_end))