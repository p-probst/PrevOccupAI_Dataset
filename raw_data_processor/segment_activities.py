"""
Functions for segmenting the sub-activities (e.g., walk_slow, stairs_up, etc.) from the sensor data using the y-axis of
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
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# internal imports
from constants import WALK, STAND, SIT, CABINETS, STAIRS
from .pre_process import pre_process_inertial_data
from .filters import get_envelope, RMS, MOVING_AVERAGE, LOW_PASS

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
Y_ACC = 'y_ACC'
SECONDS_AFTER_SYNC_JUMP = 15
SECONDS_AFTER_SYNC_JUMP_SIT = 25
SECONDS_STILL = 5
NUM_JUMPS_STAND = 3
NUM_JUMPS_CABINETS = 2
MINUTE = 60

ONSET_THRESHOLD = 0.01


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def segment_activities(sensor_data_df: pd.DataFrame, activity: str, fs: int,  envelope_type: str = RMS,
                       envelope_param: int = 100, min_segment_length_seconds: int = 30, peak_height: float = 0.4,
                       peak_dist_seconds: int = 2*MINUTE, plot_segments: bool = False) -> List[pd.DataFrame]:
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
    :param activity: the name of the activity as a string.
    :param fs: the sampling frequency of the recording (in Hz). Default: 100 (Hz)
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
    :return: List of pandas.DataFrames containing the segmented sub-activities.
    """

    # check whether the dataFrame contains the y-axis of the ACC
    if Y_ACC not in sensor_data_df.columns:
        raise ValueError(f"To perform task segmentation the {Y_ACC} sensor is needed. The provided data does not "
                         f"contain the needed sensor. The following sensors were provided: {sensor_data_df.columns}")

    # check whether a supported envelope type was utilized
    if envelope_type not in [RMS, LOW_PASS, MOVING_AVERAGE]:

        print(f"The envelope type you chose is not supported. Chosen envelope type: {envelope_type}."
              f"\nSetting envelope_type to default: {RMS}.")

        # set envelope_type to default
        envelope_type = RMS

    # get the y-axis if the ACC
    y_acc = sensor_data_df[Y_ACC].to_numpy()

    # remove synchronization jump from data
    # for sit the offset is increased as the people had to sit down after performing the jumps

    print("--> removing synchronization jump")
    if activity == SIT:
        y_acc, sensor_data_df = _remove_synchronization_jump(y_acc, sensor_data_df,
                                                             jump_offset_seconds=SECONDS_AFTER_SYNC_JUMP_SIT, fs=fs,
                                                             plot=plot_segments)

        # no further segmentation needed for sitting activity
        return [sensor_data_df]
    else:
        y_acc, sensor_data_df = _remove_synchronization_jump(y_acc, sensor_data_df,
                                                             jump_offset_seconds=SECONDS_AFTER_SYNC_JUMP, fs=fs,
                                                             plot=False)

    # list to store the segmented tasks
    segmented_tasks = []

    # pre-process the y_acc to facilitate task segmentation
    y_acc = pre_process_inertial_data(y_acc, is_acc=True, fs=fs, normalize=True)

    # removing the impulse response of the filter (it disappears after around 2 seconds)
    y_acc = y_acc[2*fs:]
    sensor_data_df = sensor_data_df.iloc[2*fs:, :].reset_index(drop=True)

    # checking for the type of activity
    if activity in [WALK, STAIRS]:

        print('--> performing onset-based task segmentation')

        # perform onset-based task segmentation
        task_indices = _walking_onset_detection(y_acc, ONSET_THRESHOLD, fs=fs, envelope_type=envelope_type,
                                                envelope_param=envelope_param,
                                                min_segment_length_seconds=min_segment_length_seconds)

    else:

        print('--> performing peak-based task segmentation')

        # perform peak-based task segmentation
        task_indices = _jump_peak_detection(y_acc, activity, peak_height=peak_height,
                                            peak_dist_seconds=peak_dist_seconds, fs=fs)
    if plot_segments:
        # create figure
        fig, ax = plt.subplots()
        plt.title(f"Segmented Activity: {activity}")

        # plot y-Acc
        ax.plot(sensor_data_df[Y_ACC].values, color='teal', label='acc-signal')

        # plot vertical lines at jump and cut
        for start_idx, stop_idx in task_indices:
            ax.axvline(x=start_idx, color='darkgreen')
            ax.axvline(x=stop_idx, color='darkorange')

        plt.legend()
        plt.show()

    # cut the tasks out of the DataFrame
    for start_idx, stop_idx in task_indices:

        segmented_tasks.append(sensor_data_df.iloc[start_idx:stop_idx, :])

    return segmented_tasks


def crop_segments(segmented_tasks: List[pd.DataFrame], n_seconds: int = 10, fs: int = 100) -> List[pd.DataFrame]:
    """
    Crops the beginning and the end of each signal by n_seconds to remove potential transitions between segments.
    :param segmented_tasks: list of pandas.DataFrames containing the segmented tasks.
    :param n_seconds: the amount of seconds that should be cropped at the beginning and end of each segment.
    :param fs: the sampling frequency of the data.
    :return: list of pandas.DataFrames containing the segmented and cropped tasks.
    """

    # calculate how many samples need to be cropped
    crop_samples = n_seconds * fs

    # cycle over the segmented task
    for task_pos, task in enumerate(segmented_tasks):

        # crop the task & overwrite the list
        segmented_tasks[task_pos] = task.iloc[crop_samples:-crop_samples, :]

    return segmented_tasks


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _remove_synchronization_jump(y_acc: np.array, sensor_data_df: pd.DataFrame, jump_offset_seconds: int,
                                 fs: int = 100, plot: bool = False) -> Tuple[np.array, pd.DataFrame]:
    """
    Identifies a synchronization jump in the acceleration signal and determines the position to cut the signal.
    The function locates the maximum acceleration value within the first three minutes of the recording, assuming
    it corresponds to a synchronization jump. It then calculates a new cut position by adding an offset (in seconds)
    to the detected jump position. Calculations are done using the accelerometer's y-axis as in the used setup it
    measures vertical displacement.
    
    If visualization is enabled, it plots the acceleration signal with vertical
    markers at the detected jump and cut positions.

    :param y_acc: array containing the accelerometer's y-axis.
    :param sensor_data_df: pandas.DataFrame containing the data (all loaded sensors) for the entire recording.
    :param jump_offset_seconds: the offset (after the jump) used to indicate where to cut the signal.
    :param fs: the sampling frequency of the recording (in Hz). Default: 100 (Hz)
    :param plot: boolean indicating whether to generate a plot that visualizes where the signal is going to be cut.
    :return: the y-axis acceleration and the sensor data with the synchronization jumps removed.
    """

    # get the position of the maximum (this should catch one of the synchronization jumps)
    # (it is assumed that the synchronization jumps are within the first 3 min of the recording)
    jump_pos = np.argmax(y_acc[: 3 * MINUTE * fs])

    # get the position where to cut the signal
    cut_pos = jump_pos + jump_offset_seconds * fs

    # visualize the cut (to verify whether the cut is performed correctly)
    if plot:
        # create figure
        fig, ax = plt.subplots()

        # plot y-Acc
        sensor_data_df.plot(y=Y_ACC, ax=ax, label='acc-signal')

        # plot vertical lines at jump and cut
        # ax.axvline(x=jump_pos, color='red', label='jump max')

        ax.axvline(x=cut_pos, color='darkgreen')
        ax.axvline(x=sensor_data_df.index[-1], color='darkorange')

        plt.show()

    return y_acc[cut_pos:], sensor_data_df.iloc[cut_pos:, :].reset_index(drop=True)


def _walking_onset_detection(y_acc: np.array, threshold: float, fs: int = 100, envelope_type: str ='rms',
                             envelope_param: int = 100, min_segment_length_seconds: int = 30) -> List[Tuple[int, int]]:
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
    :param envelope_param: the parameter for the envelope_type. The following options are available
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


def _get_task_indices_onset(binary_onset: np.array) -> List[Tuple[int, int]]:
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


def _jump_peak_detection(y_acc: np.array, activity: str, peak_height: float = 0.4,
                         peak_dist_seconds: int = 2*MINUTE, fs: int = 100) -> List[Tuple[int, int]]:
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

    # find the jumping peaks
    jump_indices, _ = find_peaks(y_acc, height=peak_height, distance=peak_dist)

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


def _get_task_indices_peaks(jump_indices: np.array, fs: int = 100) -> List[Tuple[int, int]]:
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
