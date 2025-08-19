"""
Functions for creating a dataset where each sub-activity is saved into its own file.

Available Functions
-------------------
[Public]
generate_segmented_dataset(...): Generates a dataset in which all (sub)activities are segmented into their own respective files.
------------------
[Private]
_generate_json_header(...): Generates a json file containing the sensors corresponding to each column for all segmented tasks.
_save_segmented_tasks(...): Saves the segmented_tasks into individual files.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional
import json
import pandas as pd

# internal imports
from constants import VALID_ACTIVITIES, ACTIVITY_MAP, WALK, STAIRS, CABINETS, STAND,\
    VALID_SENSORS, IMU_SENSORS, ROT, \
    VALID_FILE_TYPES, NPY, CSV,\
    SEGMENTED_DATA_FOLDER, SENSOR_COLS_JSON, LOADED_SENSORS_KEY
from .segment_activities import segment_activities, crop_segments
from .load_sensor_data import load_data_from_same_recording
from file_utils import create_dir


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def generate_segmented_dataset(raw_data_path: str, segmented_data_path: str,
                               load_sensors: Optional[List[str]] = None, fs: int = 1000, # fs from mBAN = 1000 Hz
                               crop_n_seconds: int = 5,
                               plot_segment_lines: bool = False, plot_cropped_tasks: bool = False,
                               output_file_type: str = NPY) -> None:
    """
    Generates a dataset in which all (sub)activities are segmented into their own respective files using the raw data
    files. Raw data was recorded in different recording sessions containing different sub-activities.
    (1) walking: walking slow - walking medium - walking fast (approx. 5 min each | 15 min total)
    (2) stairs: stairs up - stairs down - stairs up - stairs down (approx. 1:15 min | 5 min total)
            OR: stairs up - stairs down - stairs up - stairs down - stairs up - stairs down - stairs up - stairs down (approx. 0:38 min each | total 5 mmin)
    (3) cabinets: making coffee - organizing folder (approx. 7:30 min each | 15 min total)
    (4) standing: standing still - standing while conversion - standing still (approx. 3:45 min - 7.30 min - 3:45 min | 15 min total)
    (5) sitting: sitting while working on a computer (approx. 15 min)

    Thus, the function segments, for example, the sensor data for walking into walking_slow, walking_medium, and walking_fast.

    As the raw data is stored into individual files for each sensor type (e.g., ACC, GYR, MAG, ROTATION_VECTOR), the
    function also performs the following steps before segmenting the data:
    (1) aligns the individual sensors in time.
    (2) re-samples the data to the same sampling rate.
    (3) collects all sensors into the same data structure (pandas.DataFrame or numpy.array).
    :param raw_data_path: the path to the folder containing the raw data for each subject.
    :param segmented_data_path: the path to where the segmented data should be stored. The function generates a folder
                                named 'segmented_data' in which the data for each subject is stored.
    :param load_sensors: list of sensors (as strings) indicating which sensors should be loaded. Default: None (all sensors are loaded)
    :param fs: the sampling rate to which all sensors should be re-sampled to. Default: 1000 (Hz)
    :param crop_n_seconds: the amount of seconds that should be cropped at the beginning and end of each segmented activity.
                           This is used to ensure that any transition between activities are removed. Default: 5 (seconds)
    :param plot_segment_lines: boolean that indicates whether a plot should be shown in which the obtained segmentation
                               indexes are plotted superimposed on the raw data signal. Default: False
    :param plot_cropped_tasks: boolean that indicates whether a plot should be shown in which each segmented activity
                               is plotted into a subplot. Default: False
    :param output_file_type: the file type that should be used to store the segmented files. Supported file types are:
                             '.npy' or '.csv'. Default: '.csv'
    :return: None
    """

    if load_sensors is None:
        load_sensors = VALID_SENSORS

    # check file output file type
    if output_file_type not in VALID_FILE_TYPES:

        # set output filetype to numpy file
        output_file_type = NPY

        print(f"The file type you chose is not supported. Chosen file type: {output_file_type}."
              f"\nSetting output_file_type to default: {NPY}.")

    # list all folders within the raw_data_path
    subject_folders = os.listdir(raw_data_path)

    # get the folders that contain the subject data. Subject data folders start with 'P' (e.g., P001)
    subject_folders = [folder for folder in subject_folders if folder.startswith('P')]

    # cycle over the subjects
    for subject in subject_folders:

        print("\n#----------------------------------------------------------------------#")
        print(f"# Processing data for Subject {subject}")

        # cycle over the activities
        for num, activity in enumerate(VALID_ACTIVITIES, start=1):

            print(f"\n# ({num}) {activity}:")

            # get the path to the data folder
            data_folder = os.path.join(raw_data_path, subject, activity)

            # load data into one dataframe
            print(f"# ({num}.1) data Loading")
            aligned_data = load_data_from_same_recording(data_folder, load_sensors, fs=fs)

            # segment tasks
            print(f"# ({num}.2) task segmentation")
            
            # Skip problematic files entirely
            file_path = os.path.basename(data_folder)
            if (subject == 'P013' and activity == STAIRS and 'file03' in str(data_folder)) or \
               (subject == 'P020' and activity == STAND and 'file02' in str(data_folder)):
                print(f"# Skipping problematic file: {subject}/{activity}")
                continue
            
            # Special parameter tuning for specific subjects/activities
            if subject in ['P012', 'P015', 'P017', 'P020'] and activity in [WALK, STAIRS]:
                # adapt the window size for the envelope
                segmented_tasks = segment_activities(aligned_data, activity, fs=fs, envelope_param=300,
                                                     plot_segments=plot_segment_lines)

            elif subject == 'P009' and activity == CABINETS:
                segmented_tasks = segment_activities(aligned_data, activity, fs=fs, peak_height=0.5,
                                                     plot_segments=plot_segment_lines)

            elif subject == 'P019' and activity in [CABINETS, STAND]:
                segmented_tasks = segment_activities(aligned_data, activity, fs=fs, peak_height=0.35,
                                                     peak_dist_seconds=1,
                                                     plot_segments=plot_segment_lines)
            
            # Special handling for subjects with detection issues - use smaller envelope window and lower threshold
            elif subject in ['P003', 'P006', 'P007', 'P008', 'P010', 'P011', 'P013', 'P019'] and activity in [WALK, STAIRS]:
                # Lower onset threshold and envelope param for better sensitivity
                segmented_tasks = segment_activities(aligned_data, activity, fs=fs, 
                                                     onset_threshold=0.0005, envelope_param=50,
                                                     min_segment_length_seconds=10,
                                                     plot_segments=plot_segment_lines)
            
            # Extra sensitive for P003 (cut off start issues)
            elif subject == 'P003' and activity in [WALK, STAIRS]:
                # Very low threshold for cut-off start issues
                segmented_tasks = segment_activities(aligned_data, activity, fs=fs, 
                                                     onset_threshold=0.0003, envelope_param=30,
                                                     min_segment_length_seconds=8,
                                                     plot_segments=plot_segment_lines)
            
            # Special handling for P002, P004, P007 stairs (should detect 8 segments) - lower threshold for better detection
            elif subject in ['P002', 'P004', 'P007'] and activity == STAIRS:
                segmented_tasks = segment_activities(aligned_data, activity, fs=fs, 
                                                     onset_threshold=0.0007, envelope_param=75,
                                                     min_segment_length_seconds=10,
                                                     plot_segments=plot_segment_lines)

            else:
                segmented_tasks = segment_activities(aligned_data, activity, fs=fs, plot_segments=plot_segment_lines)
            
            # Post-process specific cases
            if subject == 'P017' and activity == WALK:
                # Keep only first 3 segments, remove erratic last segment
                if len(segmented_tasks) > 3:
                    segmented_tasks = segmented_tasks[:3]
                    print(f"# Trimmed P017 walking to first 3 segments (removed erratic last segment)")
            
            elif subject == 'P020' and activity == WALK:
                # Keep only first 3 segments
                if len(segmented_tasks) > 3:
                    segmented_tasks = segmented_tasks[:3]
                    print(f"# Trimmed P020 walking to first 3 segments")
            
            elif subject == 'P015' and activity == WALK:
                # Keep only first 3 segments
                if len(segmented_tasks) > 3:
                    segmented_tasks = segmented_tasks[:3]
                    print(f"# Trimmed P015 walking to first 3 segments")
            
            elif subject == 'P016' and activity == WALK:
                # Keep only first 3 segments, remove erratic last segment
                if len(segmented_tasks) > 3:
                    segmented_tasks = segmented_tasks[:3]
                    print(f"# Trimmed P016 walking to first 3 segments (removed erratic last segment)")

            # crop 10 seconds at the beginning and the end of each signal to ensure that
            segmented_tasks = crop_segments(segmented_tasks, n_seconds=crop_n_seconds, fs=fs)

            if plot_cropped_tasks:

                # plot the segmented tasks
                fig, axes = plt.subplots(nrows=len(segmented_tasks), ncols=1, sharex=True)
                fig.suptitle(f"Segmented tasks | Activity: {activity}")

                # ensure that axes is always an iterable (needed for sitting as there is only one segment)
                axes = np.atleast_1d(axes)

                for task_df, ax in zip(segmented_tasks, axes):
                    ax.plot(task_df['y_ACC'].values)

                    # calculate the section lengths in seconds and minutes
                    sec_len_seconds = len(task_df) / fs
                    sec_len_minutes = divmod(int(sec_len_seconds), 60)

                    print(
                        f"Section length: {sec_len_seconds:.2f} s | {sec_len_minutes[0]:02}:{sec_len_minutes[1]:02} min")

                plt.show()

            # save the segmented (and pre-processed files)
            print(f"# ({num}.3) saving data")

            # create dir to save the data
            save_path = create_dir(segmented_data_path, os.path.join(SEGMENTED_DATA_FOLDER, subject))

            # save the data
            _save_segmented_tasks(segmented_tasks, activity, save_path, file_type=output_file_type)

    # create json file containing the sensor_names (only if output file is .npy
    if output_file_type == NPY:
        _generate_json_header(loaded_sensors=load_sensors, output_path=os.path.join(segmented_data_path, SEGMENTED_DATA_FOLDER))


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _generate_json_header(loaded_sensors: List[str], output_path: str) -> None:
    """
    Generates a json file containing the sensors corresponding to each column for all segmented tasks. This function
    is only needed when storing the segmented tasks as .npy files. Given that for all tasks the same sensors are loaded,
    this allows for a more memory efficient data storage, when compared to storing as .csv, as there is only one
    json-file for all tasks, thus reducing the overhead created by each .csv file.
    :param loaded_sensors: the sensors that were loaded for the generating the segmented tasks
    :param output_path: the path where the file should be stored
    :return: None
    """

    # list for holding the sensor names
    sensor_names = []

    # cycle over the loaded sensors
    for sensor_name in loaded_sensors:

        if sensor_name in IMU_SENSORS:

            sensor_names.extend([f'x_{sensor_name}', f'y_{sensor_name}', f'z_{sensor_name}'])

        elif sensor_name == ROT:

            sensor_names.extend([f'x_{sensor_name}', f'y_{sensor_name}', f'z_{sensor_name}', f'w_{sensor_name}'])

        else:

            raise ValueError(f"The following sensor is not supported: {sensor_name}"
                             f"Consider implementing it in _generate_json_header().")

    # create json file
    json_header = {LOADED_SENSORS_KEY: sensor_names}

    # create path to store the file
    json_path = os.path.join(output_path, SENSOR_COLS_JSON)

    # store the json_file
    with open(json_path, "w") as json_file:
        json.dump(json_header, json_file)


def _save_segmented_tasks(segmented_tasks: List[pd.DataFrame], activity: str, output_path: str,
                         file_type: str = '.npy') -> None:
    """
    Saves the segmented_tasks into individual files.
    :param segmented_tasks: list containing the segmented tasks.
    :param activity: the name of the activity.
    :param output_path: the path to where the segmented files should be stored.
    :param file_type: the file type of the file in which the data should be stored.
                      The following file types are supported: '.csv', '.npy'. Default: '.npy'
    :return: None
    """

    # check for valid padding type
    if file_type not in VALID_FILE_TYPES:
        raise ValueError(f"The file type you chose is not supported. Chosen file type: {file_type}."
                         f"\nPlease choose one of the following: {', '.join(VALID_FILE_TYPES)}.")

    # get the sub-activity suffixes
    sub_activity_suffixes = ACTIVITY_MAP[activity]

    # cycle over the segments
    for task_df, task_suffix in zip(segmented_tasks, sub_activity_suffixes):

        # generate file name
        file_name = f"{activity}{task_suffix}{file_type}"

        # generate full path
        file_path = os.path.join(output_path, file_name)

        # save the file
        if file_type == CSV:

            # as csv file
            task_df.to_csv(file_path, sep=';', index=False)
        else:

            # as npy file
            np.save(file_path, task_df.values)





