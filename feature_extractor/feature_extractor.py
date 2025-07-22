"""
Functions for extracting features and windowing signal.

Available Functions
-------------------
[Public]
extract_features(...): Extracts features for all subjects and activities.
load_data(...): loads the data located at full_file_path.
extract_tsfel_features(...): Extracts features from the data windows contained in windowed_data using TSFEL.
extract_quaternion_features(...): Extracts quaterion-based features from the rotation vector sensor.
------------------
[Private]
_generate_outfolder(...): Generates the folders for storing the data.
_load_sensor_names(...): Loads the sensor names from a json file containing it.
_pre_process_sensors(...): Pre-processes the sensors contained in data_array according to their sensor type.
_extract_features(...): Extracts features from the windowed data.
_generate_labels(...): Gets the labels for the main and sub-activity corresponding to the file name.
_save_subject_features(...): Saves the features extracted for a subject.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import json
import tsfel
from pathlib import Path

# internal imports
from constants import VALID_ACTIVITIES, \
    VALID_FILE_TYPES, NPY, CSV, \
    VALID_SENSORS, ACC, GYR, MAG, ROT, \
    SENSOR_COLS_JSON, LOADED_SENSORS_KEY, CLASS_INSTANCES_JSON, MAIN_LABEL_KEY, SUB_LABEL_KEY,\
    ACTIVITY_MAIN_SUB_CLASS, MAIN_CLASS_KEY
from raw_data_processor import slerp_smoothing, pre_process_inertial_data
from .window import get_sliding_windows_indices, window_data, window_scaling, validate_scaler_input
from .quaternion_features import geodesic_distance
from file_utils import (remove_file_duplicates, create_dir, load_json_file, save_json_file, validate_activity_input,
                        get_labels)


# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
TSFEL_CONFIG_FILE = 'cfg_file.json'


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def extract_features(data_path: str, features_data_path: str, activities: List[str] = None, fs: int = 100,
                     window_size: float = 1.5, overlap: float = 0.5, window_scaler: str = None,
                     output_file_type: str = NPY, default_input_file_type: str = NPY) -> None:
    """
    Extracts features for all subjects and activities contained in data_path. Features are extracted window-wise.
    (1) TSFEL: defined in cfg_file.json
    (2) Quaternion-based: mean, std, and total geodesic distance.

    The features of each subject (for all activities) are stored together int a file that is named after the subject
    (e.g., 'P001'). The last two columns in those files correspond to the main and sub-activity label, while the other
    columns correspond to the extracted features. Files can be either stored as .npy or .csv.

    In addition to the subject feature files one json file is created that contains the names of the extracted features
    as well as the number of instances for each main and sub-activity label per subject. This information can later be
    used to balance the amount instances per subject/main class/sub-class.

    :param data_path: the path to the data. This should point to the folder containing the segmented tasks.
    :param features_data_path: the path to where the features should be stored. Within this path a folder called
                               'extracted_features' is created. Within this folder a further folder that specifies the
                               used window size and window scaler is created. Within this folder all the features are stored.
    :param activities: list containing the activities from which features should be loaded. Default: None (in this case
                       all activities are loaded).
    :param fs: the sampling rate (in Hz) of the data. Default: 100
    :param window_size: the window size in seconds that should be used for windowing the data. Default: 1.5
    :param overlap: the overlap between consecutive windows. The value has to be between [0, 1]. Default: 0.5
    :param window_scaler: the type of scaler that should be used for scaling each window. The following can be chosen
                          'minmax' (min-max scaler), 'standard' (standardize), None (no scaling applied). Default: None
    :param output_file_type: the file type in which the features should be stored. It can be either '.csv' or '.npy'.
                             Default: '.npy'
    :param default_input_file_type: The default input type that should be used. This is used to make sure that only one
                                    file is loaded in case the activity data has been stored as both '.npy' and '.csv'.
                                    It can be either '.csv' or '.npy'. Default: '.npy'
    :return: None
    """

    # check if there were no activities passed
    if activities is None:
        activities = VALID_ACTIVITIES

    # check validity of provided activities
    activities = validate_activity_input(activities)

    # check validity of the provided scaler if not None
    if window_scaler:
        validate_scaler_input(window_scaler)

    # check output file type
    if output_file_type not in VALID_FILE_TYPES:
        # set output filetype to numpy file
        output_file_type = default_input_file_type

        print(f"The file type you chose is not supported. Chosen file type: {output_file_type}."
              f"\nSetting output_file_type to default: {default_input_file_type}.")

    # list all subject folders
    subject_folders = os.listdir(data_path)

    # get the folders that contain the subject data. Subject data folders start with 'P' (e.g., P001)
    subject_folders = [folder for folder in subject_folders if folder.startswith('P')]

    # get the features to be extracted TSFEL
    features_dict = load_json_file(os.path.join(Path(__file__).parent, TSFEL_CONFIG_FILE))

    # json_dict for holding the information of the class instances and the features extracted
    json_dict = {}

    # generate output path (folder) where all the feature files are stored
    output_path = _generate_outfolder(features_data_path, window_scaler, int(window_size*fs))

    # cycle over the subjects
    for sub_num, subject in enumerate(subject_folders): # subject_folders
        print("\n#----------------------------------------------------------------------#")
        print(f"# Extracting features for Subject {subject}")

        # get the path to the subject folder
        subject_folder_path = os.path.join(data_path, subject)

        # list for holding extracted feature DataFrames
        feature_df_list = []

        # cycle over the activities
        for num, activity in enumerate(activities, start=1):

            print(f"\n# ({num}) {activity}:")

            # list all files in the path
            files = os.listdir(subject_folder_path)

            # get files that belong to the activity
            files = [file for file in files if activity in file]

            # TODO: eventually add check for the file type of the files and add this to the if as a condition
            #  (i.e., only consider loading files with the correct file types otherwise skip)
            if files:

                # remove duplicate files
                # (e.g., 'walk_slow.npy' and walk_slow.csv'  --> keep only the file that has the default input type)
                files = remove_file_duplicates(files, default_input_file_type=default_input_file_type)

                # get the number of files
                num_files = len(files)

                # cycle over the files and load the data
                for file_num, file in enumerate(files, start=1):

                    # (1) load the data
                    print(f"({file_num}.1) loading file {file_num}/{num_files}: {file}")
                    data, sensor_names = load_data(os.path.join(subject_folder_path, file))

                    # remove time column
                    data = data[:, 1:]

                    # (2) pre-process the data
                    print(f"({file_num}.2) pre-processing")
                    data = _pre_process_sensors(data, sensor_names)

                    # remove impulse response
                    data = data[250:, :]

                    # (3) window the data
                    # (since all are of the same length it is possible to use just one sensor channel)
                    print(f"({file_num}.3) windowing data")
                    indices = get_sliding_windows_indices(data[:, 0], fs=fs, window_size=window_size, overlap=overlap)
                    windowed_data = window_data(data, indices)

                    # normalize windows (this is only applied to ACC, GYR, and MAG as applying this to quaternions
                    # would distort they geometric properties
                    if window_scaler:

                        # scale each window
                        windowed_data[:, :, :-4] = window_scaling(windowed_data[:, :, :-4], scaler=window_scaler)

                    # (4) extract features and labels
                    print(f"({file_num}.4) extracting features and labels")
                    features_df = _extract_features(windowed_data, sensor_names, features_dict, fs=fs)

                    # get labels
                    labels_df = _generate_labels(file, windowed_data.shape[0])

                    # concatenate the features and the labels
                    features_df = pd.concat([features_df, labels_df], axis=1)

                    # get the extracted features and store them in the json dict
                    if sub_num == 0:

                        json_dict.update({'feature_cols': features_df.columns.to_list()})

                    # append the dataFrame to the list
                    feature_df_list.append(features_df)

            else:

                print(f"No files found for activity: {activity}. Skipping feature extraction for this activity.")

        # TODO: potential bug if no files are found
        # concatenate the features for all activities into a single file
        subject_features = pd.concat(feature_df_list, axis=0, ignore_index=True)

        # get info on number instances of 'main_label' and 'sub_label'
        counts_main_label = subject_features[MAIN_LABEL_KEY].value_counts()
        counts_sub_label = subject_features[SUB_LABEL_KEY].value_counts()

        # create dictionary containing the class instances for each subject
        class_instances = counts_main_label.to_dict()
        class_instances.update(counts_sub_label.to_dict())

        json_dict.update({subject: class_instances})

        print('saving subject data')
        _save_subject_features(subject_features, subject, output_path, file_type=output_file_type)

    # save the json_dict
    save_json_file(json_dict, CLASS_INSTANCES_JSON, output_path)
    print("finished feature extraction")


def load_data(full_file_path: str) -> np.array:
    """
    loads the data located at full_file_path.
    :param full_file_path: the path to the file to be loaded
    :return: a numpy.array containing the loaded data
    """

    # retrieve the file type from the path
    file_type = os.path.splitext(full_file_path)[1]

    # load the file
    if file_type == NPY:

        activity_data = np.load(full_file_path)

        # get the sensor names
        sensor_names = _load_sensor_names(full_file_path)

    else:  # file_type == CSV

        activity_data = pd.read_csv(full_file_path, sep=';')

        # get the names of the sensors from the DataFrame
        sensor_names = activity_data.columns.values[1:]

        # get data as numpy array
        activity_data = activity_data.values

    return activity_data, sensor_names


def extract_tsfel_features(windowed_data: np.array, sensor_names: List[str], features_dict: Dict[Any, Any],
                           fs: int = 100) -> pd.DataFrame:
    """
    Extracts features from the data windows contained in windowed_data using TSFEL. The extracted features are defined
    in 'cfg_file.json'.
    :param windowed_data: The windowed sensor data
    :param sensor_names: the name of the sensors contained in windowed_data
    :param features_dict: the features loaded from cfg_file.json
    :param fs: the sampling rate (in Hz). Default: 100
    :return: pandas.DataFrame containing the extracted features
    """

    print("--> TSFEL-based features")

    # transform the windowed data array into a list of pandas.DataFrames. This data structure works better for TSFEL
    windowed_dfs = []

    # cycle over the windows
    for window in range(windowed_data.shape[0]):

        # create pandas.DataFrame containing the data
        df_window = pd.DataFrame(windowed_data[window, :, :], columns=sensor_names)

        windowed_dfs.append(df_window)

    tsfel_features = tsfel.time_series_features_extractor(features_dict, windowed_dfs, fs=fs)

    return tsfel_features


def extract_quaternion_features(quat_windowed_data) -> pd.DataFrame:
    """
    Extracts quaternion-based features from the rotation vector sensor.
    :param quat_windowed_data: the windowed quaternion data
    :return: pandas.DataFrame containing the extracted features
    """

    print("--> quaternion-based features")

    # init array for holding the features
    quat_features = np.zeros((quat_windowed_data.shape[0], 3))

    # cycle over the windows extracting only the quaternion data
    for i, quat_window in tqdm(enumerate(quat_windowed_data), total=quat_windowed_data.shape[0],
                               ncols=50, bar_format="{l_bar}{bar}| {percentage:3.0f}% {elapsed}"):

        # calculate quaternion features
        quat_features[i] = geodesic_distance(quat_window, scalar_first=False)

    # create pandas.DataFrame
    return pd.DataFrame(quat_features, columns=["quat_mean_dist", "quat_std_dist", "quat_total_dist"])


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _generate_outfolder(features_data_path: str, window_scaler: str, window_size_samples: float) -> str:
    """
    Generates the folders for storing the data
    :param features_data_path: data path to where the data should be stored
    :param window_scaler: the window scaler
    :param window_size_samples: the window size in samples
    :return: the output folder name
    """

    if window_scaler is None:
        folder_name = f'w_{window_size_samples}_sc_none'
    else:
        folder_name = f'w_{window_size_samples}_sc_{window_scaler}'

    output_path = create_dir(features_data_path, os.path.join('extracted_features', folder_name))

    return output_path


def _load_sensor_names(data_file_path: str) -> List[str]:
    """
    Loads the sensor names from a json file containing it.
    :param data_file_path: The path to the data file for which the sensor names should be loaded
    :return: list containing the sensor names as string
    """

    # load the json file (it is assumed that the file lies two levels up
    # given how the raw_data_processor was written
    parent_dir = os.path.dirname(os.path.dirname(data_file_path))

    # generate path to json file
    json_file_path = os.path.join(parent_dir, SENSOR_COLS_JSON)

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Error: The json file containing the loaded sensors was not found."
                                f"\nPlease make sure that it is in: {parent_dir}.")

    # load the file
    json_header = load_json_file(json_file_path)

    # retrieve the sensor names
    sensor_names = json_header[LOADED_SENSORS_KEY]

    return sensor_names


def _pre_process_sensors(data_array: np.array, sensor_names: List[str], fs=100) -> np.array:
    """
    Pre-processes the sensors contained in data_array according to their sensor type.
    :param data_array: the loaded data
    :param sensor_names: the names of the sensors contained in the data array
    :return:
    """

    # make a copy to not override the original data
    processed_data = data_array.copy()

    # process each sensor
    for valid_sensor in VALID_SENSORS:

        # get the positions of the sensor in the sensor_names
        sensor_cols = [col for col, sensor_name in enumerate(sensor_names) if valid_sensor in sensor_name]

        if sensor_cols:

            print(f"--> pre-processing {valid_sensor} sensor")
            # acc pre-processing
            if valid_sensor == ACC:

                processed_data[:, sensor_cols] = pre_process_inertial_data(processed_data[:, sensor_cols], is_acc=True,
                                                                           fs=fs)

            # gyr and mag pre-processing
            elif valid_sensor in [GYR, MAG]:

                processed_data[:, sensor_cols] = pre_process_inertial_data(processed_data[:, sensor_cols], is_acc=False,
                                                                           fs=fs)

            # rotation vector pre-processing
            else:

                processed_data[:, sensor_cols] = slerp_smoothing(processed_data[:, sensor_cols], 0.3,
                                                                 scalar_first=False,
                                                                 return_numpy=True, return_scalar_first=False)
        else:

            print(f"The {valid_sensor} sensor is not in the loaded data. Skipping the pre-processing of this sensor.")

    return processed_data


def _extract_features(windowed_data: np.array, sensor_names: List[str],
                      features_dict: Dict[Any, Any], fs: int = 100) -> pd.DataFrame:
    """
    Extracts features from the windowed data.
    (1) TSFEL: defined in cfg_file.json
    (2) Quaternion-based: mean, std, and total geodesic distance.
    :param windowed_data: the windowed data
    :param sensor_names: the name of the sensors contained in windowed data
    :param features_dict: the feature dictionary loaded from cfg_file.json
    :param fs: the sampling frequency (in Hz). Default: 100
    :return: pandas.DataFrame containing the extracted features
    """
    # extract features using TSFEL for all the sensors
    features_df = extract_tsfel_features(windowed_data, sensor_names, features_dict, fs=fs)

    # check if there are quaternions in the data
    if any(ROT in sensor for sensor in sensor_names):

        # get the columns that contain the quaternion data
        quat_cols = [col for col, sensor_name in enumerate(sensor_names) if ROT in sensor_name]

        # extract features from only the quaternions
        quat_features = extract_quaternion_features(windowed_data[:, :, quat_cols])

        # concatenate features to one DataFrame
        features_df = pd.concat([features_df, quat_features], axis=1)

    return features_df


def _generate_labels(file_name: str, num_windows: int) -> pd.DataFrame():
    """
    Gets the labels for the main and sub-activity corresponding to the file name. The file name encodes the main and
    sub-activity.
    :param file_name: the name of the file
    :param num_windows: the number of windows into which the data was windowed
    :return: pandas.DataFrame containing the labels.
    """

    print("--> getting labels from file name")
    # split off the file extension
    main_sub_activity = os.path.splitext(file_name)[0]

    # get corresponding main and subclasses
    main_class, sub_class = get_labels(main_sub_activity)

    # generate the DataFrame
    label_data = np.tile(np.array([main_class, sub_class]), (num_windows, 1))

    return pd.DataFrame(label_data, columns=[MAIN_LABEL_KEY, SUB_LABEL_KEY])


def _save_subject_features(subject_feature_df: pd.DataFrame(), subject_num: str, output_path: str,
                           file_type: str = '.npy') -> None:
    """
    Saves the features extracted for a subject.
    :param subject_feature_df: pandas.DataFrame containing the extracted features
    :param file_type: the file type as which the features should be stored. Either: '.npy' or '.csv'. Default: '.npy'
    :return: None
    """
    file_name = f'{subject_num}{file_type}'
    file_path = os.path.join(output_path, file_name)

    # save the file
    if file_type == CSV:

        # as csv file
        subject_feature_df.to_csv(file_path, sep=';', index=False)
    else:

        # as npy file
        np.save(file_path, subject_feature_df.values)










