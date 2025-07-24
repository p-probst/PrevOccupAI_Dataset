"""
Functions for generating dataset for deep learning models. The data consists of windowed data. Thus, for each subject
N-number of windows are going to be generated for each activity. The N does vary between activities as each has a
somewhat different recording length. Thus, an imbalanced dataset is generated.
(Class balancing is performed when loading the Dataset before passing it to the DL model)

Available Functions
-------------------
[Public]
generate_dataset(...): Generates dataset for Deep Learning

------------------
[Private]
_generate_outfolder(...): Generates the folder for storing the dataset
_save_windowed_data(...): Saves the windowed data into individual .npy files using the window number in the naming of the file.
_calc_subject_stats(...): Calculates a set of statistics from the data contained in subject_data.
_calc_global_stats(...): Calculates global statistics (over all subjects).

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List, Dict, Union, Optional, Literal, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# internal imports
from constants import NPY, VALID_ACTIVITIES, VALID_SENSORS, ACC, GYR, MAG, MAIN_ACTIVITY_LABELS, \
    SUB_ACTIVITIES_STAND_LABELS, SUB_ACTIVITIES_WALK_LABELS, CLASS_INSTANCES_JSON, SUBJECT_STATS_JSON
from feature_extractor import get_sliding_windows_indices, window_data
from feature_extractor.feature_extractor import load_data
from file_utils import create_dir, remove_file_duplicates, save_json_file, get_labels, validate_activity_input, \
    load_json_file
from raw_data_processor import pre_process_inertial_data, slerp_smoothing

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #

# dictionary keys
SENSOR_MEAN = "sensor_mean"
SENSOR_VAR = "sensor_var"
SENSOR_MIN = "sensor_min"
SENSOR_MAX = "sensor_max"
N_SAMPLES = "n_samples"
GLOBAL_STATS = "G000"

# norm methods
Z_SCORE = "z-score"
MIN_MAX = "min-max"
NORM_METHODS = [Z_SCORE, MIN_MAX]

# norm types
WINDOW_NORM = "window"
SUBJECT_NORM = "subject"
GLOBAL_NORM = "global"
NORM_TYPES = [WINDOW_NORM, SUBJECT_NORM, GLOBAL_NORM]

# self-defined typing hints
NormalizationType = Optional[Literal[WINDOW_NORM, SUBJECT_NORM, GLOBAL_NORM]]
NormMethod = Optional[Literal[Z_SCORE, MIN_MAX]]

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
class HARDataset(Dataset):
    """
    Pytorch Dataset for the deep learning HAR dataset. In this dataset each window is stored individually as a .npy file.
    The file name is encoded in the following way:
    <SUBJECT_ID>_<MAIN_ACTIVITY_LABEL>_<SUB_ACTIVITY_LABEL>_<WINDOW_NUM>.npy
    e.g., P001_walk_slow_021.npy

    The HARDatasets handles label retrieval and normalization for each window according to the provided normalization type.

    :param data_path: the path to the dataset
    :param subject_ids: the IDs of the subject for which the data should be loaded
    :param norm_method: the normalization method. The following are available:
                        "z-score": uses z-score normalization
                        "min-max": uses min-max normalization
    :param norm_type: the normalization type. The following are available
                      "window": each data sample is normalized using its own statistics
                      "subject": each data sample is normalized using the corresponding subject's statistics
                      "global": each data sample is normalized using global statistics calculated over the entire dataset
                      None (default): no normalization applied
    """

    def __init__(self, data_path: str, subject_ids: List[str],
                 norm_method: NormMethod = None, norm_type: NormalizationType = None):

        # input validation
        # (1) check path
        if not os.path.isdir(data_path):
            raise ValueError("The data path you provided is not valid."
                             f"\nProvided data path: {data_path}")

        # (2) check subject_ids
        # collect all available subject IDs in the dataset folder
        available_subjects = {file_name.split("_")[0] for file_name in os.listdir(data_path) if
                              file_name.endswith(".npy")}

        # check if there are any subjects chosen by the user that are not in the dataset
        unknown_subjects = set(subject_ids) - available_subjects

        if len(unknown_subjects) == len(subject_ids):
            raise ValueError("The provided subject IDs were not found in the dataset."
                             f"\nPlease choose from the following subject IDs are in the dataset: {sorted(available_subjects)}."
                             f"\nProvided subject_ids: {subject_ids}")
        else:
            print(f"[WARNING]: The following subjects were not found in the dataset: {list(unknown_subjects)}"
                  "\nThese subjects are going to be ignored for model training/testing")

        # (3) check norm_method
        # (a) norm_method provided but no norm_type
        if norm_method and not norm_type:
            raise ValueError(f"norm_method was provided ({norm_method}), but no norm_type was given. "
                             f"Please provide one of: {NORM_TYPES}.")

        # (b) norm_type provided but no norm_method
        if norm_type and not norm_method:
            print(f"norm_type was provided ({norm_type}), but no norm_method was specified. "
                  "Disabling normalization by setting norm_type=None.")

            # set normalization to none
            norm_type = None
            norm_method = None

        # (c) check norm_method validity
        if norm_method and norm_method not in NORM_METHODS:
            raise ValueError(f"Invalid norm_method: {norm_method}. Must be one of: {NORM_METHODS}.")

        # (c) check norm_type validity
        if norm_type and norm_type not in NORM_TYPES:
            raise ValueError(f"Invalid norm_type: {norm_type}. Must be one of: {NORM_TYPES}.")

        # init class variables
        self.data_path = data_path
        self.subject_ids = subject_ids
        self.norm_method = norm_method
        self.norm_type = norm_type
        self.stats = {}

        # get files corresponding to the set subject_ids
        self.files = [file_name for file_name in os.listdir(data_path)
                      if file_name.endswith(".npy") and file_name.split('_')[0] in subject_ids]

        # load the statistics in case a norm_type was chosen
        if norm_type in [SUBJECT_NORM, GLOBAL_NORM]:

            # load the json-file containing the statistics
            self.stats = load_json_file(os.path.join(data_path, SUBJECT_STATS_JSON))

    def __len__(self):

        return len(self.files)

    def _min_max_norm(self, data_sample: np.array, subject_id: str):
        """
        Applies min-max normalization using the provided statistics
        :param data_sample: numpy.array of shape [window_size, num_channels]
        :param subject_id: ID of the subject to which the data sample belongs to
        :return: min-max normalized data array of the same shape as the input.
        """

        # apply window-wise normalization
        if self.norm_type == WINDOW_NORM:

            # calculate window statistics
            window_mins = np.min(data_sample, axis=0, keepdims=True)
            window_maxs = np.min(data_sample, axis=0, keepdims=True)

            return (data_sample - window_mins) / (window_maxs - window_mins)

        # apply subject-wise normalization
        elif self.norm_type == SUBJECT_NORM:

            # get the statistics of the corresponding subject
            subject_stats = self.stats.get(subject_id)
            subject_mins = np.array(subject_stats[SENSOR_MIN])
            subject_maxs = np.array(subject_stats[SENSOR_MAX])

            return (data_sample - subject_mins) / (subject_maxs - subject_mins)

        # apply population normalization
        elif self.norm_type == GLOBAL_NORM:

            # get the global/population statistics
            global_stats = self.stats.get(GLOBAL_STATS)
            global_mins = np.array(global_stats[SENSOR_MIN])
            global_maxs = np.array(global_stats[SENSOR_MAX])

            return (data_sample - global_mins) / (global_maxs - global_mins)


    def _zero_score_norm(self, data_sample: np.array, subject_id: str):
        """
        Applies zero-score normalization using the provided statistics
        :param data_sample: numpy.array of shape [window_size, num_channels]
        :param subject_id: ID of the subject to which the data sample belongs to
        :return: zero-score normalized data array of the same shape as the input.
        """

        # apply window-wise normalization
        if self.norm_type == WINDOW_NORM:

            # calculate window statistics
            window_means = np.mean(data_sample, axis=0, keepdims=True)
            window_stds = np.std(data_sample, axis=0, keepdims=True)

            return (data_sample - window_means) / window_stds

        # apply subject-wise normalization
        elif self.norm_type == SUBJECT_NORM:

            # get the statistics of the corresponding subject
            subject_stats = self.stats.get(subject_id)
            subject_means = np.array(subject_stats[SENSOR_MEAN])
            subject_stds = np.array(subject_stats[SENSOR_VAR])

            return (data_sample - subject_means) / subject_stds

        # apply population normalization
        elif self.norm_type == GLOBAL_NORM:

            # get the global/population statistical
            global_stats = self.stats.get(GLOBAL_STATS)
            global_means = np.array(global_stats[SENSOR_MEAN])
            global_stds = np.array(global_stats[SENSOR_VAR])

            return (data_sample - global_means) / global_stds

    def _normalize_data(self, data_sample: np.array, subject_id: str) -> np.array:
        """
        Applies normalization to data sample based on the selected normalization strategy
        :param data_sample: numpy.array of shape [window_size, num_channels]
        :param subject_id: ID of the subject to which the data sample belongs to
        :return: normalized data array of the same shape as the input.
        """

        if self.norm_type is None:
            return data_sample

        else:

            # apply z-score normalization
            if self.norm_method == Z_SCORE:

                return self._zero_score_norm(data_sample, subject_id)

            # apply min-max normalization
            elif self.norm_method == MIN_MAX:

                return self._min_max_norm(data_sample, subject_id)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        loads a data sample and its corresponding main and sub-activity labels.
        :param idx: index of the file to load
        :return: Tuple containing:
                 - torch.FloatTensor of shape [window_size, num_channels]
                 - torch.LongTensor containing the main activity label (as integer)
                 - torch.LongTensor containing the sub-activiyt label (as integer)
        """

        # get the file name
        file_name = self.files[idx]

        # retrieve the subject_id from the file name
        subject_id = file_name.split("_")[0]

        # extract the main and sub-activity label
        main_sub_activity = "_".join(file_name.split("_")[1:3])
        main_class, sub_class = get_labels(main_sub_activity)

        # load the data
        data_sample = np.load(os.path.join(self.data_path, file_name))

        # apply normalization
        data_sample = self._normalize_data(data_sample, subject_id)

        return torch.tensor(data_sample, dtype=torch.float32), torch.tensor(main_class, dtype=torch.long), torch.tensor(sub_class, dtype=torch.long)


def generate_dataset(data_path: str, output_path: str, activities: List[str] = None, fs: int = 100, window_size: float = 1.5,
                     overlap: float = 0.5, default_input_file_type: str = NPY) -> None:
    """
    Generates the dataset for deep learning model training based on the segmented data stored in data_path.
    The generated dataset consists of windowed data. For each subject N windows are extracted for each activity segment.
    The generated dataset is stored as '.npy' files. To distinguish the generated files the naming convention for the
    files is:
    <SUBJECT>_<ACTIVITY LABEL>_<N>, where
    SUBJECT: The subject number (e.g., P001),
    ACTIVITY LABEL: the activity label (e.g., 1)
    N: The window number (e.g., 201)

    :param data_path: the path to the data. This should point to the folder containing the segmented tasks.
    :param output_path: the path to where the dataset should be stored. Within this path a folder called
                        'HAR_deep_learning_data' is generated.
    :param activities: list containing the activities that should be considered for dataset generation. Default: None
                       (in this case all activities are loaded).
    :param fs: the sampling rate (in Hz) of the data. Default: 100
    :param window_size: the window size in seconds that should be used for windowing the data. Default: 1.5
    :param overlap: the overlap between consecutive windows. The value has to be between [0, 1]. Default: 0.5
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

    # list all subject folders
    subject_folders = os.listdir(data_path)

    # get the folders that contain the subject data. Subject data folders start with 'P' (e.g., P001)
    subject_folders = [folder for folder in subject_folders if folder.startswith('P')]

    # generate output path (folder) where all the data is stored
    output_path = _generate_outfolder(output_path, int(window_size * fs))

    # dictionary for holding the statistics for each subject
    subject_stats = {}

    # dictionary for holding the class instances for each subject
    subject_class_instances = {}

    # cycle over the subjects
    for subject in subject_folders:
        print("\n#----------------------------------------------------------------------#")
        print(f"# Extracting features for Subject {subject}")

        # get the path to the subject folder
        subject_folder_path = os.path.join(data_path, subject)

        # list all files in the path
        files = os.listdir(subject_folder_path)

        # remove file duplicates
        # (e.g., 'walk_slow.npy' and walk_slow.csv'  --> keep only the file that has the default input type)
        files = remove_file_duplicates(files, default_input_file_type=default_input_file_type)

        # remove all activities that were not chosen
        files = [file for file in files if any(activity in file for activity in activities)]

        # inform user
        print(f"The following files are going to be read: {files}")

        # extract the main-sub activity string from the file name
        # (each file name encodes the main and sub-activity in the first to strings that are separated by '_',
        # {main_activity}_{sub_activity}...
        main_sub_activities = ['_'.join(os.path.splitext(file)[0].split('_')[:2]) for file in files]

        # remove duplicate main-sub activities
        main_sub_activities = list(dict.fromkeys(main_sub_activities))

        # init dict for holding class instances of the current subject
        class_instances = dict.fromkeys(MAIN_ACTIVITY_LABELS + SUB_ACTIVITIES_STAND_LABELS + SUB_ACTIVITIES_WALK_LABELS, 0)

        # list to hold the subject data for calculation of subject-wise statistics
        subject_data = []

        # cycle over the sub-activities
        for num, main_sub_activity in enumerate(main_sub_activities, start=1):

            print(f"\n#({num}) {main_sub_activity}:")

            # get the files corresponding to the sub-activity
            sub_files = [file for file in files if main_sub_activity in file]

            # get the number of files
            num_files = len(sub_files)

            # initialize window tracker
            num_extracted_windows = 0

            # get the main and sub_activity
            main_class, sub_class = get_labels(main_sub_activity)

            # cycle over the files
            for file_num, file in enumerate(sub_files, start=1):

                # (1) load the data
                # TODO add to file_utils.py or pipeline file
                print(f"({file_num}.1) loading file {file_num}/{num_files}: {file}")
                data, sensor_names = load_data(os.path.join(subject_folder_path, file))

                # remove time column
                data = data[:, 1:]

                # (2) pre-process the data
                # TODO: create some agnostic pipeline file that does the entire pre-processing
                print(f"({file_num}.2) pre-processing")
                data = _pre_process_sensors(data, sensor_names)

                # remove impulse response
                data = data[250:, :]

                # (3) add pre-processed data to list for calculating subject statistics later on
                subject_data.append(data)

                # (4) window the data
                # (since all are of the same length it is possible to use just one sensor channel)
                # TODO: until here (including) should be in the pipeline.py
                print(f"({file_num}.3) windowing data")
                indices = get_sliding_windows_indices(data[:, 0], fs=fs, window_size=window_size, overlap=overlap)
                windowed_data = window_data(data, indices)

                # get the number of windos
                num_windows = windowed_data.shape[0]
                print(f"num_windows: {num_windows}")

                # estimate '00' padding for file name (if-else: file_num > 1: estimate | else: just get the true length of the windows)
                if num_files > 1:

                    # only do the estimate once
                    if file_num == 1:

                        # estimate zero padding
                        num_windows_estimate = num_windows * num_files

                        print(f"estimated number of windows: {num_windows_estimate}")
                        zeros_pad = len(str(num_windows_estimate  - 1))

                else:

                    zeros_pad = len(str(num_windows -1))

                # store the windows
                _save_windowed_data(windowed_data, output_path, subject, main_sub_activity, zeros_pad, num_extracted_windows)

                # update the window tracker
                num_extracted_windows += num_windows

            # update the class instances
            class_instances[main_class] += num_extracted_windows
            class_instances[sub_class] = num_extracted_windows
            subject_class_instances[subject] = class_instances

            # inform user on how many windows were extracted
            print(f"--> extracted windows {num_extracted_windows}")

        # calculate subject sensor statistics and add them to the dict for holding all subject statistics
        subject_stats.update(_calc_subject_stats(subject_data, subject_id=subject))

    # calculate global statistics
    _calc_global_stats(subject_stats)

    # save the dictionaries into json files
    save_json_file(subject_class_instances, CLASS_INSTANCES_JSON, output_path)
    save_json_file(subject_stats, SUBJECT_STATS_JSON, output_path)

    print("finished DL dataset generation")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _generate_outfolder(features_data_path: str,  window_size_samples: float) -> str:
    """
    Generates the folder for storing the generated dataset
    :param features_data_path: data path to where the data should be stored.
    :param window_size_samples: the window size in samples
    :return: the output folder path
    """

    # generate folder name
    folder_name = f"w_{window_size_samples}"

    output_path = create_dir(features_data_path, os.path.join('DL_dataset', folder_name))

    return output_path

# TODO: define in pipeline.py (or similar) and import to this file
def _pre_process_sensors(data_array: np.array, sensor_names: List[str], fs=100) -> np.array:
    """
    Pre-processes the sensors contained in data_array according to their sensor type.
    :param data_array: the loaded data
    :param sensor_names: the names of the sensors contained in the data array
    :return: the processed sensor data
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


def _save_windowed_data(windowed_data: np.array, output_path: str, subject: str, label: str, zeros_pad, num_extracted_windows) -> None:
    """
    Saves the windowed data into individual .npy files using the window number in the naming of the file.
    Files use the naming convention <SUBJECT>_<MAIN_ACTIVITY>_<SUB_ACTIVITY>_<WINDOW_NUM>
    :param windowed_data: numpy.array of size [num_windows, window_size_samples, num_channels] containing the windowed data
    :param output_path: the path to where the windowed data should be stored
    :param subject: The subject ID (e.g., "P001")
    :param label: The activity label consisting of <MAIN_ACTIVITY>_<SUB_ACTIVITY>
    :param zeros_pad: the number of zeros needed for padding in the filename (e.g., _001, _0020)
    :param num_extracted_windows: the number of extracted windows
    :return: none
    """

    # cycle over the windows
    for num, window in enumerate(windowed_data, start=1):

        # create file name
        filename = f"{subject}_{label}_{num + num_extracted_windows:0{zeros_pad}d}.npy"

        # save the data
        np.save(os.path.join(output_path, filename), window)


def _calc_subject_stats(subject_data: List[np.array], subject_id: str) -> Dict[str, Union[int, List[int]]]:
    """
    Calculates a set of statistics from the data contained in subject_data. subject_data is assumed to be a 2D array
    containing the data of a sensor-axis (e.g., x_ACC, y_GYR, etc.) in each column. The statistics are calculated for
    each sensor-axis. The following statistics are calculated
    (1) mean | shape: [num_sensors, ]
    (2) variance | shape: [num_sensors, ]
    (3) minimum | shape: [num_sensors, ]
    (4) maximum | shape: [num_sensors, ]
    (5) number of samples (calculated over the entire data) | shape: 1 (int)
    :param subject_data: List of numpy.arrays containing all the data of one activity
    :param subject_id: ID of the subject (e.g., "P001")
    :return: nested dictionary where the main key is the subject_id and the sub-dictionary contains the extracted measures
    """

    # concatenate all data
    subject_data = np.vstack(subject_data)

    # mean and std for each sensor
    subject_means = np.mean(subject_data, axis=0).tolist()
    subject_vars = np.var(subject_data, axis=0).tolist()

    # min-max for each sensor
    subject_mins = np.min(subject_data, axis=0).tolist()
    subject_maxs = np.max(subject_data, axis=0).tolist()

    # total number of samples (over all data)
    print(np.shape(subject_data))
    sub_samples = subject_data.shape[0]

    return {subject_id: {N_SAMPLES: sub_samples, SENSOR_MEAN: subject_means,
                         SENSOR_VAR: subject_vars, SENSOR_MIN: subject_mins, SENSOR_MAX: subject_maxs}}


def _calc_global_stats(subject_stats_dict: Dict[str, Dict[str, Union[int, List[int]]]]) -> None:
    """
    calculates global statistics (over all subjects). The following global stats are calculated
    (1) global mean
    (2) global variance: the global variance is based on the combined/pooled variance formula. See: https://en.wikipedia.org/wiki/Pooled_variance
    (3) global min
    (4) global max
    After calculation, the global stats are added to the subject_stats_dict using the "G000".
    :param subject_stats_dict: nested dictionary containing subject-wise sensor statistics
    :return: None
    """

    # collect the sensor stats into arrays of size [n_subjects, n_sensor_columns]
    sensor_means = np.array([subject_dict[SENSOR_MEAN] for subject_dict in subject_stats_dict.values()])
    sensor_vars = np.array([subject_dict[SENSOR_VAR] for subject_dict in subject_stats_dict.values()])
    sensor_min = np.array([subject_dict[SENSOR_MIN] for subject_dict in subject_stats_dict.values()])
    sensor_max = np.array([subject_dict[SENSOR_MAX] for subject_dict in subject_stats_dict.values()])
    n_samples = np.array([subject_dict[N_SAMPLES] for subject_dict in subject_stats_dict.values()]).reshape(-1, 1)

    # calculate global N and mean
    global_n = np.sum(n_samples, axis=0).item()
    global_mean = np.sum((n_samples * sensor_means), axis=0) / global_n

    # calculate sum[(n_i - 1) * var_i]
    within_group_ss = np.sum((n_samples - 1) * sensor_vars, axis=0)

    # calculate sum[n_i * (mu_i - mu_g)]
    between_group_ss = np.sum(n_samples * np.square((sensor_means - global_mean)), axis=0)

    # calculate combined variance
    global_var = 1 / (global_n - 1) * (within_group_ss + between_group_ss)

    # calculate global min and max
    global_min = np.min(sensor_min, axis=0)
    global_max = np.max(sensor_max, axis=0)

    # add the calculated statistics to the subject stats dict
    subject_stats_dict[GLOBAL_STATS] = {N_SAMPLES: global_n, SENSOR_MEAN: global_mean.tolist(), SENSOR_VAR: global_var.tolist(),
                                  SENSOR_MIN: global_min.tolist(), SENSOR_MAX: global_max.tolist()}



