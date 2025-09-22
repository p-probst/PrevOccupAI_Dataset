"""
Functions for loading data for Human Activity Recognition.

Available Functions
-------------------
[Public]
load_features(...): loads the extracted features from all the subjects.
load_labels_from_log(...): loads the labels from a txt file and generates a label vector
load_production_model(...): Loads the pre-trained model
------------------
[Private]
_get_feature_names_and_instances(...): Extracts feature names and determines the number of instances per sub-class based on the selected balancing strategy.
_load_file(...): loads the feature data based on its file type.
_balance_subject_data(...): Balances the subject's feature data by selecting the needed number of instances from each subclass.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestClassifier

# internal imports
from constants import CLASS_INSTANCES_JSON, FEATURE_COLS_KEY, NPY, SUB_LABEL_KEY, MAIN_LABEL_KEY
from file_utils import remove_file_duplicates, load_json_file
from ..data_balancer import balance_main_class, balance_sub_class, balance_subject_data, MAIN_CLASS_BALANCING, SUB_CLASS_BALANCING


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def load_features(feature_data_path: str, balance_data: str = None, default_input_file_type: str = NPY,
                  random_state: int = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Loads the extracted features from all the subjects. The function allows for balancing the data so that there is
    the same amount of instances for all subjects, thus avoiding data bias towards a specific class or subject.
    :param feature_data_path: the path to the folder containing the extracted features from all subjects
    :param balance_data: the data balancing type. Can be either:
                         'main_classes': for balancing the data in such a way that each main class has the (almost) the
                                       same amount of data. This ensures that each sub-class within the main class has
                                       the same amount of instances.
                         'sub_classes': for balancing that all sub-classes have the same amount of instances
                         None: no balancing applied. Default: None
    :param default_input_file_type: The default input type that should be used. This is used to make sure that only one
                                    file is loaded in case the feature data has been stored as both '.npy' and '.csv'.
                                    It can be either '.csv' or '.npy'. Default: '.npy'
    :param random_state: Sets a seed to always return the same balancing. Only needed when data balancing is applied.
                         When the random seed is set before using the function using np.random.seed() then setting it
                         in this function is not necessary anymore.
    :return: The features for all subjects as pandas.DataFrame, the main class labels as pandas.Series,
             the sub-class labels as pandas.Series, the subject IDs for each instance as pandas.Series

    """
    # set the random seed
    # this is done as the datasets are balanced by randomly picking (without replacement) the needed amount of instances
    # from each class
    if random_state:
        np.random.seed(random_state)

    # check input for balanced_data
    if balance_data and balance_data not in [MAIN_CLASS_BALANCING, SUB_CLASS_BALANCING]:

        raise ValueError(f"The data balancing type you chose is not valid. Provided type: {balance_data}."
                         f"\nPlease choose from the following {[MAIN_CLASS_BALANCING, SUB_CLASS_BALANCING, None]}.")

    # retrieve the feature names and the instances per sub-class for data balancing (if needed)
    feature_names, instances_per_sub_class = _get_feature_names_and_instances(feature_data_path, balance_data=balance_data)

    # list all files in the folder
    subject_files = os.listdir(feature_data_path)

    # get the files that are subject files. Subject features start with 'P' (e.g., P001)
    subject_files = [folder for folder in subject_files if folder.startswith('P')]

    # remove file duplicates in case the files have been saved as both '.npy' and '.csv'
    subject_files = remove_file_duplicates(subject_files, default_input_file_type=default_input_file_type)

    # get the file type
    file_type = os.path.splitext(subject_files[0])[1]

    # lists for holding the loaded data
    loaded_features = []
    loaded_subject_ids = []

    # variable to hold the number of instances per subject
    instances_per_subject = None

    # cycle over the subject files
    for subject_file in tqdm(subject_files, desc="loading feature data"):

        # create the full file path
        file_path = os.path.join(feature_data_path, subject_file)

        # load the file
        subject_features = _load_file(file_path, file_type)

        # perform balancing if needed
        if instances_per_sub_class:

            # perform balancing
            subject_features = _perform_subject_balancing(subject_features, feature_names,
                                                          instances_sit=instances_per_sub_class[0],
                                                          instances_stand=instances_per_sub_class[1],
                                                          instances_walk=instances_per_sub_class[2])

        instances_per_subject = subject_features.shape[0]
        # generate subject id (needed for groupKFold)
        subject_id = [subject_file.split('.')[0]] * len(subject_features)

        # append the loaded data to their respective lists
        loaded_features.append(subject_features)
        loaded_subject_ids.extend(subject_id)

    # combine all data to one array and transform it to a pandas.DataFrame
    loaded_features = pd.DataFrame(np.vstack(loaded_features), columns=feature_names)
    print(f"Total number of instances per subject: {instances_per_subject}")
    print(f"Total number of instances: {loaded_features.shape[0]} over {len(subject_files)} subjects")

    # get the main and sub-classes
    main_class_labels = loaded_features[MAIN_LABEL_KEY].astype(int)
    sub_class_labels = loaded_features[SUB_LABEL_KEY].astype(int)

    loaded_features = loaded_features.drop(columns=[MAIN_LABEL_KEY, SUB_LABEL_KEY])

    return loaded_features, main_class_labels, sub_class_labels, pd.Series(loaded_subject_ids)


def load_labels_from_log(filepath: str, label_mapping: Dict[str, int], num_samples_recording: int, fs: int = 100) -> List[int]:
    """
    Creates a label vector from a log file containing activity timestamps.

    :param filepath: Path to the log file. Each line should follow the format: "hh:mm:ss.ms; activity".
    :param label_mapping: A dictionary mapping activity strings to numeric labels.
                          e.g., {"sitting": 1, "standing": 2, "walking": 3}.
    :param num_samples_recording: The number of samples in the recording
    :param fs: Sampling rate in Hz (default is 100).
    :returns: A list of integers representing the label vector.
    """
    # Load log file into a DataFrame
    df = pd.read_csv(
        filepath,
        sep=';',
        header=None,
        names=['timestamp', 'activity']
    )

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip(), format='%H:%M:%S.%f')
    df['activity'] = df['activity'].str.strip()

    label_vector = []

    # Iterate through consecutive rows
    for i in range(len(df) - 1):
        start_time = df.iloc[i]['timestamp']
        end_time = df.iloc[i + 1]['timestamp']
        activity = df.iloc[i]['activity']

        # Duration in seconds
        duration = (end_time - start_time).total_seconds()

        # Number of samples based on sampling rate
        num_samples = int(duration) * fs

        # Corresponding numeric label
        label_value = label_mapping.get(activity)

        if label_value is None:
            raise ValueError(f"Unknown activity label: {activity}")

        # Append labels for the duration
        label_vector.extend([label_value] * num_samples)

    # get the last label
    last_label = label_mapping.get(df.iloc[-1]['activity'])

    # calculate difference between length of label vector and data acquisition
    # (since the timetamps only account for the beginning of the activity, it is still needed to extend the labels until the end of the recording)
    num_missing_labels = num_samples_recording - len(label_vector)

    label_vector.extend([last_label] * num_missing_labels)

    return label_vector


def load_production_model(model_path: str) -> Tuple[RandomForestClassifier, List[str]]:
    """
    Loads the production model
    :param model_path: path o the model
    :return: a tuple containing the model and the list of features used
    """
    # load the classifier
    har_model = joblib.load(model_path)

    # print model name
    print(f"model: {type(har_model).__name__}")
    print(f"\nhyperparameters: {har_model.get_params()}")

    # print the classes that the model saw during training
    print(f"\nclasses: {har_model.classes_}")

    # get the features that the model was trained with
    feature_names = har_model.feature_names_in_
    print(f"\nnumber of features: {len(feature_names)}")
    print(f"features: {feature_names}")

    return har_model, feature_names

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _get_feature_names_and_instances(feature_data_path: str, balance_data: str = None) -> Tuple[List[str], Tuple[int, int, int]]:
    """
    Extracts feature names and determines the number of instances per sub-class based on the selected balancing strategy.

    This function loads a JSON file containing feature names and class instance counts,
    then optionally applies a balancing method to ensure fair representation of activity classes.

    :param feature_data_path: Path to the directory containing the JSON file with feature names
                              and class instance counts.
    :param balance_data: Optional balancing strategy. Can be one of the following:
                         - `main_classes`: Balances instances at the main class level.
                         - `sub_classes`: Balances instances at the sub-class level.
                         - None: No balancing is applied.

    :return: A tuple containing:
             - A list of feature names extracted from the JSON file.
             - A tuple (instances_sit, instances_stand, instances_walk) representing
               the number of instances per sub-class based on the chosen balancing strategy.
               Returns `None` if no balancing is applied.
    """

    # create full path to JSON file (to extract feature names and class instances)
    json_file_path = os.path.join(feature_data_path, CLASS_INSTANCES_JSON)

    json_dict = load_json_file(json_file_path)

    # get the feature names (removing them from the dict to facilitate processing of class instances
    feature_names = json_dict.pop(FEATURE_COLS_KEY)

    # create DataFrame with the class instances per subject
    class_instances_df = pd.DataFrame.from_dict(json_dict, orient='index')

    # apply balancing if needed
    if balance_data == MAIN_CLASS_BALANCING:

        instances_per_sub_class = balance_main_class(class_instances_df)

    elif balance_data == SUB_CLASS_BALANCING:

        instances_per_sub_class = balance_sub_class(class_instances_df)

    else:

        # no balancing
        instances_per_sub_class = None

    return feature_names, instances_per_sub_class


def _load_file(file_path: str, file_type: str) -> np.array:
    """
    loads the feature data based on its file type.
    :param file_path: the full path to the file
    :return: numpy.array containing the loaded features
    """

    # load the file
    if file_type == NPY:

        subject_features = np.load(file_path)

    else:  # file typle .csv

        # load the csv as numpy array to facilitate the code below (everything is handled as numpy.array)
        subject_features = pd.read_csv(file_path, sep=';').values

    return subject_features


def _perform_subject_balancing(subject_features: np.array, feature_names: List[str], instances_sit: int,
                          instances_stand: int, instances_walk: int) -> np.array:
    """
    Balances the subject's feature data by selecting the needed number of instances from each subclass.

    :param subject_features: The feature data for a subject, either as a numpy.array or pandas.DataFrame.
    :param feature_names: List of feature names, used to locate the subclass label column.
    :param instances_sit: The number of instances to retain for sitting subclasses.
    :param instances_stand: The number of instances to retain for standing subclasses.
    :param instances_walk: The number of instances to retain for walking subclasses.
    :return: A  numpy.array with balanced subclass instances.
    """

    # get the subclass labels
    sub_class_col_idx = feature_names.index(SUB_LABEL_KEY)
    sub_class_labels = subject_features[:, sub_class_col_idx].astype(int, copy=False)

    # get the indices for balancing
    indices_for_balancing = balance_subject_data(sub_class_labels, instances_sit, instances_stand, instances_walk)

    # retrieve the balanced features
    return subject_features[indices_for_balancing, :]
