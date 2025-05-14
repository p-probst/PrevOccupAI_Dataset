"""
Functions for loading data for Human Activity Recognition.

Available Functions
-------------------
[Public]
load_features(...): loads the extracted features from all the subjects.
------------------
[Private]
_get_feature_names_and_instances(...): Extracts feature names and determines the number of instances per sub-class based on the selected balancing strategy.
_balance_main_class(...): Determines the number of instances to sample from each sub-class so that the aggregated main class distributions (Sit, Stand, Walk) remain balanced.
_balance_sub_class(...): Determines the number of instances to sample from each sub-class so that all sub-classes have an equal number of instances.
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
from typing import Tuple, List
from tqdm import tqdm

# internal imports
from constants import CLASS_INSTANCES_JSON, FEATURE_COLS_KEY, SUB_ACTIVITIES_WALK_LABELS, SUB_ACTIVITIES_STAND_LABELS, \
    NPY, SUB_LABEL_KEY, MAIN_LABEL_KEY, RANDOM_SEED
from file_utils import remove_file_duplicates, load_json_file

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
MAIN_CLASS_BALANCING = 'main_classes'
SUB_CLASS_BALANCING = 'sub_classes'


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

    # TODO: @Sara you can use instances_per_sub_class to calculate the the number of instances.
    #  Calculate the instances_per_subject and total_num_instances. You can then just print the results.
    #  Please also print the instances for each sub_class and format the print statement nicely.
    #  The calculation should be dynamic meaning, that you should extract the number of subjects dynamically from the code below
    #  (you should be able to get it from subject_files after the duplicates have been removed)
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

    # cycle over the subject files
    for subject_file in tqdm(subject_files, desc="loading feature data"):

        # create the full file path
        file_path = os.path.join(feature_data_path, subject_file)

        # load the file
        subject_features = _load_file(file_path, file_type)

        # perform balancing if needed
        if instances_per_sub_class:

            # perform balancing
            subject_features = _balance_subject_data(subject_features, feature_names,
                                                     instances_sit=instances_per_sub_class[0],
                                                     instances_stand=instances_per_sub_class[1],
                                                     instances_walk=instances_per_sub_class[2])

        # generate subject id (needed for groupKFold)
        subject_id = [subject_file.split('.')[0]] * len(subject_features)

        # append the loaded data to their respective lists
        loaded_features.append(subject_features)
        loaded_subject_ids.extend(subject_id)

    # combine all data to one array and transform it to a pandas.DataFrame
    loaded_features = pd.DataFrame(np.vstack(loaded_features), columns=feature_names)

    # get the main and sub-classes
    main_class_labels = loaded_features[MAIN_LABEL_KEY].astype(int)
    sub_class_labels = loaded_features[SUB_LABEL_KEY].astype(int)

    loaded_features = loaded_features.drop(columns=[MAIN_LABEL_KEY, SUB_LABEL_KEY])

    return loaded_features, main_class_labels, sub_class_labels, pd.Series(loaded_subject_ids)


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

        instances_per_sub_class = _balance_main_class(class_instances_df)

    elif balance_data == SUB_CLASS_BALANCING:

        instances_per_sub_class = _balance_sub_class(class_instances_df)

    else:

        # no balancing
        instances_per_sub_class = None

    return feature_names, instances_per_sub_class


def _balance_main_class(class_instances_df: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Determines the number of instances to sample from each sub-class so that the aggregated main class
    distributions (Sit, Stand, Walk) remain balanced.

    Mapping of main classes to their sub-classes:
        - Sit: sit
        - Stand: stand_still, stand_talk, stand_coffee, stand_folders
        - Walk: walk_medium, walk_fast, walk_stairs_up, walk_stairs_down

    The balancing ensures that the total number of instances per main class is equal by adjusting
    the number of instances drawn from each sub-class.

    Note: this function could produce a bug when being used on another dataset. The code works because:
          min_instances_walk < min_instances_stand is always the case for all subjects as the stairs recordings were
          the shortest.

    :param class_instances_df: pandas.DataFrame where:
                                 - Rows represent subjects.
                                 - Columns represent the number of instances per sub-class.
                                 - The first three columns contain the total amount of instances for the main classes.
    :return: A tuple of integers (instances_sit, instances_stand, instances_walk), representing the number of instances
             to sample from each sub-class so that Sit, Stand, and Walk remain balanced.
    """

    # calculate the minimum per class over all subjects
    min_instances = class_instances_df.min(axis=0)
    min_instances.index = min_instances.index.astype(int)

    # get the number of instances belonging to the walk and stand subclasses
    min_instances_walk = min_instances.loc[SUB_ACTIVITIES_WALK_LABELS].min()
    min_instances_stand = min_instances.loc[SUB_ACTIVITIES_STAND_LABELS].min()

    # calculate how many main instances that would give
    total_walk = min_instances_walk * len(SUB_ACTIVITIES_WALK_LABELS)
    total_stand = min_instances_stand * len(SUB_ACTIVITIES_STAND_LABELS)

    # find the class with the least amount of total instances
    instances_sit = min(total_stand, total_walk)
    instances_walk = instances_sit // len(SUB_ACTIVITIES_WALK_LABELS)
    instances_stand = instances_sit // len(SUB_ACTIVITIES_STAND_LABELS)

    return instances_sit, instances_stand, instances_walk


def _balance_sub_class(class_instances_df: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Determines the number of instances to sample from each sub-class so that all sub-classes
    (from Sit, Stand, Walk) have an equal number of instances.

    Mapping of main classes to their sub-classes:
        - Sit: sit
        - Stand: stand_still, stand_talk, stand_coffee, stand_folders
        - Walk: walk_medium, walk_fast, walk_stairs_up, walk_stairs_down

    This ensures that no specific sub-class is overrepresented, reducing bias in the model.

    :param class_instances_df: pandas.DataFrame where:
                                 - Rows represent subjects.
                                 - Columns represent the number of instances per sub-class
                                 - The first three columns contain the total amount of instances for the main classes.

    :return: A tuple of integers (instances_sit, instances_stand, instances_walk),
             representing the number of instances to sample from each sub-class
             so that all sub-classes have the same number of instances.
    """

    # calculate the minimum per class over all subjects
    min_instances = class_instances_df.min(axis=0)
    min_instances.index = min_instances.index.astype(int)

    # get the number of instances belonging to the walk and stand subclasses
    min_instances_walk = min_instances.loc[SUB_ACTIVITIES_WALK_LABELS].min()
    min_instances_stand = min_instances.loc[SUB_ACTIVITIES_STAND_LABELS].min()

    # find the class with the least amount of instances and set this as the number of instances
    # that need to be sampled from each sub-class
    instances_sit = min(min_instances_walk, min_instances_stand)
    instances_walk = instances_sit
    instances_stand = instances_sit

    return instances_sit, instances_stand, instances_walk


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


def _balance_subject_data(subject_features: np.array, feature_names: List[str], instances_sit: int,
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

    # list for holding the indices for each sub_class
    indices_for_balancing = []

    # cycle over the unique labels
    for sub_class_label in np.unique(sub_class_labels):

        # get the indices of the class
        class_indices = np.where(sub_class_labels == sub_class_label)[0]

        # shuffle the indices
        class_indices = np.random.permutation(class_indices)

        # retrieve the indices to balance the data
        if sub_class_label in SUB_ACTIVITIES_STAND_LABELS:

            indices_for_balancing.append(class_indices[:instances_stand])

        elif sub_class_label in SUB_ACTIVITIES_WALK_LABELS:

            indices_for_balancing.append(class_indices[:instances_walk])

        else:  # sit

            indices_for_balancing.append(class_indices[:instances_sit])

    # concatenate all indices
    indices_for_balancing = np.concatenate(indices_for_balancing)

    # retrieve the balanced features
    return subject_features[indices_for_balancing, :]
