"""
Utility functions for file handling.

Available Functions
-------------------
[Public]
create_dir(...): creates a new directory in the specified path.
remove_file_duplicates(...): Removes duplicate files in case the file is stored as both '.npy' and '.csv'.
validate_activity_input(...): Checks whether the provided activities are valid.
get_labels(...): Gets the labels for the main and sub-activity string contained in the file name.
load_json_file(...): Loads a json file.
save_json_file(...): Stores a dict into a json file
------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import json
from typing import List, Dict, Any, Tuple

# internal imports
from constants import ACTIVITY_MAIN_SUB_CLASS, MAIN_CLASS_KEY, VALID_ACTIVITIES

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def create_dir(path, folder_name):
    """
    creates a new directory in the specified path
    :param path: the path in which the folder_name should be created
    :param folder_name: the name of the folder that should be created
    :return: the full path to the created folder
    """

    # join path and folder
    new_path = os.path.join(path, folder_name)

    # check if the folder does not exist yet
    if not os.path.exists(new_path):
        # create the folder
        os.makedirs(new_path)

    return new_path


def remove_file_duplicates(found_files, default_input_file_type) -> List[str]:
    """
    Removes duplicate files in case the file is stored as both '.npy' and '.csv'. In this case only the files
    corresponding to the default file type are kept.
    :param found_files: the files that were found
    :param default_input_file_type: the default input file type.
    :return: a list of the files to be loaded
    """

    # get the file types
    file_types = list(set(os.path.splitext(file)[1] for file in found_files))

    # check if there were multiple file types found
    # (e.g., segmented activities were stored as both .csv and .npy)
    # in this case only consider the files that match the default input file type
    if len(file_types) >= 2:
        print(
            f"Found more than one file type, for the activity to be loaded, in the folder. Found file types: {file_types}."
            f"\nOnly considering \'{default_input_file_type}\' files.")

        # get only the files that correspond to input file type
        found_files = [file for file in found_files if default_input_file_type in file]

    return found_files


def validate_activity_input(activities: List[str]) -> List[str]:
    """
    Checks whether the provided activities are valid.
    :param activities: list of string containing the activities
    :return: List of strings containing the valid activities
    """

    # check validity of provided activities
    invalid_activities = [chosen_activity for chosen_activity in activities if chosen_activity not in VALID_ACTIVITIES]

    # remove invalid activities
    if invalid_activities:

        print(f"-->The following provided activities are not valid: {invalid_activities}"
              "\n-->These activities are not considered for feature extraction")

        # filter out invalid activities
        activities = [valid_activity for valid_activity in activities if valid_activity in VALID_ACTIVITIES]

        # only provided non-valid activity strings
        if not activities:
            raise ValueError(
                f"None of the provided activities is supported. Please chose from the following: {VALID_ACTIVITIES}")

    return activities


def get_labels(main_sub_activity_str: str) -> Tuple[str, str]:
    """
    Gets the labels for the main and sub-activity string contained in the file name. The string encodes the main and
    sub-activity as {main_activity}_{sub_activity} (e.g., sitting_sit, cabinets_folders, stairs_down).
    :param main_sub_activity_str: the filename without its file type ending (e.g, .py)
    :return: tuple containing the corresponding main and sub-class labels as integers
    """

    print("--> getting labels from file name")
    # get main and sub-activity
    main_activity, sub_activity = main_sub_activity_str.split('_')[:2]

    # get corresponding main and subclasses
    main_class = ACTIVITY_MAIN_SUB_CLASS[main_activity][MAIN_CLASS_KEY]
    sub_class = ACTIVITY_MAIN_SUB_CLASS[main_activity][sub_activity]

    print(f'--> main activity: {main_activity} | class: {main_class}'
          f'\n--> sub-activity: {sub_activity} | class: {sub_class}')

    return main_class, sub_class


def load_json_file(json_path: str) -> Dict[Any, Any]:
    """
    Loads a json file.
    :param json_path: str
        Path to the json file
    :return: Dict[Any,Any]
    Dictionary containing the features from TSFEL
    """

    # read json file to a features dict
    with open(json_path, "r") as file:
        json_dict = json.load(file)

    return json_dict


def save_json_file(json_dict: Dict[Any, Any], file_name: str, folder_path: str) -> None:
    """
    Stores a dict into a json file
    :param json_dict: dictionary for storing into json file
    :param file_name: name of the json file
    :param folder_path: path to where the file should be stored
    :return: None
    """

    with open(os.path.join(folder_path, file_name), "w") as json_file:

        json.dump(json_dict, json_file)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
