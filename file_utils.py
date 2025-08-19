"""
Utility functions for file handling.

Available Functions
-------------------
[Public]
create_dir(...): creates a new directory in the specified path.
remove_file_duplicates(...): Removes duplicate files in case the file is stored as both '.npy' and '.csv'.
load_json_file(...): Loads a json file.
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
from typing import List, Dict, Any


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

    return json_dict #output dict 
