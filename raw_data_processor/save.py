"""
Functions for loading sensor data from the devices: 'phone', 'watch', 'emg'.

Available Functions
-------------------
[Public]
save_segmented_tasks(...): Saves the segmented_tasks into individual files.
create_dir(...): creates a new directory in the specified path.
------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from typing import List
import os

# internal imports
from constants import ACTIVITY_MAP, VALID_FILE_TYPES, CSV


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def save_segmented_tasks(segmented_tasks: List[pd.DataFrame], activity: str, output_path: str,
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
# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


