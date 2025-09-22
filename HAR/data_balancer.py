"""
Functions for balancing data for model training. The functions allow for balancing based on the main (sit, stand, walk)
or sub-classes.

Available Functions
-------------------
[Public]
balance_main_class(...): Determines the number of instances to sample from each sub-class so that the aggregated main class distributions (Sit, Stand, Walk) remain balanced.
balance_sub_class(...): Determines the number of instances to sample from each sub-class so that all sub-classes (from Sit, Stand, Walk) have an equal number of instances.
balance_subject_data(...): Balances the subject's data by selecting the needed number of instances from each subclass.
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
from typing import Tuple

# internal imports
from constants import SUB_ACTIVITIES_WALK_LABELS, SUB_ACTIVITIES_STAND_LABELS
# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
MAIN_CLASS_BALANCING = 'main_classes'
SUB_CLASS_BALANCING = 'sub_classes'

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def balance_main_class(class_instances_df: pd.DataFrame) -> Tuple[int, int, int]:
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


def balance_sub_class(class_instances_df: pd.DataFrame) -> Tuple[int, int, int]:
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


def balance_subject_data(sub_class_labels: np.array, instances_sit: int,
                          instances_stand: int, instances_walk: int) -> np.array:
    """
    Balances the subject's data by selecting the needed number of instances from each subclass.
    The function returns a numpy.array containing the indices of the instances that were chosen. This index array
    can be used to select the data instances from the loaded data.

    :param sub_class_labels: Label vector containing the sub-class labels for each instance that pertain to the subject
    within the dataset.
    :param instances_sit: The number of instances to retain for sitting subclasses.
    :param instances_stand: The number of instances to retain for standing subclasses.
    :param instances_walk: The number of instances to retain for walking subclasses.
    :return: A  numpy.array with the indices of the instances that were chosen.
    """

    # list for holding the indices for each sub_class
    indices_for_balancing = []

    # list for holding the existing subclasses
    list_subclass = []

    # cycle over the unique labels
    for sub_class_label in np.unique(sub_class_labels):

        # add the subclass to the list
        list_subclass.append(sub_class_label)

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
    return indices_for_balancing