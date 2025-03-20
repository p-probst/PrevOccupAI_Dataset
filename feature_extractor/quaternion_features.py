"""
Functions for extracting high-level features from quaternion timeseries'.

Available Functions
-------------------
[Public]
geodesic_distance(...): Calculates the geodesic distance between consecutive quaternions contained in the quaternion_array
------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from pyquaternion import Quaternion


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def geodesic_distance(quaternion_array: np.array, scalar_first: bool = False) -> np.array:
    """
    Calculates the geodesic distance between consecutive quaternions contained in the quaternion_array and calculates
    (1) mean distance
    (2) standard deviation of distances
    (3) total distance (total rotational path)

    :param quaternion_array: an array of shape (M x 4), where M represents the time steps (samples) of the quaternion
                             timeseries.
    :param scalar_first: boolean indicating whether scalar first (w, x, y, z) notation is used or not (x, y, z, w)
    :return: array containing the extracted features
    """

    # change quaternion notation to scalar first notation (w, x, y, z) if it is not
    # this is needed as pyquaternion assumes this notation
    if not scalar_first:
        quaternion_array = np.hstack((quaternion_array[:, -1:], quaternion_array[:, :-1]))

    # get the number of rows
    num_rows = quaternion_array.shape[0]

    # array for holding the geodesic distances
    # this array has one item less than the original array as the differences are calculated between quaternions
    geo_dist = np.zeros(num_rows - 1)

    # cycle over the quaternions
    for row in range(1, num_rows):

        # get the quaternions
        q_prev = Quaternion(quaternion_array[row - 1])
        q_curr = Quaternion(quaternion_array[row])

        # calculate geodesic distance
        geo_dist[row - 1] = Quaternion.distance(q_prev, q_curr)

    # calculate the features
    mean_dist = np.mean(geo_dist)
    std_dist = np.std(geo_dist)
    total_dist = np.sum(geo_dist)

    return np.array([mean_dist, std_dist, total_dist])

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
