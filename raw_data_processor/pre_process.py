"""
Functions for pre-processing ACC, GYR, MAG, and ROTATION VECTOR signals.

Available Functions
-------------------
[Public]
pre_process_inertial_data(...): Applies the pre-processing pipeline of "A Public Domain Dataset for Human Activity Recognition Using Smartphones".
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
from tqdm import tqdm

# internal imports
from .filters import median_and_lowpass_filter, gravitational_filter


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def pre_process_inertial_data(sensor_data: np.array, is_acc: bool = False, fs: int = 100, normalize: bool = False) -> np.array:
    """
    Applies the pre-processing pipeline of "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    (https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf). The pipeline consists of:
    (1) applying a median filter
    (2) applying a 3rd order low-pass filter with a cut-off at 20 Hz

    in case the sensor data belongs to an ACC sensor the following additional steps are performed.
    (3) applying a 3rd order low-pass filter with a cut-off at 0.3 Hz to obtain gravitational component
    (4) subtract gravitational component from ACC signal

    :param sensor_data: the sensor data.
    :param is_acc: boolean indicating whether the sensor is an accelerometer.
    :param fs: the sampling frequency of the sensor data (in Hz).
    :param normalize: boolean to indicate whether the data should be normalized (division by the max)
    :return: numpy.array containing the pre-processed data.
    """

    # apply median and lowpass filter
    filtered_data = median_and_lowpass_filter(sensor_data, fs=fs)

    # check if signal is supposed to be normalized
    if normalize:
        # normalize the signal
        filtered_data = filtered_data / np.max(filtered_data)

    # check if sensor is ACC (additional steps necessary
    if is_acc:
        # print('Applying additional processing steps')

        # get the gravitational component
        gravitational_component = gravitational_filter(filtered_data, fs=fs)

        # subtract the gravitational component
        filtered_data = filtered_data - gravitational_component

    return filtered_data


def slerp_smoothing(quaternion_array: np.array, smooth_factor: float = 0.5, scalar_first: bool = False,
                    return_numpy: bool = True, return_scalar_first: bool = False) -> np.array:
    """
    The implementation is based on:
    https://www.mathworks.com/help/fusion/ug/lowpass-filter-orientation-using-quaternion-slerp.html
    :param quaternion_array:
    :param smooth_factor:
    :param scalar_first:
    :param return_numpy:
    :param return_scalar_first:
    :return: returns quaternions in either scalar first (w, x, y, z) or scalar last notation (x, y, z, w)
    """

    # change quaternion notation to scalar first notation (w, x, y, z) if it is not
    # this is needed as pyquaternion assumes this notation
    if not scalar_first:

        quaternion_array = np.hstack((quaternion_array[:, -1:], quaternion_array[:, :-1]))

    # get the number of rows
    num_rows = quaternion_array.shape[0]

    # array for holding the result
    smoothed_quaternion_array = np.zeros(num_rows, dtype=object)

    # initialize the first quaternion
    smoothed_quaternion_array[0] = Quaternion(quaternion_array[0])

    # cycle over the quaternion series
    for row in tqdm(range(1, num_rows), ncols=50, bar_format="{l_bar}{bar}| {percentage:3.0f}% {elapsed}"):

        # get the previous and the current quaternion
        q_prev = smoothed_quaternion_array[row - 1]
        q_curr = Quaternion(quaternion_array[row])

        # perform SLERP
        q_slerp = Quaternion.slerp(q_prev, q_curr, smooth_factor)

        # add the quaternion to the smoothed series
        smoothed_quaternion_array[row] = q_slerp

    # return as numpy array
    if return_numpy:

        # transform the output into a 2D numpy array
        smoothed_quaternion_series_numpy = np.zeros_like(quaternion_array)

        for row, quat in enumerate(smoothed_quaternion_array):

            smoothed_quaternion_series_numpy[row] = quat.elements

        # return in (x, y, z, w) notation
        if not return_scalar_first:

            smoothed_quaternion_series_numpy = np.hstack((smoothed_quaternion_series_numpy[:, 1:],
                                                          smoothed_quaternion_series_numpy[:, :1]))

        return smoothed_quaternion_series_numpy

    return smoothed_quaternion_array

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
