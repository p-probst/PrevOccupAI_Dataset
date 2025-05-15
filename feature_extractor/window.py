"""
Functions for signal windowing.

Available Functions
-------------------
[Public]
get_sliding_windows_indices(...): Function to obtain window indices with an adjustable overlap between windows
window_data(...): Function to slice the data into windows defined by indices
window_scaling(...): Performs scaling on each window using the provided scaler.
validate_scaler_input(...): Checks whether the provided scaler is valid
------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
MINMAX_SCALER = 'minmax'
STANDARD_SCALER = 'standard'
SCALERS = [MINMAX_SCALER, STANDARD_SCALER]


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def get_sliding_windows_indices(signal: np.array, fs: int, window_size: float, overlap: float = 0.5):
    """
    Function to obtain window indices with an adjustable overlap between windows.
    :param signal: a 1-D signal or the time axis of multiple synchronised signals
    :param fs: the sampling rate of the signal
    :param window_size: the window size in seconds
    :param overlap: the overlap

    :return: An index array containing all the windows and the number of samples that need to be padded the signal
             in order to accommodate all windows.
    """

    # check validity of input
    if np.ndim (signal) > 1:
        raise ValueError("Invalid input: Only 1-D signal or time axis of multiple synchronised signals allowed.")
    if window_size < 1 / fs:
        raise ValueError(
            "Invalid input: The window size you chose is smaller than the sampling interval T=1/fs.")
    if not (0 <= overlap < 1):
        raise ValueError("Invalid input: The overlap parameter has to be within the interval [0, 1).")

    # get the number of samples in the signal
    num_samples_signal = len(signal)

    # calculate the number of samples that fit into a window
    num_samples_window = int(fs * window_size)

    # calculate the number of samples in a hop
    num_samples_hop = int(num_samples_window * (1 - overlap))

    if num_samples_hop < 1:
        raise ValueError("Invalid overlap: Hop size is less than 1 sample. Reduce the overlap.")

    # calculate the number full windows that fit into the signal
    num_windows = (num_samples_signal - num_samples_window) // num_samples_hop + 1

    # calculate the starts of the windows
    starts = np.arange(0, num_windows) * num_samples_hop

    # calculate the indices array (broadcasting is applied here)
    indices = np.expand_dims(starts, axis=1) + np.arange(num_samples_window)

    return indices


def window_data(data, indices):
    """
    Function to slice the data into windows defined by indices.
    :param data: array containing the signals to be sliced. Either 1-D or (MxN), where M is the signal length in samples
                 and N is the number of signals / channels.
    :param indices: array containing the indices that indicate the positions of the windows. The indices array can be
                    obtained by using the get_sliding_windows_indices(...) function
    :return: a multi-dimensional array containing the signal(s) sliced into windows.
             In case data is 1-D, then the array is of shape [number of windows, window length (in samples)].
             In case data is 2-D, then the array is of shape [number of windows, window length (in sampples), number of signals/channels]
    """
    return data[indices]


def window_scaling(windowed_data: np.array, scaler: str = 'minmax') -> np.array:
    """
    Performs scaling on each window using the provided scaler.
    :param windowed_data: the windowed data
    :param scaler: the type of scaler. Can be either 'minmax' (min-max scaling) or 'standard' (standardizing).
                   Default: 'minmax'.
    :return: The windowed data, but with each window scaled according to the provided scaler.
    """
    # check input validity
    validate_scaler_input(scaler)

    if windowed_data.ndim != 3:

        raise ValueError(f"Invalid input size. The data you provided has {windowed_data.ndim} dimension(s)."
                         f"\nThe windowed_data should have dimension 3 dimension: [num_windows, window_size, channels].")

    # scale data
    if scaler == MINMAX_SCALER:
        print('--> min-max normalization of each window')

        min_vals = np.min(windowed_data, axis=1, keepdims=True)
        max_vals = np.max(windowed_data, axis=1, keepdims=True)

        scaled_data = (windowed_data - min_vals) / (max_vals - min_vals)

    else:

        mean_vals = np.mean(windowed_data, axis=1, keepdims=True)
        std_vals = np.mean(windowed_data, axis=1, keepdims=True)

        scaled_data = (windowed_data - mean_vals) / std_vals

    return scaled_data


def validate_scaler_input(scaler: str) -> None:
    """
    Checks whether the provided scaler is valid.
    :param scaler: the type of scaler. Can be either 'minmax' (min-max scaling) or 'standard' (standardizing).
                   Default: 'minmax'
    :return: None
    """

    # check input validity
    if scaler not in SCALERS:

        raise ValueError(f"The window scaler you have chosen is not supported. Chosen scaler: {scaler}."
                         f"\nPlease choose from the following: {SCALERS}")

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


