"""
Functions for filtering ACC, GYR, MAG, and ROTATION VECTOR signals

Available Functions
-------------------
[Public]
median_and_lowpass_filter(...): Applies a median filter followed by a butterworth lowpass filter.
gravitational_filter(): Function to filter out the gravitational component of ACC signals.
get_envelope(...): Gets the envelope of the passed signal.

------------------
[Private]
_butter_lowpass_filter(...): Filters a signal using a butterworth lowpass filter.
_moving_average(...): Application of a moving average filter for signal smoothing.
_window_rms(...): Passes a root-mean-square filter over the data.
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from scipy import signal

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
RMS = 'rms'
LOW_PASS = 'lowpass'
MOVING_AVERAGE = 'MA'


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def median_and_lowpass_filter(sensor_data: np.ndarray, fs: int, medfilt_window_length=11) -> np.ndarray:
    """
    Applies a median filter followed by a butterworth lowpass filter. The lowpass filter is 3rd order with a cutoff
    frequency of 20 Hz . The processing scheme is based on:
    "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf

    :param sensor_data: a 1-D or (MxN) array, where M is the signal length in samples and
                        N is the number of signals / channels.
    :param fs: the sampling frequency of the acc data.
    :param medfilt_window_length: the length of the median filter (has to be odd). Default: 11
    :return: the filtered data
    """

    # define the filter
    order = 3
    f_c = 20
    filt = signal.butter(order, f_c, fs=fs, output='sos')

    # copy the array
    filtered_data = sensor_data.copy()

    # check the dimensionality of the input
    if filtered_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(filtered_data.shape[1]):
            # get the channel
            sig = sensor_data[:, channel]

            # apply the median filter
            sig = signal.medfilt(sig, medfilt_window_length)

            # apply butterworth filter
            filtered_data[:, channel] = signal.sosfilt(filt, sig)

    else:  # 1-D array

        # apply median filter
        med_filt = signal.medfilt(sensor_data, medfilt_window_length)

        # apply butterworth filter
        filtered_data = signal.sosfilt(filt, med_filt)

    return filtered_data


def gravitational_filter(acc_data: np.ndarray, fs: int) -> np.ndarray:
    """
    Function to filter out the gravitational component of ACC signals using a 3rd order butterworth lowpass filter with
    a cuttoff frequency of 0.3 Hz
    The implementation is based on:
    "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf
    :param acc_data: a 1-D or (MxN) array, where where M is the signal length in samples and
                 N is the number of signals / channels.
    :param fs: the sampling frequency of the acc data.
    :return: the gravitational component of each signal/channel contained in acc_data
    """

    # define the filter
    order = 3
    f_c = 0.3
    filter = signal.butter(order, f_c, fs=fs, output='sos')

    # copy the array
    gravity_data = acc_data.copy()

    # check the dimensionality of the input
    if gravity_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(gravity_data.shape[1]):
            # get the channel
            sig = acc_data[:, channel]

            # apply butterworth filter
            gravity_data[:, channel] = signal.sosfilt(filter, sig)

    else:  # 1-D array

        gravity_data = signal.sosfilt(filter, acc_data)

    return gravity_data


def get_envelope(signal_array: np.array, envelope_type: str = RMS, type_param: int = 10, fs: int = 100) -> np.array:
    """
    Gets the envelope of the passed signal. There are three types available
    1. 'lowpass': uses a lowpass filter
    2. 'ma': uses a moving average filter
    3. 'rms': uses a root-mean-square filter
    :param signal_array: the signal
    :param envelope_type: the type of filter that should be used for getting the envelope as defined above
    :param type_param: the parameter for the envelope_type. The following options are available (based on the envelope_type)
                       'lowpass': type_param is the cutoff frequency of the lowpass filter
                       'ma': type_param is the window size in samples
                       'rms': type_param is the window size in samples
    :param fs: the sampling frequency of the acc data.
    :return: numpy.array containing the envelope of the signal
    """

    # check for the passed type
    if envelope_type == LOW_PASS:
        # apply lowpass filter
        filtered_signal = _butter_lowpass_filter(signal_array, cutoff=10, fs=fs)
    elif envelope_type == MOVING_AVERAGE:
        # apply moving average
        filtered_signal = _moving_average(signal_array, wind_size=type_param)
    elif envelope_type == RMS:
        # apply rms
        filtered_signal = _window_rms(signal_array, window_size=type_param)

    else:
        # undefined filter type passed
        raise IOError('the type you chose is not defined.')

    return filtered_signal


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _butter_lowpass_filter(signal_array: np.array, cutoff: int, fs: int, order: int = 4) -> np.array:
    """
    Filters a signal using a butterworth lowpass filter.
    :param signal_array: the signal
    :param cutoff: frequency cutoff
    :param fs: sampling frequency
    :param order: order of the filter
    :return: the filtered signal
    """
    # design filter
    b, a = signal.butter(order, cutoff, btype='low', fs=fs, analog=False)

    # apply filter
    filtered_signal = signal.lfilter(b, a, signal_array)

    return filtered_signal


def _moving_average(signal_array: np.array, wind_size: int = 3) -> np.array:
    """
    Application of a moving average filter for signal smoothing.
    :param signal_array: the signal
    :param wind_size: the window_size
    :return: filtered signal
    """

    # cast window size to int
    wind_size = int(wind_size)

    # calculate cumulative sum
    ret = np.cumsum(signal_array, dtype=float)

    # apply window
    ret[wind_size:] = ret[wind_size:] - ret[:-wind_size]

    return np.concatenate((np.zeros(wind_size - 1), ret[wind_size - 1:] / wind_size))


def _window_rms(signal_array: np.array, window_size: int = 3) -> np.array:
    """
    Passes a root-mean-square filter over the data.
    :param signal_array: the data for which the root-mean-square should be calculated
    :param window_size: the window size
    :return: the rms for the given window
    """

    # square the data
    data_squared = np.power(signal_array, 2)

    # create window
    window = np.ones(window_size) / float(window_size)

    # calculate RMS
    return np.sqrt(np.convolve(data_squared, window, 'valid'))


