"""
Functions for model pipelines to apply models on real-world data

Available Functions
-------------------
[Public]
dl_pipeline(...): pipeline for using the DL model for HAR classification.
normalize_data(...): normalizes the sensor data
dl_windowing(...): windows the data into the format needed for the DL models
classify(...):
------------------
[Private]

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import torch
from typing import List, Tuple

from HAR.dl.dataset_generator import GLOBAL_STATS, SENSOR_MEAN, SENSOR_VAR
from HAR.post_process import expand_classification
from constants import IMPULSE_LENGTH
from feature_extractor import trim_data, get_sliding_windows_indices, window_data
from file_utils import load_json_file
from raw_data_processor import pre_process_sensors, load_data_from_same_recording

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def dl_pipeline(data_path: str, har_model: torch.nn.Module, sensors: List[str], w_size: int, seq_len: int, fs=100, stats_path: str = None) -> np.ndarray:
    """
    pipeline for using the DL model for HAR classification. The pipeline handles all necessary steps:
    (1) loading the data
    (2) pre-processing
    (3) normalization
    (4) windowing
    (5) classification
    (6) post-processing: TODO NOT YET IMPLEMENTED
    :param data_path:
    :param har_model:
    :param sensors:
    :param w_size:
    :param seq_len:
    :param fs:
    :param stats_path:
    :return: numpy array containing the model output
    """

    # load the data
    subject_data = load_data_from_same_recording(data_path, sensors, fs=fs)

    # convert data to a numpy array and remove the time ('t') column
    sensor_data = subject_data.values[:, 1:]

    # get the sensor names (needed for pre-processing)
    sensor_names = subject_data.columns[1:]
    print(f"loaded sensors: {sensor_names}")

    # pre-process the data
    sensor_data = pre_process_sensors(sensor_data, sensor_names)

    # remove impulse response
    sensor_data = sensor_data[IMPULSE_LENGTH:, :]

    # trim the data to accommodate full windowing
    sensor_data, _ = trim_data(sensor_data, w_size=w_size, fs=fs)

    # normalize the data
    sensor_data = normalize_data(sensor_data, stats_path)

    # window the data
    windowed_data = dl_windowing(sensor_data, w_size, seq_len, fs)

    # classify the data
    y_pred, _ = dl_classify(har_model, windowed_data, w_size, fs)

    return y_pred


def normalize_data(sensor_data: np.ndarray, stats_path: str = None) -> np.ndarray:
    """
    normalizes the sensor data using either global statistics obtained from the training dataset or subject statistics
    obtained from the sensor data by applying z-score normalization. If stats_path is given, global normalization is
    used, otherwise subject statistics are applied.
    :param sensor_data: numpy.array containing the sensor data needed for classification
    :param stats_path: path to the file containing the global statistics. If no path is provided, subject statistics are
                       obtained from the sensor data to normalize the data.
    :return: the normalized data
    """
    data_copy = sensor_data.copy()

    # check whether a path to the global statistics was provided
    if stats_path:
        print("global normalization applied")
        # load the statistics
        stats = load_json_file(stats_path).get(GLOBAL_STATS)

        # extract the mean and standard deviation
        mean = np.array(stats[SENSOR_MEAN][:sensor_data.shape[1]])
        std = np.sqrt(np.array(stats[SENSOR_VAR])[:sensor_data.shape[1]])

    else:

        print("subject-wise normalization applied")
        # calculate mean and standard deviation directly from the sensor data
        mean = np.mean(data_copy, axis=0)
        std = np.std(data_copy, axis=0)

    # return normalized data
    return (data_copy - mean) / std


# TODO: a windowing scheme for 2D-CNN-LSTM models might be needed
def dl_windowing(sensor_data: np.ndarray, w_size: int, seq_len: int, fs: int = 100) -> np.ndarray:
    """
    windows the data into the format needed for the DL models
    :param sensor_data: numpy.array containing the sensor data needed for classification
    :param w_size: the window size in seconds
    :param seq_len: the sub-sequence length in samples
    :param fs: the sampling frequency in Hz
    :return: the windowed data in the shape [num_windows, time_steps, sequence_length, num_channels]
    """

    # perform main windowing (windowing the data into windows of w_size)
    indices = get_sliding_windows_indices(sensor_data[:, 0], fs, w_size, overlap=0)
    windowed_data = window_data(sensor_data, indices)

    # perform sub-windowing [num_windows, time_steps, sequence_length, num_channels]
    num_time_steps = windowed_data.shape[1] // seq_len
    windowed_data = windowed_data.reshape(windowed_data.shape[0], num_time_steps, seq_len, windowed_data.shape[-1])

    return windowed_data


def dl_classify(har_model: torch.nn.Module, data: np.ndarray, w_size: int, fs:int = 100, expand_classif: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    classifies the provided data using the provided HAR model
    :param har_model: the loaded HAR model
    :param data: the data to be classified in shape: [num_windows, time_steps, sequence_length, num_channels]
    :param w_size: the window size in seconds
    :param fs: the sampling frequency in Hz
    :param expand_classif: whether to expand classification to the original data length or not. Default: True
    :return: numpy.array containing the classification results
    """

    # transform data into torch tensor
    data = torch.from_numpy(data).float()

    # pass the data to the model
    logits = har_model(data)

    # pass the output through argmax to obtain classification according to class
    y_pred = logits.argmax(dim=1).detach().numpy()

    # obtain the probabilities
    y_probabilities = logits.softmax(dim=1).detach().numpy()

    if expand_classif:
        # expand the prediction to the original length of the input signal
        y_pred = expand_classification(y_pred, w_size, fs)

    return y_pred, y_probabilities

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #




