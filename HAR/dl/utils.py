"""
utility functions for deep learning module

Available Functions
-------------------
[Public]
select_idle_gpu(...): Selects an idle GPU that meets the usage thresholds.
configure_seed(...): Configure random seeds for reproducibility in Python, NumPy, and PyTorch.
------------------
[Private]
None
------------------
"""
import os.path

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import GPUtil
import torch
import random
import numpy as np
from typing import List, Tuple

from HAR.dl import HARRnn
from constants import LSTM_MODEL, GRU_MODEL, VALID_SENSORS, ACC, GYR, MAG, ROT


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def select_idle_gpu(max_load: float = 0.1, max_memory: float = 0.1) -> torch.device:
    """
    Selects an idle GPU that meets the usage thresholds.
    Raises RuntimeError if none available, printing current GPU stats.
    :param max_load: Maximum allowed GPU load (0–1).
    :param max_memory: Maximum allowed memory usage (0–1).
    :return: CUDA device for the selected GPU.
    """

    # Get available GPUs based on criteria
    available = GPUtil.getAvailable(order='first', limit=1,
                                     maxLoad=max_load, maxMemory=max_memory)

    if available:
        gpu_id = available[0]
        print(f"INFO: Selected GPU {gpu_id}: {GPUtil.getGPUs()[gpu_id].name}")
        return torch.device(f"cuda:{gpu_id}")
    else:
        # No free GPU found → print stats and raise error
        gpus = GPUtil.getGPUs()
        print("INFO: No idle GPU found. Current GPU usage:")
        for gpu in gpus:
            print(f"GPU {gpu.id} | {gpu.name} | "
                  f"Load: {gpu.load*100:.1f}% | "
                  f"Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB "
                  f"({gpu.memoryUtil*100:.1f}%)")
        raise RuntimeError(
            "ERROR: All GPUs are currently in use. "
            "Consider waiting or manually setting a gpu_id.")


def configure_seed(seed: int):
    """
    Configure random seeds for reproducibility in Python, NumPy, and PyTorch.

    This function sets the seed for Python's `random` module, NumPy, and PyTorch
    (both CPU and GPU). It also sets PyTorch to use deterministic algorithms
    for reproducible results on GPU and disables the CuDNN benchmark to avoid
    nondeterministic behavior.

    :param seed: Integer seed value to use for all random number generators.
    :type seed: int

    :raises TypeError: If the seed is not an integer.
    """

    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU support
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_har_model(model_path: str, use_sensors: List[str], device: torch.device) -> Tuple[torch.nn.Module, int, int]:
    """
    load the deep learning model for human activity recognition based on the provided model file
    :param model_path: path to the model state_dict containing all the model params
    :param use_sensors: list of sensors (as strings) indicating which sensor should be used for classification. When
                        a sensor is chosen, then all channels of this sensor are used.
                        The following options can be chosen: 'ACC', 'GYR', 'MAG', 'ROT'.
    :param device: device to run the model on. Can be either a GPU or CPU
    :return: HAR deep learning model in .eval() mode (only for classification)
    """

    # check input validity
    # 1. model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist. Please make sure that your path is pointing"
                                f"to the correct file.")

    # 2. use_sensors
    invalid_sensors = set(use_sensors) - set(VALID_SENSORS)
    if invalid_sensors:
        raise ValueError(f"Invalid sensors provided: {invalid_sensors}"
                         f"\nPlease choose from the following options: {VALID_SENSORS}")

    # TODO: implement loading for CNN-LSTM
    # extract relevant params from path
    # model type (lstm, gru, or CNN-LSTM)
    path_parts = model_path.split("\\")
    model_string = next((part for part in path_parts if part.startswith("trained")), None)

    # extract the window size, sequence length, and model type
    window_size = _extract_hyper_params(model_string, "wsize")
    seq_len = _extract_hyper_params(model_string, "seqlen")

    # extract hyper parameters
    state_dict_file = os.path.basename(model_path)
    model_type = state_dict_file.split("_")[1]
    hidden_size = _extract_hyper_params(state_dict_file, "hs")
    num_layers = _extract_hyper_params(state_dict_file, "nl")

    # calculate the number of input features needed
    num_sensor_channels = _calc_num_channels(use_sensors)

    # load the model
    if model_type == LSTM_MODEL or model_type == GRU_MODEL:

        # init the model
        har_model = HARRnn(model_type=model_type, num_features= num_sensor_channels * seq_len,
                       hidden_size=hidden_size, num_layers=num_layers, num_classes=3, dropout=0)


    elif "cnn" in model_type:

        # har_model = ...
        print('loading CNN model')

    else:

        ValueError(f"The model type that was loaded is not supported. Loaded model type: {model_type}")

    # load the model parameters
    state_dict = torch.load(model_path, map_location=torch.device(device))

    # load the parameters into the model and set the model to eval mode
    har_model.load_state_dict(state_dict)
    har_model.eval()

    return har_model, window_size, seq_len




# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _extract_hyper_params(state_dict_filename: str, search_param: str) -> int:
    """

    :param state_dict_filename:
    :param search_param:
    :return:
    """

    # remove file extension
    filename_parts, _ = os.path.splitext(state_dict_filename)

    # split file by underscore
    filename_parts = filename_parts.split("_")

    # get the part containing the searched hyperparam
    hyper_param = next((part for part in filename_parts if part.startswith(search_param)), None)

    return int(hyper_param.split("-")[-1])


def _calc_num_channels(use_sensors: List[str]) -> int:
    """
    Calculates the number of channels that result from the chosen sensors
    :param use_sensors: list of sensors (as strings) indicating which sensor should be used for classification. When
                        a sensor is chosen, then all channels of this sensor are used.
                        The following options can be chosen: 'ACC', 'GYR', 'MAG', 'ROT'.
    :return: total number of channels
    """

    # init num channels
    num_channels = 0

    # cycle over the sensor list
    for sensor in use_sensors:

        if sensor in [ACC, GYR, MAG]:

            # update num_channels
            num_channels += 3

        elif sensor == ROT:

            num_channels += 4

    return num_channels


