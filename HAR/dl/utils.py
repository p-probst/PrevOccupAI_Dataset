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

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import GPUtil
import torch
import random
import numpy as np

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
# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #