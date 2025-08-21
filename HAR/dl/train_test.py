"""
Functions for handling training and test steps

Available Functions
-------------------
[Public]

------------------
[Private]
None
------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import torch
import torch.nn as nn

from constants import MAIN_ACTIVITY_LABELS


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def train_step(model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion: nn.Module,
               optimizer: torch.optim.Optimizer, cuda_device: torch.device) -> tuple[float, float]:
    """
    performs the model training for one epoch
    :param model: PyTorch model
    :param data_loader: dataloader containing the data for training
    :param criterion: loss function
    :param optimizer: optimizer
    :param cuda_device: the CUDA device to run the model on
    :return: average epoch training loss and accuracy
    """

    # set model to training (enable model parameters)
    model.train()

    # variables for tracking loss and accuracy
    running_loss = 0.0
    running_correct_preds = 0
    total_num_samples = 0

    # cycle over the batches contained in data loader
    for X_batch, y_main_batch, y_sub_batch in data_loader:

        # move batch to cuda device
        X_batch = X_batch.to(cuda_device)

        # check whether the model does main or sub-class classification
        # TODO: define constant for the classes
        if model.num_classes == len(MAIN_ACTIVITY_LABELS):
            y_batch = y_main_batch.to(cuda_device)

        else:
            y_batch = y_sub_batch.to(cuda_device)

        # get batch size and track the total number of samples
        batch_size = X_batch.size(0)
        total_num_samples += batch_size

        # clear accumulated gradients from previous batch
        optimizer.zero_grad()

        # pass data through the model (forward pass)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # compute backprop + update params
        loss.backward()
        optimizer.step()

        # track loss
        running_loss += loss.item() * batch_size

        # track accuracy
        preds = outputs.argmax(dim=1)
        running_correct_preds += (preds == y_batch).sum().item()

    # compute epoch averages
    epoch_loss = running_loss / total_num_samples
    epoch_accuracy = running_correct_preds / total_num_samples

    return epoch_loss, epoch_accuracy


def test_step(model: nn.Module, data_loader: torch.utils.data.DataLoader,
              criterion: nn.Module, cuda_device: torch.device) -> tuple[float, float]:
    """
    performs model testing for one epoch (no gradient updates)
    :param model: PyTorch model
    :param data_loader: data loader containing the data for testing
    :param criterion:  loss function
    :param cuda_device: CUDA device to run the model on
    :return: average epoch validation loss and accuracy
    """

    # set model to evaluation mode
    model.eval()

    # variables for tracking loss and accuracy
    running_loss = 0.0
    running_correct_preds = 0
    total_num_samples = 0

    # turn off gradient computation
    with torch.no_grad():

        # cycle over the batch
        for X_batch, y_main_batch, y_sub_batch in data_loader:

            # move batch to cuda device
            X_batch = X_batch.to(cuda_device)

            # check whether the model does main or sub-class classification
            if model.num_classes == len(MAIN_ACTIVITY_LABELS):
                y_batch = y_main_batch.to(cuda_device)

            else:
                y_batch = y_sub_batch.to(cuda_device)

            # get batch size and track the total number of samples
            batch_size = X_batch.size(0)
            total_num_samples += batch_size

            # pass data through the model (forward pass)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # track loss
            running_loss += loss.item() * batch_size

            # track accuracy
            preds = outputs.argmax(dim=1)
            running_correct_preds += (preds == y_batch).sum().item()

    # compute epoch averages
    epoch_loss = running_loss / total_num_samples
    epoch_accuracy = running_correct_preds / total_num_samples

    return epoch_loss, epoch_accuracy

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #