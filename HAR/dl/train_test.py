"""
Functions for handling training and test steps

Available Functions
-------------------
[Public]
run_model_training(...): training/test loop to train the model and evaluate its performance.
train_step(...): performs the model training for one epoch.
test_step(...): performs model testing for one epoch (no gradient updates)
plot_performance_history(...): plots the performance history (loss and accuracy) over the epochs.
------------------
[Private]
None
------------------
"""
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import os

# internal imports
from constants import MAIN_ACTIVITY_LABELS

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
TRAIN_ACC_KEY = "train_acc"
TRAIN_LOSS_KEY = "train_loss"
VAL_ACC_KEY = "val_acc"
VAL_LOSS_KEY = "val_loss"

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def run_model_training(
                       model: nn.Module, model_save_path: str,
                       train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader,
                       criterion: nn.Module, optimizer: torch.optim.Optimizer,
                       cuda_device: torch.device, num_epochs: int, patience: int = 5) -> dict[str, list[float]]:
    """
    training/test loop to train the model and evaluate its performance. During the loop the best model is stored.
    The loop is terminated using early stopping in case the performance does not improve according to the set patience.
    :param model: PyTorch model to be trained
    :param model_save_path: file path to where the model is stored
    :param train_dataloader: PyTorch DataLoader for training data
    :param test_dataloader: PyTorch DataLoader for validation data
    :param criterion: loss function to be used
    :param optimizer: optimizer to be used
    :param cuda_device: CUDA device to run the model on
    :param num_epochs: number training epochs
    :param patience: number of epochs to wait for improvement before initiating early stopping
    :return: dictionary containing train/val loss and accuracy per epoch. These can be accessed through the following
             keys: "train_loss", "train_acc", "val_loss", "val_acc"
    """

    # init performance dict
    performance_history = {
        TRAIN_LOSS_KEY: [],
        TRAIN_ACC_KEY: [],
        VAL_LOSS_KEY: [],
        VAL_ACC_KEY: []
    }

    # init variables for tracking performance
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    # cycle over the epochs
    for epoch in range(1, num_epochs + 1):

        print("\n----------------------------------------------------")
        print(f"Epoch: {epoch}/{num_epochs}")

        # perform training step
        train_loss, train_acc = train_step(model, train_dataloader, criterion, optimizer, cuda_device)

        # perform test/validation step
        val_loss, val_acc = test_step(model, test_dataloader, criterion, cuda_device)

        # store metrics
        performance_history[TRAIN_LOSS_KEY].append(train_loss)
        performance_history[TRAIN_ACC_KEY].append(train_acc)
        performance_history[VAL_LOSS_KEY].append(val_loss)
        performance_history[VAL_ACC_KEY].append(val_acc)

        # check performance improvement
        if val_acc > best_val_acc:

            # update best accuracy and epoch
            best_val_acc = val_acc
            best_epoch = epoch

            # reset early-stopping counter
            epochs_no_improve = 0

            # obtain model name
            model_name = f"{model.__class__.__name__}.pt"

            # store best model params
            torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
            print(f"INFO: New best model obtained at epoch {epoch} (val. acc: {best_val_acc:.4f}")

        else: # no improvement
            epochs_no_improve += 1
            print(f"INFO: No improvement in validation accuracy for {epochs_no_improve} epoch(s)")

        # print progress
        print(f"Metrics:"
              f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
              f"\n  Val Loss: {val_loss:.4f} |   Val Acc: {val_acc:.4f}")

        # check if early-stopping is needed
        if epochs_no_improve >= patience:
            print(f"INFO: Early-stopping at epoch {epoch}.")
            break

    print(f"\nTraining loop completed. Best val. acc: {best_val_acc:.4f} obtain in epoch: {best_epoch}")
    print(f"\nBest model saved to: {model_save_path}")

    return performance_history


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


def plot_performance_history(performance_dict: dict, model_name: str, save_path: str) -> None:
    """
    plots the performance history (loss and accuracy) over the epochs. The plot is stored as a .svg file.
    :param performance_dict: dictionary containing train/val loss and accuracy per epoch stored as lists.
                             These can be accessible through the following keys: "train_loss", "train_acc", "val_loss", "val_acc"
    :param model_name: the name of the model (used for storing the plot)
    :param save_path: path to where the plot should be saved
    :return: None
    """

    # get the epochs from the dict
    epochs = range(1, len(performance_dict[TRAIN_LOSS_KEY]) + 1)

    # create plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # generate loss subplot
    axes[0].plot(epochs, performance_dict[TRAIN_LOSS_KEY], label="Train Loss", color="#1c222b")
    axes[0].plot(epochs, performance_dict[VAL_LOSS_KEY], label="Validation Loss", color="#4d92d0")
    axes[0].set_title("Training vs. Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # generate accuracy subplot
    axes[1].plot(epochs, performance_dict[TRAIN_ACC_KEY], label="Train Accuracy", color="#1c222b")
    axes[1].plot(epochs, performance_dict[VAL_ACC_KEY], label="Validation Accuracy", color="#4d92d0")
    axes[1].set_title("Training vs. Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # Add main title for the figure
    fig.suptitle("Model Training History", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # save the plot
    plt.savefig(os.path.join(save_path, f"{model_name}.svg"))

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #



