"""
Classes containing the deep learning models

Available Functions
-------------------
[Public]
HARLstm(): simple lstm model consisting of: (1) lstm layer, (2) dropout layer, (3) fully connected layer
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
from typing import List, Tuple

# internal imports
from constants import LSTM_MODEL, GRU_MODEL

# ------------------------------------------------------------------------------------------------------------------- #
#  constants
# ------------------------------------------------------------------------------------------------------------------- #

SUPPORTED_DL_MODELS = [LSTM_MODEL, GRU_MODEL]

# ------------------------------------------------------------------------------------------------------------------- #
#  public classes
# ------------------------------------------------------------------------------------------------------------------- #
class HARRnn(nn.Module):

    def __init__(self, model_type: str, num_features: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        """
        simple lstm model consisting of:
        (1) lstm layer
        (2) dropout layer
        (3) fully connected layer
        The model assumes inputs of shape [batch_size, time_steps, seq_len, num_features], where num features equates to the
        number of channels (i.e. sensor channels).
        :param model_type: Type of RNN. Can be 'lstm' or 'gru'
        :param num_features: number of features in each timestep (input size). This equates to the number of
                             sensor channels passed.
        :param hidden_size: number of hidden units in the LSTM cell
        :param num_layers: number of stacked lstm layers
        :param num_classes: number of output classes for classification
        :param dropout: dropout rate should be between [0.0 and 1.0]
        """

        # call to super class
        super().__init__()

        # init class variables
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        if self.model_type not in SUPPORTED_DL_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'lstm' or 'gru'.")

        # model architecture
        # check if it is a LSTM layer or a GRU layer
        rnn_type = nn.LSTM if self.model_type == LSTM_MODEL else nn.GRU
        self.rnn = rnn_type(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # (2) Dropout layer
        self.dropout_layer = nn.Dropout(p=dropout)

        # (3) fully connected
        self.fc_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass
        :param x: input batch of size [batch_size, time_steps, sequence_length, num_channels]
        :return:
        """

        # reshape input tensor to shape [batch_size, time_steps, sequence_length * num_channels]
        x = x.reshape(x.size(0), x.size(1), -1)

        # pass the inputs through the RNN
        # shape out: [batch_size, time_steps, hidden_size]
        out, _ = self.rnn(x)

        # obtain the last output (many-to-one classification)
        # shape last_out: [batch_size, hidden_size]
        last_out = out[:, -1, :]

        # pass the output through the dropout layer
        # no changes in shape
        last_out = self.dropout_layer(last_out)

        # pass the tensors through the classification layer
        # shape logits: [batch_size, num_classes]
        logits = self.fc_layer(last_out)

        return logits


class CNNLSTM(nn.Module):

    def __init__(self, num_features: int, filters: List[int], hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        """
        CNN-LSTM model with the following architecture:
        (1) 1D convolutional layer
            - ReLU activation function
            - Max pooling
        (2) 1D convolutional layer
            - ReLU activation function
            - Max pooling
        (3) LSTM layer
        (4) Dropout layer
        (5) Fully connected layer

        The model assumes the inputs for the CNN of shapes [batch_size, n_channels, n_timesteps].

        :param num_features: number of features in each timestep (input size). This equates to the number of
                             sensor channels passed.
        :param filters: number of filters to be used for the two convolutional layers (i.e. [32, 64]) in order.
        :param hidden_size: number of hidden units in the LSTM cell
        :param num_layers: number of stacked lstm layers
        :param num_classes: number of output classes for classification
        :param dropout: dropout rate should be between [0.0 and 1.0]
        """
        # call to super class
        super().__init__()

        # init class variables
        self.num_features = num_features
        self.filters = filters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # model architecture
        # (1) first convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=filters[0],
            kernel_size=3
        )

        # ReLU activation function
        self.relu = nn.ReLU()

        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # (2) second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=filters[0],
            out_channels=filters[1],
            kernel_size=3)

        # (3) LSTM layer
        self.lstm = nn.LSTM(
            input_size=filters[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # (4) dropout layer
        self.dropout_layer = nn.Dropout(p=dropout)

        # (5) fully connected layer
        self.fc_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        forward pass
        :param x: input batch of size [batch_size, time_steps, sequence_length, num_channels]
        :return: output of shape [batch_size, num_classes]
        """
        # remove redundant dimension 'time_steps' since the CNN-LSTM loads a full window and time_steps = 1
        # shape: [batch_size, sequence_length, num_channels]
        x = x.squeeze(1)

        # reshape tensor to shape [batch_size, num_channels, sequence_length]
        x = x.transpose(1,2)

        # pass the inputs through the first convolutional layer, ReLU, and max pooling
        # output shape: [batch_size, num_filters1, L_conv1]
        x = self.pool(self.relu(self.conv1(x)))

        # pass the inputs through the second convolutional layer, ReLU, and max pooling
        # output shape: [batch_size, num_filters2, L_conv2]
        x = self.pool(self.relu(self.conv2(x)))

        # reshape tensor to shape [batch_size, L_conv, num_filters] for the LSTM input
        x = x.transpose(1,2)

        # pass though the LSTM cell
        # output shape: [batch_size, time_step, hidden_size]
        out, _ = self.lstm(x)

        # obtain the last output (many-to-one classification)
        # shape last_out: [batch_size, hidden_size]
        last_out = out[:, -1, :]

        # pass the output through the dropout layer
        # no changes in shape
        last_out = self.dropout_layer(last_out)

        # pass the tensors through the classification layer
        # shape logits: [batch_size, num_classes]
        logits = self.fc_layer(last_out)

        return logits


class CNNLSTM2d(nn.Module):

    def __init__(self, num_features: int, seq_len: int, filters: List[int], kernel_size_conv: List[Tuple[int, int]],
                 kernel_size_pool: List[Tuple[int, int]], stride_pool: List[Tuple[int, int]],
                 stride_conv: List[Tuple[int, int]], hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        """
        2D CNN-LSTM model with the following architecture:
        (1) 2D convolutional layer
            - ReLU activation function
            - Max pooling
        (2) 2D convolutional layer
            - ReLU activation function
            - Max pooling
        (3) LSTM layer
        (4) Dropout layer
        (5) Fully connected layer

        The model assumes the inputs for the 2D CNN-LSTM of shapes [batch_size, time_steps, sequence_length, num_channels].

        :param num_features: number of sensor channels.
        :param seq_len: the size of the window to window the sample into sub-sequences
        :param filters: Number of filters for the first and second convolutional layers, respectively.
        :param kernel_size_conv: List with the tuples corresponding to the kernel size for both convolutional layers, in order.
        :param kernel_size_pool: List with the tuples corresponding to the kernel size for both pooling layers, in order.
        :param stride_pool: List with the tuples corresponding to the stride for both pooling layers, in order.
        :param stride_conv: List with the tuples corresponding to the stride for both convolutional layers, in order.
        :param hidden_size: Number of hidden units in the LSTM cell
        :param num_layers: Number of stacked lstm layers
        :param num_classes: Number of output classes for classification
        :param dropout: Dropout rate should be between [0.0 and 1.0]
        """
        # call to super class
        super().__init__()

        # init class variables
        self.num_features = num_features
        self.seq_len = seq_len
        self.filters = filters
        self.kernel_size_conv = kernel_size_conv
        self.stride_conv = stride_conv
        self.kernel_size_pool = kernel_size_pool
        self.stride_pool = stride_pool
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # model architecture
        # (1) 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=filters[0], kernel_size=kernel_size_conv[0], stride=stride_conv[0])

        # ReLU activation function
        self.relu = nn.ReLU()

        # Max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_size_pool[0], stride=stride_pool[0])

        # (2) 2D convolutional layer
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_size_conv[1], stride=stride_conv[1])

        # Max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=kernel_size_pool[1], stride=stride_pool[1])

        # compute CNN output size dynamically
        cnn_output_size = self._get_cnn_output_size(num_features=num_features, seq_len=seq_len)

        # (3) LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_output_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )

        # (4) dropout layer
        self.dropout_layer = nn.Dropout(p=dropout)

        # (5) fully connected layer
        self.fc_layer = nn.Linear(hidden_size, num_classes)


    def _get_cnn_output_size(self, num_features: int, seq_len: int) -> int:
        """
        Computes the total number of features produced by the 2D CNN, given an input of shape:
        [batch_size, 1 (num_channels), num_features (height), seq_len (width)].
        This gives the input_size to the LSTM.

        :param num_features: number of sensor channels
        :param seq_len: the size of the window to window the sample into sub-sequences
        :return: the output size of the CNN (input size to the LSTM)
        """

        # disable gradients
        with torch.no_grad():

            # create dummy tensor that mimics a CNN input of shape: [batch, num_channels, num_features (H), seq_len (W)]
            dummy_tensor = torch.zeros(1, 1, num_features, seq_len)

            # Pass the dummy tensor through the first convolutional, relu, and pooling layers
            out = self.pool1(self.relu(self.conv1(dummy_tensor)))

            # Pass the dummy tensor through the second convolutional, relu, and pooling layers
            # output after the second conv layer: [1, f2, H_out, W_out]
            out = self.pool2(self.relu(self.conv2(out)))

            # calculate the total number of elements in the tensor - will be the input size to the LSTM
            # f2 * H_out * W_out
            lstm_input_size = out.numel()

        return lstm_input_size


    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        forward pass
        :param x: input batch of size [batch_size, timesteps, sequence_length, num_sensor_channels]
        :return: output of shape [batch_size, num_classes]
        """

        # get tensor dimensions
        batch_size, timesteps, seq_len, num_sensor_channels = x.shape

        # permute the last two dimensions to allow for convolving along the time dimension (transposing the images)
        # [batch_size, timesteps, num_sensor_channels, sequence_length]
        x = x.permute(0, 1, 3, 2)

        # merge batch and timesteps dims for CNN: [batch_size * num_timesteps, channels_in = 1, num_sensor_channels, sequence_length]
        x = x.reshape(batch_size * timesteps, 1, num_sensor_channels, seq_len)

        # pass the data through the first convolutional layer, ReLU, and max pooling
        # output shape: [batch_size, f1, height, width]
        x = self.pool1(self.relu(self.conv1(x)))

        # pass the data through the second convolutional layer, ReLU, and max pooling
        # output shape: [batch_size, f2, height', width']
        x = self.pool2(self.relu(self.conv2(x)))

        # flatten num_filters * Hout * Wout
        x = x.reshape(batch_size * timesteps, -1)

        # unmerge batch size and timesteps back to the LSTM input: [batch_size, timesteps, f2 * Hout * Wout]
        x = x.reshape(batch_size, timesteps, -1)

        # pass though the LSTM cell
        # output shape: [batch_size, time_step, hidden_size]
        out, _ = self.lstm(x)

        # obtain the last output (many-to-one classification)
        # shape last_out: [batch_size, hidden_size]
        last_out = out[:, -1, :]

        # pass the output through the dropout layer
        # no changes in shape
        last_out = self.dropout_layer(last_out)

        # pass the tensors through the classification layer
        # shape logits: [batch_size, num_classes]
        logits = self.fc_layer(last_out)

        return logits
