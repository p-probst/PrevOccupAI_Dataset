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

# internal imports
from constants import LSTM, GRU

# ------------------------------------------------------------------------------------------------------------------- #
#  constants
# ------------------------------------------------------------------------------------------------------------------- #

SUPPORTED_DL_MODELS = [LSTM, GRU]

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
        The model assumes inputs of shape [batch_size, time_steps, num_features], where num features equates to the
        number of channels (i.e. sensor channels).
        :param model_type:
        :param num_features: number of features in each timestep (input size). This equates to the number of
                             sensor channels passed.
        :param hidden_size: number of hidden units in the LSTM cell
        :param num_layers: number of stacked lstm layers
        :param num_classes: number of output classes for classification
        :param dropout: dropout rate should be between [0.0 and 1.0]
        """

        # call to super class
        super().__init__()

        if self.model_type not in SUPPORTED_DL_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'lstm' or 'gru'.")

        # init class variables
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # model architecture
        # check if it is a LSTM layer or a GRU layer
        rnn_type = nn.LSTM if self.model_type == LSTM else nn.GRU
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

        # pass the tensors through the classification later
        # shape logits: [batch_size, num_classes]
        logits = self.fc_layer(last_out)

        return logits

