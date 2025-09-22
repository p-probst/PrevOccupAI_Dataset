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
# ------------------------------------------------------------------------------------------------------------------- #
# public classes
# ------------------------------------------------------------------------------------------------------------------- #
class HARLstm(nn.Module):

    def __init__(self, num_features: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        """
        simple lstm model consisting of:
        (1) lstm layer
        (2) dropout layer
        (3) fully connected layer
        The model assumes inputs of shape [batch_size, time_steps, num_features], where num features equates to the
        number of channels (i.e. sensor channels).
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # model architecture
        # (1) LSTM layer
        self.lstm = nn.LSTM(
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

        # pass the inputs through the LSTM
        # shape out: [batch_size, time_steps, hidden_size]
        out, _ = self.lstm(x)

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

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #