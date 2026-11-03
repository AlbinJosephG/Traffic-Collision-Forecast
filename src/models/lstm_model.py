
import torch
import torch.nn as nn


class CollisionLSTM(nn.Module):
    """
    LSTM-based binary collision prediction model.

    Input:
        (batch_size, sequence_length, input_size)

    Output:
        (batch_size, 1) raw logits
    """

    def __init__(
        self,
        input_size=96,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):

        # LSTM output:
        # (batch_size, sequence_length, hidden_size)
        output, (hidden, cell) = self.lstm(x)

        # Last layer's final hidden state
        final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)

        logits = self.classifier(final_hidden)

        return logits.squeeze(1)
