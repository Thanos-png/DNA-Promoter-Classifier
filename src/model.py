import torch
import torch.nn as nn
from torch import Tensor


class PromoterCNN(nn.Module):
    """
    Convolutional Neural Network for binary classification of DNA promoter sequences.
    Input shape: (batch_size, seq_len, 4)
    Output: raw logits for promoter classification
    """
    def __init__(self, sequence_length: int = 57) -> None:
        super().__init__()
        # 1D convolution expects (batch_size, in_channels, seq_len)
        self.conv1: nn.Conv1d = nn.Conv1d(in_channels=4,
                                        out_channels=32,
                                        kernel_size=5)
        # After conv1: length = sequence_length - (kernel_size - 1)
        conv_output_length: int = sequence_length - (5 - 1)
        self.pool: nn.MaxPool1d = nn.MaxPool1d(kernel_size=2)
        # After pool: length = conv_output_length // 2
        pooled_length: int = conv_output_length // 2
        self.flattened_size: int = 32 * pooled_length

        self.fc1: nn.Linear = nn.Linear(self.flattened_size, 64)
        self.dropout: nn.Dropout = nn.Dropout(p=0.5)
        self.fc2: nn.Linear = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, seq_len, 4)
        # Permute to (batch_size, in_channels, seq_len)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        # Flatten: (batch_size, flattened_size)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(sequence_length: int = 57) -> nn.Module:
    """
    Utility function to instantiate the PromoterCNN model.
    Args:
        sequence_length: Length of the one-hot encoded DNA sequences.
    Returns:
        An instance of PromoterCNN.
    """
    return PromoterCNN(sequence_length)
