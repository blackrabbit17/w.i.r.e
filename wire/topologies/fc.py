import torch
import torch.nn as nn
import torch.nn.functional as F

from wire.layers.wavelet import WaveletTransform


class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, steps_ahead, wavelet_type="db4", mode="zero", padding=True):
        super(TimeSeriesModel, self).__init__()

        # Wavelet transform layer
        self.wavelet_transform = WaveletTransform(wavelet_type, mode, padding)

        output_shape = self.wavelet_transform.get_output_dim(input_dim)

        # Fully connected layers
        self.fc0 = nn.Linear(output_shape, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, steps_ahead)

    def forward(self, x):
        # Apply the wavelet transform
        x = self.wavelet_transform(x)

        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x
