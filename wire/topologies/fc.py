import torch
import torch.nn as nn
import torch.nn.functional as F

from wire.layers.wavelet import WaveletTransform


class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, steps_ahead, wavelet_type="db4", mode="zero", padding=True):
        super(TimeSeriesModel, self).__init__()

        self.wavelet_transform = WaveletTransform(wavelet_type, mode, padding)
        output_shape = self.wavelet_transform.get_output_dim(input_dim)

        self.fc0 = nn.Linear(output_shape, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, steps_ahead)

    @property
    def input_dim(self):
        return self.wavelet_transform.input_dim

    def forward(self, x):

        x = self.wavelet_transform(x)

        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x


class TimeSeriesMultivariateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, steps_ahead, wavelet_type="db4", mode="zero", padding=True):
        super(TimeSeriesMultivariateModel, self).__init__()

        # Wavelet transform layers
        self.wv_transform_open = WaveletTransform(wavelet_type, mode, padding)
        self.wv_transform_high = WaveletTransform(wavelet_type, mode, padding)
        self.wv_transform_low = WaveletTransform(wavelet_type, mode, padding)
        self.wv_transform_close = WaveletTransform(wavelet_type, mode, padding)
        self.wv_transform_volume = WaveletTransform(wavelet_type, mode, padding)

        open_shape = self.wv_transform_open.get_output_dim(input_dim)
        high_shape = self.wv_transform_high.get_output_dim(input_dim)
        low_shape = self.wv_transform_low.get_output_dim(input_dim)
        close_shape = self.wv_transform_close.get_output_dim(input_dim)
        volume_shape = self.wv_transform_volume.get_output_dim(input_dim)

        output_shape = open_shape + high_shape + low_shape + close_shape + volume_shape

        # Fully connected layers
        self.fc0 = nn.Linear(output_shape, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, steps_ahead)

    def forward(self, x_open, x_high, x_low, x_close, x_volume):

        # Apply the wavelet transforms
        x_open = self.wv_transform_open(x_open)
        x_high = self.wv_transform_high(x_high)
        x_low = self.wv_transform_low(x_low)
        x_close = self.wv_transform_close(x_close)
        x_volume = self.wv_transform_volume(x_volume)

        # Concat all the features together
        x = torch.cat((x_open, x_high, x_low, x_close, x_volume), dim=1)

        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x

