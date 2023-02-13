import torch
import torch.nn as nn
from wire.layers.timestamp import CyclicTimestampEncoding
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

        self.timestamp_encoding = CyclicTimestampEncoding()

        # Wavelet transform layers
        self.wv_transform_open = WaveletTransform(wavelet_type, mode, padding)
        self.wv_transform_high = WaveletTransform(wavelet_type, mode, padding)
        self.wv_transform_low = WaveletTransform(wavelet_type, mode, padding)
        self.wv_transform_close = WaveletTransform(wavelet_type, mode, padding)
        #self.wv_transform_volume = WaveletTransform(wavelet_type, mode, padding)

        open_shape = self.wv_transform_open.get_output_dim(input_dim)
        high_shape = self.wv_transform_high.get_output_dim(input_dim)
        low_shape = self.wv_transform_low.get_output_dim(input_dim)
        close_shape = self.wv_transform_close.get_output_dim(input_dim)
        #volume_shape = self.wv_transform_volume.get_output_dim(input_dim)

        output_shape = self.timestamp_encoding.get_output_dim(input_dim - steps_ahead) + \
         open_shape + high_shape + low_shape + close_shape # + volume_shape

        # Fully connected layers
        self.fc0 = nn.Linear(output_shape, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, steps_ahead)

    def batch_to_tensor(self, batch_data) -> torch.Tensor:

        x_timestamp = batch_data["time"]
        x_open = torch.tensor(batch_data["open"].values, dtype=torch.float32)
        x_high = torch.tensor(batch_data["high"].values, dtype=torch.float32)
        x_low = torch.tensor(batch_data["low"].values, dtype=torch.float32)
        x_close = torch.tensor(batch_data["close"].values, dtype=torch.float32)
        # x_volume = torch.tensor(batch_data["volume"].values, dtype=torch.float32)

        # Apply the wavelet transforms
        x_timestamp = self.timestamp_encoding(x_timestamp)
        x_open = self.wv_transform_open(x_open)
        x_high = self.wv_transform_high(x_high)
        x_low = self.wv_transform_low(x_low)
        x_close = self.wv_transform_close(x_close)
        # x_volume = self.wv_transform_volume(x_volume)

        # x = torch.cat((x_timestamp, x_open, x_high, x_low, x_close, x_volume), dim=0)
        x = torch.cat((x_timestamp, x_open, x_high, x_low, x_close), dim=0)

        return x

    def forward(self, x):

        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x

