import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletImplicitCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, wavelet_layer, wavelet='db1', mode='zero'):
        super(WaveletImplicitCNN, self).__init__()
        self.wavelet_layer = WaveletImplicitLayer(wavelet, mode)
        self.conv1 = nn.Conv1d(1, hidden_sizes[0], kernel_size=input_size)
        self.hidden_layers = nn.ModuleList([nn.Conv1d(hidden_sizes[i], hidden_sizes[i+1], kernel_size=1) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.wavelet_layer(x)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = torch.relu(x)
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = torch.relu(x)
        x = torch.mean(x, dim=(2,3))
        x = self.output_layer(x)
        return x