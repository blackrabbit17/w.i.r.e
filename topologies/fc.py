import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, wavelet_layer):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.wavelet = wavelet_layer()

    def forward(self, x):
        x = self.wavelet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
