from .model_io import save_checkpoint
from .layers.wavelet import WaveletTransform
from .topologies.fc import FullyConnectedNet

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import uuid


# Load the data
data = pd.read_csv(
        "data/ES_continuous_adjusted_1min.txt",
        engine="pyarrow",
        header=None,
        names=["time", "open", "high", "low", "close", "volume"],
    )

inputs = torch.tensor(data['close'], dtype=torch.float32)
inputs = inputs.unsqueeze(1)

# Define the model

wavelet_layer = WaveletTransform()
model = FullyConnectedNet(input_size=1, hidden_size=32, num_classes=16, wavelet_layer=WaveletTransform)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

unqiue_id = uuid.uuid4().hex[0:8]

# Train the model
best_loss = float('inf')

for epoch in range(100):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, inputs)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save the model if it has improved
    if loss.item() < best_loss:
        best_loss = loss.item()
        save_checkpoint(model, epoch, "checkpoints/model.pth")

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item()}")

print("Training finished!")

