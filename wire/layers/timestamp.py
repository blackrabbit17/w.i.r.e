import numpy as np
import pandas as pd
import torch
from typing import List

"""
TODO: IMPLEMENTATION NOT COMPLETE
"""


def cyclic_encoding(timestamp: str, period: int) -> np.ndarray:
    """
    Calculate the cyclic encoding of a datetime string.

    Args:
    - timestamp: A string representation of a datetime.
    - period: The period to use for the encoding.

    Returns:
    - The cyclic encoding of the datetime as a numpy array.
    """

    timestamp = pd.to_datetime(timestamp)

    seconds = np.int64(timestamp.timestamp()) % period

    sin_encoding = np.sin(2 * np.pi * seconds / period)
    cos_encoding = np.cos(2 * np.pi * seconds / period)

    return np.stack([sin_encoding, cos_encoding], axis=1)


class CyclicEncoding(torch.nn.Module):
    def __init__(self, period: int):
        """
        Initialize the cyclic encoding layer.

        Args:
        - period: The period to use for the encoding.
        """
        super(CyclicEncoding, self).__init__()
        self.period = period

    def forward(self, timestamps: List[str]) -> torch.Tensor:
        """
        Calculate the cyclic encoding for a batch of timestamps.

        Args:
        - timestamps: A tensor of datetime strings.

        Returns:
        - The cyclic encoding of the timestamps as a PyTorch tensor.
        """

        encoded = np.stack([cyclic_encoding(ts, self.period) for ts in timestamps], axis=0)

        return torch.from_numpy(encoded).float()
