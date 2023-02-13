import numpy as np
import pandas as pd
import torch
from typing import List


SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY


def cyclic_encoding(timestamp: str) -> np.ndarray:
    """
    Calculate the cyclic encoding of a datetime string.

    Args:
    - timestamp: A string representation of a datetime.
    - period: The period to use for the encoding.

    Returns:
    - The cyclic encoding of the datetime as a numpy array.
    """

    timestamp = pd.to_datetime(timestamp)

    seconds_sin_encoding = np.sin(2 * np.pi * timestamp.second / 60)
    seconds_cos_encoding = np.cos(2 * np.pi * timestamp.second / 60)

    minutes_sin_encoding = np.sin(2 * np.pi * timestamp.minute / 60)
    minutes_cos_encoding = np.cos(2 * np.pi * timestamp.minute / 60)

    hours_sin_encoding = np.sin(2 * np.pi * timestamp.hour / 24)
    hours_cos_encoding = np.cos(2 * np.pi * timestamp.hour / 24)

    days_sin_encoding = np.sin(2 * np.pi * timestamp.day_of_week / 7)
    days_cos_encoding = np.cos(2 * np.pi * timestamp.day_of_week / 7)

    encoding = np.stack(
        [seconds_sin_encoding,
         seconds_cos_encoding,
         minutes_sin_encoding,
         minutes_cos_encoding,
         hours_sin_encoding,
         hours_cos_encoding,
         days_sin_encoding,
         days_cos_encoding],
        axis=0,
    )

    return encoding


class CyclicTimestampEncoding(torch.nn.Module):

    def get_output_dim(self, input_shape) -> int:
        return 8 * input_shape

    def forward(self, timestamps: List[str]) -> torch.Tensor:
        """
        Calculate the cyclic encoding for a batch of timestamps.

        Args:
        - timestamps: A tensor of datetime strings.

        Returns:
        - The cyclic encoding of the timestamps as a PyTorch tensor.
        """

        encoded = np.stack([cyclic_encoding(ts) for ts in timestamps], axis=0)

        return torch.tensor(encoded.flatten(), dtype=torch.float32)
