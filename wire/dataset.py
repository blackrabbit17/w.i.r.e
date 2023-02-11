import pandas as pd
import torch
from typing import Tuple

RAW_DATA = "data/ES_continuous_adjusted_1min.txt"


def get_univariate_dataset(column, limit=None) -> pd.Series:

    data = pd.read_csv(
        RAW_DATA,
        engine="pyarrow",
        header=None,
        names=["time", "open", "high", "low", "close", "volume"]
    )

    data = data[column]

    if limit is not None:
        data = data[:limit]

    return data

def train_test_split(
        data, train_ratio=0.8,
        val_ratio=0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    num_data = data.shape[0]
    train_data = data[:int(train_ratio * num_data)]
    val_data = data[int(train_ratio * num_data):int((train_ratio + val_ratio) * num_data)]
    test_data = data[int((train_ratio + val_ratio) * num_data):]

    return (
        torch.tensor(train_data.values, dtype=torch.float32),
        torch.tensor(val_data.values, dtype=torch.float32),
        torch.tensor(test_data.values, dtype=torch.float32)
    )
