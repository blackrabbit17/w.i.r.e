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

    train_start_index = 0
    train_end_index = int(train_ratio * num_data)

    val_start_index = int(train_ratio * num_data)
    val_end_index = int((train_ratio + val_ratio) * num_data)

    test_start_index = int((train_ratio + val_ratio) * num_data)
    test_end_index = num_data

    print(
        f'Dataset shape: {data.shape}\n',
        f'Train:         {train_start_index}:{train_end_index}\n',
        f'Val:           {val_start_index}:{val_end_index}\n',
        f'Test:          {test_start_index}:{num_data}\n'
    )

    train_data = data[train_start_index:train_end_index]
    val_data = data[val_start_index:val_end_index]
    test_data = data[test_start_index:num_data]

    return (
        torch.tensor(train_data.values, dtype=torch.float32),
        torch.tensor(val_data.values, dtype=torch.float32),
        torch.tensor(test_data.values, dtype=torch.float32)
    )
