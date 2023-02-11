from wire.dataset import get_univariate_dataset, train_test_split
from wire.topologies.fc import TimeSeriesModel
from wire.train import train_model

from sklearn.model_selection import ParameterGrid
import torch.nn as nn
import torch.optim as optim


def train_with_grid_params(params):

    data = get_univariate_dataset(column='close', limit=10000)
    train_data, val_data, test_data = train_test_split(data)

    steps_ahead = 1
    batch_size = 64

    model = TimeSeriesModel(batch_size, params['hidden_dim'], steps_ahead, params['wavelet_type'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    run_id, test_loss = train_model(
        model,
        criterion,
        optimizer,
        train_data,
        val_data,
        test_data,
        batch_size,
        steps_ahead,
        num_epochs,
        params['base_dir']
    )

    with open(f"{params['base_dir']}/params", "a") as f:
        f.write(run_id + '|' + str(test_loss) + '|' + str(params) + '\n')


def grid_search():

    param_grid = [
        {
            'hidden_dim': [64, 128, 256, 512],
            'wavelet_type': ['db4', 'db8', 'haar'],
            'base_dir': ['checkpoints/fc_univ_1layer', ]}
    ]

    for grid in ParameterGrid(param_grid):
        train_with_grid_params(grid)


if __name__ == '__main__':
    grid_search()
