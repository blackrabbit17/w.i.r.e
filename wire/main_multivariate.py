from wire.dataset import get_multivariate_dataset, train_test_split
from wire.topologies.fc import TimeSeriesMultivariateModel
from wire.train import train_multiv_model

from sklearn.model_selection import ParameterGrid
import torch.nn as nn
import torch.optim as optim


def train_with_grid_params(params):

    data = get_multivariate_dataset(columns=['time', 'high', 'low', 'open', 'close', 'volume'], limit=100_000)
    train_data, val_data, test_data = train_test_split(data)

    steps_ahead = params['forecast']
    batch_size = params['batch_size']

    model = TimeSeriesMultivariateModel(batch_size, params['hidden_dim'], steps_ahead, params['wavelet_type'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50

    run_id, test_loss = train_multiv_model(
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
            'forecast': [1, ],
            'batch_size': [64],
            'hidden_dim': [256, 512],
            'wavelet_type': ['db4', 'db8', 'haar'],
            'base_dir': ['checkpoints/fc_multiv', ]
        }
    ]

    for grid in ParameterGrid(param_grid):
        train_with_grid_params(grid)

    # Sort the results
    lines = None
    with open(f"{grid['base_dir']}/params", "r") as f:
        lines = f.readlines()
        lines.sort(key=lambda x: float(x.split('|')[1]))

    with open(f"{grid['base_dir']}/params", "w") as f:
        f.writelines(lines)


if __name__ == '__main__':
    grid_search()
