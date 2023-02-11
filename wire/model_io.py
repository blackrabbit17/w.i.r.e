import os
import torch


def save_checkpoint(model, epoch, path):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(model, path):

    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    return model, state['epoch']
