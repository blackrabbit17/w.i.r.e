import torch


def save_checkpoint(model, epoch, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict()
    }
    torch.save(state, path)

