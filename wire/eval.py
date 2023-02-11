import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os


def writeout_losses(training_dir, train_losses, val_losses, warmup_discard=5):

    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    with open(f"{training_dir}/train_loss.csv", "w") as f:
        f.write("\n".join([str(x) for x in train_losses[warmup_discard:]]))

    with open(f"{training_dir}/val_loss.csv", "w") as f:
        f.write("\n".join([str(x) for x in val_losses[warmup_discard:]]))


def writeout_graphs(
    training_dir,
    train_losses,
    val_losses,
    warmup_discard=5,
    predictions=None,
    actual=None):

    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(15, 15))

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses[warmup_discard:], label="Training Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(val_losses[warmup_discard:], label="Validation Loss")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    if predictions is not None and actual is not None:
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(actual, label="Actual")
        ax3.plot(predictions, label="Predicted")
        ax3.set_title("Actual vs Predicted")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Value")
        ax3.legend()

    plt.savefig(f"{training_dir}/performance.png")
