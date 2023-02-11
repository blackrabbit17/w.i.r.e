from wire.eval import writeout_losses, writeout_graphs
from wire.model_io import save_checkpoint, load_checkpoint
from wire.utils.unique_id import lexographical_id

import torch
from tqdm import tqdm



def train_model(model, criterion, optimizer, train_data, val_data, test_data, batch_size, steps_ahead, num_epochs, base_dir):

    # Train the model
    unique_id = lexographical_id()
    training_dir = f"{base_dir}/{unique_id}"
    best_val_loss = float('inf')
    num_train_batches = train_data.shape[0] // batch_size
    num_val_batches = val_data.shape[0] // batch_size
    num_test_batches = test_data.shape[0] // batch_size

    # Tracking
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        batch_losses = []
        for batch in tqdm(range(num_train_batches)):
            # Clear the gradients
            optimizer.zero_grad()

            # Select the current batch of training data
            start = batch * batch_size
            end = (batch + 1) * batch_size
            batch_data = train_data[start:end]

            # Forward pass
            outputs = model(batch_data[:-steps_ahead])
            loss = criterion(outputs, batch_data[-steps_ahead:])

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Tracking
            batch_losses.append(loss.item())

        train_losses.append(sum(batch_losses) / len(batch_losses))

        # Evaluate the model on the validation data
        with torch.no_grad():
            # Compute val_loss in mini-batches
            val_loss = 0
            for batch in range(num_val_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                batch_data = val_data[start:end]
                outputs = model(batch_data[:-steps_ahead])
                val_loss += criterion(outputs, batch_data[-steps_ahead:])

            # Tracking
            val_losses.append(val_loss.item())

            # Save the model if it has improved
            if val_loss.item() < best_val_loss:
                best_val_loss = loss.item()
                save_checkpoint(model, epoch, f"{training_dir}/best_val_loss_model.pth")

        if (epoch + 1) % 10 == 0:
            # Pad epoch with leading zeros
            save_checkpoint(model, epoch, f"{training_dir}/epoch_{str(epoch).zfill(3)}.pth")

            print(f"Epoch [{epoch+1}/100], " \
                f" Loss: {loss.item()} | " \
                f" Val Loss: {val_loss.item()}" \
                f" Best Val Loss: {best_val_loss}"
            )


    # Evaluate the model on the test data
    with torch.no_grad():

        best_model, _ = load_checkpoint(model, f"{training_dir}/best_val_loss_model.pth")

        test_loss = 0

        actual = []
        predictions = []

        for batch in range(num_test_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            batch_data = test_data[start:end]
            outputs = best_model(batch_data[:-steps_ahead])
            test_loss += criterion(outputs, batch_data[-steps_ahead:])

            actual.append(batch_data[-steps_ahead:].numpy())
            predictions.append(outputs.numpy())

        print(f"Test Loss: {test_loss.item()}")

    writeout_losses(training_dir, train_losses, val_losses)
    writeout_graphs(
        training_dir,
        train_losses,
        val_losses,
        predictions=predictions,
        actual=actual
    )

    return unique_id, test_loss.item()
