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
            batch_tensor = torch.tensor(batch_data.values, dtype=torch.float32)

            # Forward pass
            outputs = model(batch_tensor[:-steps_ahead])
            loss = criterion(outputs, batch_tensor[-steps_ahead:])

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Tracking
            batch_losses.append(loss.item())

        train_losses.append(sum(batch_losses) / len(batch_losses))

        # Evaluate the model on the validation data
        # Tumbling time window for efficiency since this is no_grad
        with torch.no_grad():
            # Compute val_loss in mini-batches
            val_loss = 0
            for batch in range(num_val_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                batch_data = val_data[start:end]

                batch_tensor = torch.tensor(batch_data.values, dtype=torch.float32)

                bd_in = batch_tensor[:-steps_ahead]
                bd_out = batch_tensor[-steps_ahead:]

                outputs = model(bd_in)
                val_loss += criterion(outputs, bd_out)

            # Tracking
            val_losses.append(val_loss.item())

            # Save the model if it has improved
            if val_loss.item() < best_val_loss:
                best_val_loss = loss.item()
                save_checkpoint(model, epoch, f"{training_dir}/best_val_loss_model.pth")

        if (epoch + 1) % 5 == 0:
            # Pad epoch with leading zeros
            save_checkpoint(model, epoch, f"{training_dir}/epoch_{str(epoch).zfill(3)}.pth")

            print(f"Epoch [{epoch+1} / {num_epochs}], " \
                f" Loss: {loss.item()} | " \
                f" Val Loss: {val_loss.item()}" \
                f" Best Val Loss: {best_val_loss}"
            )


    # Evaluate the model on the test data, sliding time window - we want accuracy!
    with torch.no_grad():

        best_model, _ = load_checkpoint(model, f"{training_dir}/best_val_loss_model.pth")

        test_loss = 0

        actual = []
        predictions = []

        for batch_idx in range(len(test_data) - batch_size):

            batch_data = test_data[batch_idx:batch_idx+batch_size]

            batch_tensor = torch.tensor(batch_data.values, dtype=torch.float32)

            model_inputs = batch_tensor[:-steps_ahead]
            actual_outputs = batch_tensor[-steps_ahead:]
            predicted_outputs = best_model(model_inputs)

            test_loss += criterion(predicted_outputs, actual_outputs)

            actual.extend(list(actual_outputs.detach().numpy().flatten()))
            predictions.extend(list(predicted_outputs.detach().numpy().flatten()))

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


def train_multiv_model(model, criterion, optimizer, train_data, val_data, test_data, batch_size, steps_ahead, num_epochs, base_dir):

    # Train the model
    unique_id = lexographical_id()
    training_dir = f"{base_dir}/{unique_id}"
    best_val_loss = float('inf')
    num_train_batches = train_data.shape[0] // batch_size
    num_val_batches = val_data.shape[0] // batch_size

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
            batch_tensor = model.batch_to_tensor(batch_data[:-steps_ahead])
            outputs = model(batch_tensor)

            # Compute loss
            expected = torch.tensor(batch_data['close'][-steps_ahead:].values, dtype=torch.float32)
            loss = criterion(outputs, expected)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Tracking
            batch_losses.append(loss.item())

        train_losses.append(sum(batch_losses) / len(batch_losses))

        # Evaluate the model on the validation data
        # Tumbling time window for efficiency since this is no_grad
        with torch.no_grad():
            # Compute val_loss in mini-batches
            val_loss = 0
            for batch in range(num_val_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                batch_data = val_data[start:end]

                batch_tensor = model.batch_to_tensor(batch_data[:-steps_ahead])
                outputs = model(batch_tensor)

                expected = torch.tensor(batch_data['close'][-steps_ahead:].values, dtype=torch.float32)
                val_loss += criterion(outputs, expected)

            # Tracking
            val_losses.append(val_loss.item())

            # Save the model if it has improved
            if val_loss.item() < best_val_loss:
                best_val_loss = loss.item()
                save_checkpoint(model, epoch, f"{training_dir}/best_val_loss_model.pth")

        if (epoch + 1) % 5 == 0:
            # Pad epoch with leading zeros
            save_checkpoint(model, epoch, f"{training_dir}/epoch_{str(epoch).zfill(3)}.pth")

            print(f"Epoch [{epoch+1} / {num_epochs}], " \
                f" Loss: {loss.item()} | " \
                f" Val Loss: {val_loss.item()}" \
                f" Best Val Loss: {best_val_loss}"
            )


    # Evaluate the model on the test data, sliding time window - we want accuracy!
    with torch.no_grad():

        best_model, _ = load_checkpoint(model, f"{training_dir}/best_val_loss_model.pth")

        test_loss = 0

        actual = []
        predictions = []

        for batch_idx in range(len(test_data) - batch_size):

            batch_data = test_data[batch_idx:batch_idx+batch_size]
            model_inputs = batch_data[:-steps_ahead]
            input_tensors = best_model.batch_to_tensor(model_inputs)

            actual_outputs = torch.tensor(batch_data['close'][-steps_ahead:].values, dtype=torch.float32)
            predicted_outputs = best_model(input_tensors)

            test_loss += criterion(predicted_outputs, actual_outputs)

            actual.extend(list(actual_outputs.detach().numpy().flatten()))
            predictions.extend(list(predicted_outputs.detach().numpy().flatten()))

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
