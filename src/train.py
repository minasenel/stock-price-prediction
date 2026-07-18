import os
import torch

def validate_model(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    return val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs,
                device='cpu', patience=5, checkpoint_path='../models/best_model.pth'):
    model.to(device)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, criterion, device=device)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)')
                break

    # Restore the best checkpoint before returning
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')
    return predictions, actuals
