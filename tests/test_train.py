import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMModel
from train import validate_model, train_model


def make_loader(num_samples=16, seq_length=5, batch_size=4):
    x = torch.randn(num_samples, seq_length, 1)
    y = torch.randn(num_samples, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_validate_model_returns_float_loss():
    model = LSTMModel(input_size=1, hidden_size=4, num_layers=1, output_size=1)
    loader = make_loader()
    criterion = nn.MSELoss()

    loss = validate_model(model, loader, criterion)
    assert isinstance(loss, float)


def test_train_model_runs_and_saves_checkpoint(tmp_path):
    model = LSTMModel(input_size=1, hidden_size=4, num_layers=1, output_size=1)
    train_loader = make_loader()
    val_loader = make_loader()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoint_path = str(tmp_path / "best_model.pth")

    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=2, patience=5, checkpoint_path=checkpoint_path,
    )

    assert trained_model is model
    assert (tmp_path / "best_model.pth").exists()
