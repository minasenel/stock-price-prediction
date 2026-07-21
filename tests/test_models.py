import torch

from models import LSTMModel, GRUModel


def test_lstm_model_forward_shape():
    model = LSTMModel(input_size=1, hidden_size=8, num_layers=1, output_size=1)
    x = torch.randn(4, 10, 1)
    out = model(x)
    assert out.shape == (4, 1)


def test_gru_model_forward_shape():
    model = GRUModel(input_size=1, hidden_size=8, num_layers=1, output_size=1)
    x = torch.randn(4, 10, 1)
    out = model(x)
    assert out.shape == (4, 1)
