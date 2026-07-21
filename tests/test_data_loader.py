import numpy as np
import pandas as pd
import pytest

from data_loader import StockDataset, prepare_data


def test_stock_dataset_len_and_shapes():
    data = np.arange(20, dtype=np.float32).reshape(-1, 1)
    seq_length = 5
    ds = StockDataset(data, seq_length)

    assert len(ds) == len(data) - seq_length

    x, y = ds[0]
    assert x.shape == (seq_length, 1)
    assert y.shape == (1,)
    assert torch_allclose(x[:, 0], data[0:seq_length, 0])
    assert y[0] == data[seq_length, 0]


def torch_allclose(t, arr):
    return np.allclose(t.numpy(), arr)


@pytest.fixture
def fake_df():
    close = np.linspace(100, 200, num=100)
    return pd.DataFrame({"Close": close})


def test_prepare_data_split_sizes(fake_df):
    train_ds, val_ds, test_ds, _ = prepare_data(fake_df, sequence_length=5, train_split=0.8, val_split=0.1)

    n = len(fake_df)
    expected_test_idx = int(n * 0.8)
    expected_val_idx = int(n * (0.8 - 0.1))

    assert len(train_ds.data) == expected_val_idx
    assert len(val_ds.data) == expected_test_idx - expected_val_idx
    assert len(test_ds.data) == n - expected_test_idx


def test_prepare_data_scaler_fit_only_on_train(fake_df):
    _, _, _, scaler = prepare_data(fake_df, sequence_length=5, train_split=0.8, val_split=0.1)

    close = fake_df["Close"].values
    val_idx = int(len(close) * (0.8 - 0.1))
    train_max = close[:val_idx].max()
    train_min = close[:val_idx].min()

    # If look-ahead bias were present, scaler would be fit on the full series
    # (max=200), not just the train slice.
    assert scaler.data_max_[0] == pytest.approx(train_max)
    assert scaler.data_min_[0] == pytest.approx(train_min)
    # fake_df is monotonically increasing, so the full-series max is higher
    # than the train-only max whenever look-ahead bias is correctly avoided.
    assert scaler.data_max_[0] < close.max()
