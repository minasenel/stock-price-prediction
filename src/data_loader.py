import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def load_stock_data(ticker, start_date, end_date):
    """Fetches stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_data(df, sequence_length=60, train_split=0.8, val_split=0.1):
    """
    Prepares the data by scaling and creating chronological train/val/test datasets.
    The validation slice is carved out of the end of the training portion, so the
    effective split is (train_split - val_split) / val_split / (1 - train_split).
    """
    # Use only the 'Close' prices
    close_prices = df['Close'].values.reshape(-1, 1)

    # Chronological split on raw values, before scaling
    test_idx = int(len(close_prices) * train_split)
    val_idx = int(len(close_prices) * (train_split - val_split))
    train_raw = close_prices[:val_idx]
    val_raw = close_prices[val_idx:test_idx]
    test_raw = close_prices[test_idx:]

    # Scale the data: fit only on train to avoid look-ahead bias, transform val/test with it
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    # Create datasets
    train_dataset = StockDataset(train_data, sequence_length)
    val_dataset = StockDataset(val_data, sequence_length)
    test_dataset = StockDataset(test_data, sequence_length)

    return train_dataset, val_dataset, test_dataset, scaler

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader