import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import load_stock_data, prepare_data, get_dataloaders
from models import LSTMModel, GRUModel
from train import train_model
from utils import compare_models
from visualize import plot_predictions, save_metrics_report

def main():
    # Hyperparameters
    ticker = 'AMZN'
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=10*365)).strftime('%Y-%m-%d') # 10 years of data
    
    seq_length = 60
    batch_size = 32
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    epochs = 20
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Downloading data for {ticker}...")
    df = load_stock_data(ticker, start_date, end_date)
    
    if df.empty:
        print("Failed to download data. Exiting.")
        return
        
    print(f"Downloaded {len(df)} rows.")

    # 2. Prepare Data
    train_dataset, val_dataset, test_dataset, scaler = prepare_data(df, sequence_length=seq_length)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

    # 3. Initialize Models and Loss
    models = {
        'LSTM': LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1),
        'GRU': GRUModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1),
    }
    criterion = nn.MSELoss()

    # 4. Train Models
    for name, model in models.items():
        print(f"Starting training for {name}...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        models[name] = train_model(model, train_loader, val_loader, criterion, optimizer, epochs,
                                   device=device, checkpoint_path=f'../models/best_{name.lower()}.pth')

    # 5. Evaluate and compare on the same test set
    results = compare_models(models, test_loader, criterion, scaler, device=device)

    # 6. Visualize and report
    plot_predictions(results, ticker)
    save_metrics_report(results, ticker)


if __name__ == '__main__':
    main()
