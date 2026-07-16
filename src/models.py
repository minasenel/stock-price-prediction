import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, sequence_length, hidden_size]
        lstm_out, _ = self.lstm(x)

        # Only take the output from the final sequence timestep
        out = self.fc(lstm_out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through GRU layer
        # shape of gru_out: [batch_size, sequence_length, hidden_size]
        gru_out, _ = self.gru(x)

        # Only take the output from the final sequence timestep
        out = self.fc(gru_out[:, -1, :])
        return out
