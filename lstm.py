import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Output layer
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # print(f"Input x shape: {x.shape}")  # Debug: Print input shape

        lstm_output, (hn, cn) = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        # print(f"LSTM output shape: {lstm_output.shape}")  # Debug: Print LSTM output shape
        # print(f"Hidden state shape: {hn.shape}")  # Debug: Print hidden state shape
        # print(f"Cell state shape: {cn.shape}")  # Debug: Print cell state shape

        # Output layer
        output = self.output_fc(lstm_output)  # (batch_size, seq_len, output_dim)
        # print(f"Output shape: {output.shape}")  # Debug: Print final output shape

        return output, lstm_output  # Return output and LSTM outputs as features
