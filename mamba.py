import torch
import torch.nn as nn

class MambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(MambaModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Recurrent Layer (GRU or LSTM)
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Gated Attention Mechanism
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

        # Output layer
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # print("Input x shape:", x.shape)  # Debug: Print input shape

        # Recurrent layer
        rnn_output, _ = self.rnn(x)  # (batch_size, seq_len, hidden_dim)
        # print("RNN output shape:", rnn_output.shape)  # Debug: Print RNN output shape

        # Attention mechanism
        energy = torch.tanh(self.attention(rnn_output))  # (batch_size, seq_len, hidden_dim)
        # print("Energy shape:", energy.shape)  # Debug: Print energy shape

        attention_weights = torch.softmax(self.context_vector(energy), dim=1)  # (batch_size, seq_len, 1)
        # print("Attention weights shape:", attention_weights.shape)  # Debug: Print attention weights shape

        # Weighted sum of rnn_output
        context = torch.sum(attention_weights * rnn_output, dim=1)  # (batch_size, hidden_dim)
        # print("Context shape:", context.shape)  # Debug: Print context shape

        # Expand context to match seq_len
        context = context.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch_size, seq_len, hidden_dim)
        # print("Expanded context shape:", context.shape)  # Debug: Print expanded context shape

        # Combine context with rnn_output
        combined = rnn_output + context  # (batch_size, seq_len, hidden_dim)
        # print("Combined shape:", combined.shape)  # Debug: Print combined shape

        # Output layer
        output = self.output_fc(combined)  # (batch_size, seq_len, output_dim)
        # print("Output shape:", output.shape)  # Debug: Print output shape

        return output, attention_weights  # Return output and attention weights as features
