import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_heads, num_layers):
        super(SimpleTransformer, self).__init__()
        self.model_dim = model_dim

        # Input embedding
        self.input_fc = nn.Linear(input_dim, model_dim)

        # Positional encoding (optional for small sequences)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))  # Max seq_len = 1000

        # Transformer Encoder
        #encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])


        # Output layer
        self.output_fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # print("Input x shape:", x.shape)  # Debug: Input shape

        # Embed input
        x = self.input_fc(x)  # (batch_size, seq_len, model_dim)
        # print("After input_fc, x shape:", x.shape)  # Debug: After input_fc

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        # print("After adding positional encoding, x shape:", x.shape)  # Debug: After positional encoding

        # Transformer expects input of shape (seq_len, batch_size, model_dim)
        x = x.permute(1, 0, 2)
        # print("After permute, x shape:", x.shape)  # Debug: After permute

        features = []
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            features.append(x)
            # print(f"After transformer layer {i}, x shape:", x.shape)  # Debug: After each transformer layer

        # Back to (batch_size, seq_len, model_dim)
        x = x.permute(1, 0, 2)
        # print("After final permute, x shape:", x.shape)  # Debug: After final permute

        # Output layer
        output = self.output_fc(x)  # (batch_size, seq_len, output_dim)
        # print("Output shape:", output.shape)  # Debug: Output shape

        # Return only the last layer's features as a tensor
        return output, x  # Return output and the last transformer layer's output as features
