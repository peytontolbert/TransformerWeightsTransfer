import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MambaLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear transformations for input and hidden state
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Mamba-style transformation: input to hidden to output
        h = F.relu(self.input_layer(x))
        h = F.relu(self.hidden_layer(h))
        out = self.output_layer(h)
        return out


class LlamaMambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, vocab_size):
        super(LlamaMambaModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Define embedding layer to convert input tokens into dense representations
        self.embedding = nn.Embedding(vocab_size, input_dim)
        
        # Stack Mamba layers
        self.mamba_layers = nn.ModuleList([MambaLayer(input_dim, hidden_dim) for _ in range(num_layers)])

        # Final output layer to map hidden states back to vocabulary size
        self.output_layer = nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        # Convert input indices to embeddings
        x = self.embedding(x)

        # Pass through each Mamba layer
        for layer in self.mamba_layers:
            x = layer(x)

        # Project back to vocabulary size
        logits = self.output_layer(x)
        return logits

    def generate(self, input_sequence, max_length=50):
        # Switch to evaluation mode for generation
        self.eval()
        generated_sequence = input_sequence
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)
        
        for _ in range(max_length):
            logits = self.forward(input_tensor)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()
            generated_sequence.append(next_token)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

            # Break if the model predicts an end-of-sequence token
            if next_token == self.vocab_size - 1:
                break
        return generated_sequence