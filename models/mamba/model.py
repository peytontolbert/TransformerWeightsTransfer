import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace as Namespace  # Changed import
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight



class MambaLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MambaLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear transformations for input and hidden state
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)  # Added LayerNorm

    def forward(self, x):
        # Mamba-style transformation: input to hidden to output
        h = F.relu(self.input_layer(x))
        h = F.relu(self.hidden_layer(h))
        h = self.norm(h)  # Apply normalization
        out = self.output_layer(h)
        return out


class LlamaMambaModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super(LlamaMambaModel, self).__init__()
        self.params = params
        self.input_dim = params.dim
        self.hidden_dim = params.dim
        self.num_layers = params.n_layers
        self.vocab_size = params.vocab_size
        self.pad_id = 0  # Ensure pad_id is set correctly

        # Recurrent Layer (GRU or LSTM)
        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)


        # Gated Attention Mechanism
        self.attention = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context_vector = nn.Linear(self.hidden_dim, 1, bias=False)

        # Output layer
        self.output_fc = nn.Linear(self.hidden_dim, self.vocab_size)




        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # Define embedding layer to convert input tokens into dense representations
        self.embedding = nn.Embedding(self.vocab_size, self.input_dim)
        
        # Stack Mamba layers
        self.mamba_layers = nn.ModuleList([MambaLayer(self.input_dim, self.hidden_dim) for _ in range(self.num_layers)])

        # Final output layer to map hidden states back to vocabulary size
        self.output_layer = nn.Linear(params.dim, self.vocab_size, bias=False)
        
        # Added params attribute
        self.params = Namespace(
            max_seq_len=params.max_seq_len,
            max_batch_size=params.max_batch_size
        )

    @torch.inference_mode()
    def forward(self, x, target_ids=None):
        # Convert input indices to embeddings
        x = self.embedding(x)
        
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
        logits = self.output_layer(combined)  # (batch_size, seq_len, vocab_size)
        assert logits.size(-1) == self.vocab_size, f"Logits dim {logits.size(-1)} does not match vocab_size {self.vocab_size}."

        return logits, attention_weights  # Return output and attention weights as features







"""
        # Convert input indices to embeddings
        x = self.embedding(x)

        # Pass through each Mamba layer
        for layer in self.mamba_layers:
            x = layer(x)

        # Project back to vocabulary size
        logits = self.output_layer(x)
        return logits

    def generate(self, input_sequence, max_length=50):
        # ... existing generate method unchanged ...
        pass  # Placeholder for existing generate method
    """