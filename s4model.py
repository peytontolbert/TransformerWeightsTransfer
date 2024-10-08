import torch
import torch.nn as nn


class LiquidS4(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, liquid_order=2):
        super(LiquidS4, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.liquid_order = liquid_order  # The order of the liquid kernel

        # Matrices A, B, and C as described in the paper
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(input_dim, state_dim))
        self.C = nn.Parameter(torch.randn(state_dim, output_dim)) # State to output transition

        # Liquid Kernel parameters for input correlations
        self.liquid_kernel = nn.Parameter(torch.randn(state_dim, state_dim))



        # Initialize parameters using Xavier uniform initialization
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.liquid_kernel)


    def forward(self, inputs):
        # Assuming inputs is of shape [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = inputs.shape
        #print(f"Input shape: {inputs.shape}")

        # Initialize hidden state
        hidden_state = torch.zeros(batch_size, self.state_dim).to(inputs.device)  # Size should match A's dimension
        outputs = []
        hidden_states = []  # To store hidden states as features

        for t in range(seq_len):
            # Current input
            u_t = inputs[:, t, :]  # [batch_size, input_dim]
            #print(f"Time step {t}, input: {u_t}")

            # Liquid interaction term (second order input correlation)
            if t > 0:
                u_prev = inputs[:, t - 1, :]  # Previous input
                # Apply liquid kernel interaction term: element-wise product with previous input
                interaction = torch.matmul(hidden_state, self.liquid_kernel) * torch.matmul(u_prev, self.B)
                #print(f"Time step {t}, interaction: {interaction}")
            else:
                interaction = 0
            
            # Hidden state update: A x_{k-1} + B u_k + liquid interaction
            hidden_state = torch.matmul(hidden_state, self.A) + torch.matmul(u_t, self.B) + interaction
            #print(f"Time step {t}, hidden state: {hidden_state}")
            
            # Output computation: C x_k
            output = torch.matmul(hidden_state, self.C)  # Shape [batch_size, output_dim]
            #print(f"Time step {t}, output: {output}")
            outputs.append(output)
            hidden_states.append(hidden_state)  # Store hidden state

        # Stack outputs across time steps
        outputs = torch.stack(outputs, dim=1)  # Shape: [batch_size, seq_len, output_dim]
        hidden_states = torch.stack(hidden_states, dim=1)  # Shape: [batch_size, seq_len, state_dim]

        #print(f"Final outputs shape: {outputs.shape}")
        #print(f"Final hidden states shape: {hidden_states.shape}")

        return outputs, hidden_states  # Return outputs and hidden states as features
"""
# Example usage
input_dim = 10
state_dim = 20
output_dim = 5
seq_len = 100
batch_size = 32

model = LiquidS4(state_dim, input_dim, output_dim, liquid_order=2)
inputs = torch.randn(batch_size, seq_len, input_dim)  # Random input data
outputs, hidden_states = model(inputs)
print(outputs.shape)  # Should be [batch_size, seq_len, output_dim]
print(hidden_states.shape)  # Should be [batch_size, seq_len, state_dim]
"""