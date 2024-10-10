import torch
import torch.nn as nn


class LiquidS4(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, liquid_order=2, num_layers=1):
        super(LiquidS4, self).__init__()
        self.num_layers = num_layers  # {{ edit_4: Add num_layers parameter }}

        # Initialize multiple LiquidS4 layers with output_dim=state_dim for all but the last layer
        self.layers = nn.ModuleList([
            SingleLiquidS4Layer(state_dim, input_dim, state_dim, liquid_order)
            for _ in range(num_layers - 1)
        ])  # {{ edit_5: Initialize intermediate layers with output_dim=state_dim }}

        # Final layer maps to vocab_size
        self.final_layer = SingleLiquidS4Layer(state_dim, state_dim, output_dim, liquid_order)  # {{ edit_6: Add final output layer }}

        # {{ edit_modify_B_dimensions: Initialize B with shape [output_dim, input_dim] to match LLaMA's up_proj.weight ]]
        self.B = nn.Parameter(torch.randn(output_dim, input_dim))
        # {{ edit_end }}

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs, _ = layer(outputs)
        outputs, _ = self.final_layer(outputs)  # {{ edit_7: Pass through final output layer }}
        return outputs


# Define a single layer for modularity
class SingleLiquidS4Layer(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, liquid_order):
        super(SingleLiquidS4Layer, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.liquid_order = liquid_order

        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(input_dim, state_dim))
        self.C = nn.Parameter(torch.randn(state_dim, output_dim))
        self.liquid_kernel = nn.Parameter(torch.randn(state_dim, state_dim))
        
        # {{ edit_1: Add A_bias parameter }}
        self.A_bias = nn.Parameter(torch.zeros(state_dim))  # Initialize A_bias

        # Include LayerNorm
        self.layer_norm = nn.LayerNorm(state_dim)

        # Include activation function
        self.activation = nn.GELU()


        # Initialize parameters
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.liquid_kernel)
        # {{ edit_2: Initialize A_bias }}

    def forward(self, inputs):
        batch_size, seq_len, input_dim = inputs.shape
        hidden_state = torch.zeros(batch_size, self.state_dim).to(inputs.device)
        outputs = []
        hidden_states = []

        for t in range(seq_len):
            u_t = inputs[:, t, :]
            if t > 0:
                u_prev = inputs[:, t - 1, :]
                interaction = torch.matmul(hidden_state, self.liquid_kernel) * torch.matmul(u_prev, self.B)
            else:
                interaction = 0

            # {{ edit_3: Include A_bias in hidden state update }}
            new_hidden_state = torch.matmul(hidden_state, self.A) + self.A_bias + torch.matmul(u_t, self.B) + interaction
            # Apply LayerNorm and activation
            new_hidden_state = self.layer_norm(new_hidden_state)
            new_hidden_state = self.activation(new_hidden_state)
            hidden_state = hidden_state + new_hidden_state
            #hidden_state = torch.matmul(hidden_state, self.A) + torch.matmul(u_t, self.B) + interaction
            output = torch.matmul(hidden_state, self.C)
            outputs.append(output)
            hidden_states.append(hidden_state)

        outputs = torch.stack(outputs, dim=1)
        hidden_states = torch.stack(hidden_states, dim=1)

        return outputs, hidden_states


# ... existing example usage ...