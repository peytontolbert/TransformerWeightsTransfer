import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the LSSM class
class LSSM(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, A_init, B_init, C_init, D_init):
        super(LSSM, self).__init__()
        self.A = nn.Parameter(A_init)  # State transition matrix
        self.B = nn.Parameter(B_init)  # Input-to-state matrix
        self.C = nn.Parameter(C_init)  # State-to-output matrix
        self.D = nn.Parameter(D_init)  # Input-to-output matrix
        self.state_dim = state_dim
        self.reset_state()

    def reset_state(self):
        self.state = torch.zeros(self.state_dim)

    def forward(self, u):
        """
        Forward pass for the LSSM.
        :param u: Input tensor of shape (batch_size, seq_len, input_dim)
        :return: Output tensor of shape (batch_size, seq_len, output_dim)
        """
        outputs = []
        batch_size, seq_len, _ = u.size()
        self.state = self.state.to(u.device).unsqueeze(0).expand(batch_size, -1)
        
        for t in range(seq_len):
            # Update state
            self.state = torch.matmul(self.state, self.A.t()) + torch.matmul(u[:, t, :], self.B.t())
            # Compute output
            y = torch.matmul(self.state, self.C.t()) + torch.matmul(u[:, t, :], self.D.t())
            outputs.append(y.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        return outputs

def create_lssm_from_llama(llama_model):
    """
    Create an LSSM model by transferring weights from a pretrained LLaMA model.
    :param llama_model: Pretrained LLaMA model
    :return: Initialized LSSM model
    """
    # Extract dimensions
    hidden_size = llama_model.config.hidden_size
    intermediate_size = llama_model.config.intermediate_size
    vocab_size = llama_model.config.vocab_size
    print(f"hidden_size: {hidden_size}")
    print(f"intermediate_size: {intermediate_size}")
    print(f"vocab_size: {vocab_size}")

    # Extract MLP weights from the first transformer layer
    gate_proj_weight = llama_model.model.layers[0].mlp.gate_proj.weight  # Shape: (intermediate_size, hidden_size)
    up_proj_weight = llama_model.model.layers[0].mlp.up_proj.weight      # Shape: (intermediate_size, hidden_size)
    down_proj_weight = llama_model.model.layers[0].mlp.down_proj.weight  # Shape: (hidden_size, intermediate_size)
    print(f"gate_proj_weight shape: {gate_proj_weight.shape}")
    print(f"up_proj_weight shape: {up_proj_weight.shape}")
    print(f"down_proj_weight shape: {down_proj_weight.shape}")

    # Since we cannot directly combine these due to non-linearities,
    # we approximate the MLP as a single linear transformation for the LSSM.
    # Note: This is a simplification and ignores activation functions.
    W_mlp_approx = down_proj_weight @ (gate_proj_weight + up_proj_weight)  # Shape: (hidden_size, hidden_size)
    print(f"W_mlp_approx shape: {W_mlp_approx.shape}")

    # Initialize LSSM matrices
    A_init = W_mlp_approx.clone().detach()
    B_init = torch.eye(hidden_size)  # Identity matrix
    C_init = llama_model.lm_head.weight.clone().detach()  # Shape: (vocab_size, hidden_size)
    D_init = torch.zeros(vocab_size, hidden_size)  # Zero matrix

    print(f"A_init shape: {A_init.shape}")
    print(f"B_init shape: {B_init.shape}")
    print(f"C_init shape: {C_init.shape}")
    print(f"D_init shape: {D_init.shape}")

    # Create LSSM model
    lssm_model = LSSM(
        input_dim=hidden_size,
        state_dim=hidden_size,
        output_dim=vocab_size,
        A_init=A_init,
        B_init=B_init,
        C_init=C_init,
        D_init=D_init
    )
    return lssm_model

def main():
    # Load pretrained LLaMA model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

    # Create LSSM model from LLaMA weights
    lssm_model = create_lssm_from_llama(llama_model)

    # Tokenize input text
    input_text = "What is the capital of France?"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Shape: (1, seq_len)
    print(f"Input IDs: {input_ids}")
    print(f"inputs_ids shape: {input_ids.shape}")

    # Get input embeddings from LLaMA
    input_embeddings = llama_model.model.embed_tokens(input_ids)  # Shape: (1, seq_len, hidden_size)
    print(f"Input embeddings shape: {input_embeddings.shape}")

    # Pass embeddings through the LSSM
    lssm_outputs = lssm_model(input_embeddings)  # Shape: (1, seq_len, vocab_size)
    print(f"LSSM outputs shape: {lssm_outputs.shape}")

    # Decode LSSM outputs
    lssm_decoded = tokenizer.decode(torch.argmax(lssm_outputs, dim=-1).squeeze().tolist())
    print(f"LSSM decoded text: {lssm_decoded}")
    # Save LSSM model weights
    # torch.save(lssm_model.state_dict(), "lssm_weights.pth")


    # Pass inputs through the original LLaMA model for comparison
    llama_outputs = llama_model(input_ids, use_cache=False).logits  # Shape: (1, seq_len, vocab_size)
    print(f"LLaMA outputs shape: {llama_outputs.shape}")

    # Decode LLaMA outputs
    llama_decoded = tokenizer.decode(torch.argmax(llama_outputs, dim=-1).squeeze().tolist())
    print(f"LLaMA decoded text: {llama_decoded}")

    output_ids = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B").generate(input_ids, max_new_tokens=100, num_return_sequences=1)
    print(f"output_ids shape: {output_ids.shape}")
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")

    # Compare outputs
    difference = torch.mean(torch.abs(lssm_outputs - llama_outputs))
    print(f"Mean absolute difference between LSSM and LLaMA outputs: {difference.item()}")

if __name__ == "__main__":
    main()
