import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.s4model import LiquidS4  # Update import to LiquidS4



def create_liquid_s4_from_llama(llama_model):
    """
    Create a LiquidS4 model by transferring weights from a pretrained LLaMA model.
    :param llama_model: Pretrained LLaMA model
    :return: Initialized LiquidS4 model
    """
    # Extract dimensions
    hidden_size = llama_model.config.hidden_size
    vocab_size = llama_model.config.vocab_size
    num_layers = llama_model.config.num_hidden_layers  # {{ edit_1: Get the number of layers }}

    # Initialize LiquidS4 model with multiple layers
    liquid_s4_model = LiquidS4(
        vocab_size=vocab_size,
        state_dim=hidden_size,
        input_dim=hidden_size,
        output_dim=vocab_size,
        liquid_order=2,
        num_layers=num_layers  # {{ edit_2: Pass the number of layers }}
    )

    # Check if layer counts match
    assert len(llama_model.model.layers) == num_layers, "Mismatch in number of layers between LLaMA and LiquidS4 models."

    # Iterate over all layers to transfer weights
    for layer_idx in range(num_layers):
        # Transfer A, B, C matrices for each layer
        layer = llama_model.model.layers[layer_idx]
        if layer_idx < num_layers - 1:
            liquid_layer = liquid_s4_model.layers[layer_idx]  # {{ edit_3: Access corresponding LiquidS4 layer }}

            # Transfer query projection weights and biases
            liquid_layer.A.data.copy_(layer.self_attn.q_proj.weight.clone().detach())
            if layer.self_attn.q_proj.bias is not None:
                liquid_layer.A_bias.data.copy_(layer.self_attn.q_proj.bias.clone().detach())
            else:
                liquid_layer.A_bias.data.zero_()  # Initialize A_bias to zeros if q_proj.bias is None
            
            # Map feed-forward weights
            # Remove transpose to match dimensions
            print(f"Copying B with shape {layer.mlp.up_proj.weight.clone().detach().shape} to liquid_layer.B with shape {liquid_layer.B.data.shape}")
            liquid_layer.B.data.copy_(layer.mlp.up_proj.weight.clone().detach())  # {{ edit_4: Remove .t() to match dimensions }}
            liquid_layer.C.data.copy_(layer.mlp.down_proj.weight.clone().detach())

            torch.manual_seed(42)  # For reproducibility
            liquid_layer.liquid_kernel.data.copy_(torch.randn(hidden_size, hidden_size))  # Initialize as needed

            # Transfer LayerNorm parameters
            liquid_layer.layer_norm.weight.data.copy_(layer.self_attn.norm.weight.clone().detach())  # {{ edit_9: Transfer LayerNorm weight }}
            liquid_layer.layer_norm.bias.data.copy_(layer.self_attn.norm.bias.clone().detach())      # {{ edit_10: Transfer LayerNorm bias }}
        else:
            liquid_layer = liquid_s4_model.final_layer  # {{ edit_5: Access final output layer }}
            
            # Transfer query projection weights and biases
            liquid_layer.A.data.copy_(layer.self_attn.q_proj.weight.clone().detach())
            if layer.self_attn.q_proj.bias is not None:
                liquid_layer.A_bias.data.copy_(layer.self_attn.q_proj.bias.clone().detach())
            else:
                liquid_layer.A_bias.data.zero_()  # Initialize A_bias to zeros if q_proj.bias is None
            
            # Remove transpose to match dimensions
            print(f"Copying B with shape {layer.mlp.up_proj.weight.clone().detach().shape} to liquid_layer.B with shape {liquid_layer.B.data.shape}")
            liquid_layer.B.data.copy_(layer.mlp.up_proj.weight.clone().detach())  # {{ edit_6: Remove .t() to match dimensions }}
            liquid_layer.C.data.copy_(layer.mlp.down_proj.weight.clone().detach())

            torch.manual_seed(42)  # For reproducibility
            liquid_layer.liquid_kernel.data.copy_(torch.randn(hidden_size, hidden_size))  # Initialize as needed

            # Transfer LayerNorm parameters
            liquid_layer.layer_norm.weight.data.copy_(layer.self_attn.norm.weight.clone().detach())  # {{ edit_9: Transfer LayerNorm weight }}
            liquid_layer.layer_norm.bias.data.copy_(layer.self_attn.norm.bias.clone().detach())      # {{ edit_10: Transfer LayerNorm bias }}

        print(f"Transferred weights for layer {layer_idx + 1}/{num_layers}")  # {{ edit_7: Logging weight transfer }}

    return liquid_s4_model

def main():
    # Load pretrained LLaMA model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

    # Create LiquidS4 model from LLaMA weights
    liquid_s4_model = create_liquid_s4_from_llama(llama_model)

    # Tokenize input text
    input_text = "What is the capital of France?"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Shape: (1, seq_len)

    # Get input embeddings from LLaMA
    input_embeddings = llama_model.model.embed_tokens(input_ids)  # Shape: (1, seq_len, hidden_size)

    # Pass embeddings through the LiquidS4
    liquid_s4_outputs = liquid_s4_model(input_embeddings)  # Shape: (1, seq_len, vocab_size)
    liquid_s4_decoded = tokenizer.decode(torch.argmax(liquid_s4_outputs[0], dim=-1).squeeze().tolist())  # Access the first element of the tuple
    print(f"LiquidS4 decoded text: {liquid_s4_decoded}")
    # Save LiquidS4 model weights
    torch.save(liquid_s4_model.state_dict(), "liquid_s4_weights.pth")

    # Pass inputs through the original LLaMA model for comparison
    llama_outputs = llama_model(input_ids, use_cache=False).logits  # Shape: (1, seq_len, vocab_size)

    # Decode LLaMA outputs
    llama_decoded = tokenizer.decode(torch.argmax(llama_outputs, dim=-1).squeeze().tolist())
    print(f"LLaMA decoded text: {llama_decoded}")
    output_ids = llama_model.generate(input_ids, max_new_tokens=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")
    # Compare outputs
    difference = torch.mean(torch.abs(liquid_s4_outputs[0] - llama_outputs))  # Access the first element of the tuple
    print(f"Mean absolute difference between LiquidS4 and LLaMA outputs: {difference.item()}")

    # {{ edit_11: Validation Step }}
    assert torch.allclose(liquid_s4_outputs, llama_outputs, atol=1e-5), "LiquidS4 outputs do not match LLaMA outputs within tolerance."
    print("Validation successful: LiquidS4 is a close clone of LLaMA.")

if __name__ == "__main__":
    main()