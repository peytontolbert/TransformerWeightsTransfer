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

    # Initialize LiquidS4 matrices using LLaMA weights
    A_init = llama_model.model.layers[0].self_attn.q_proj.weight.clone().detach()  # Example initialization
    B_init = torch.eye(hidden_size)  # Identity matrix or appropriate initialization
    C_init = llama_model.lm_head.weight.clone().detach()
    liquid_kernel_init = torch.randn(hidden_size, hidden_size)  # Initialize as needed

    # Create LiquidS4 model
    liquid_s4_model = LiquidS4(
        state_dim=hidden_size,
        input_dim=hidden_size,
        output_dim=vocab_size,
        liquid_order=2  # Adjust as necessary
    )
    liquid_s4_model.A.data.copy_(A_init)
    liquid_s4_model.B.data.copy_(B_init)
    liquid_s4_model.C.data.copy_(C_init.transpose(0, 1))  # Transpose to match (hidden_size, vocab_size)
    liquid_s4_model.liquid_kernel.data.copy_(liquid_kernel_init)

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

if __name__ == "__main__":
    main()