import torch
import torch.nn as nn
import torch.optim as optim
from models.llama.generation import Llama  # Assuming your original model is here
from models.mamba.generation import LlamaMamba
from typing import List

# Updated checkpoint directory to the absolute path where checkpoint files are located
ckpt_dir: str = "checkpoints/llama-3.2-3b/"
# Updated tokenizer path to match the absolute checkpoint directory
tokenizer_path: str = "checkpoints/llama-3.2-3b/tokenizer.model"
max_seq_len: int = 128
max_batch_size: int = 4

# Initialize the models
target_model = LlamaMamba.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )  # This is the pretrained model
source_model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )  # This is the new architecture

# Initialize the tokenizer from the source model
tokenizer = source_model.tokenizer  # Assuming LlamaMamba exposes the tokenizer

# Load data - replace this with your actual data loader
def get_dummy_data(batch_size=32, seq_len=128):
    # Create dummy tokenized data
    X: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    return X

# Training function
def train_model(source_model, target_model, criterion, optimizer, num_epochs=5):
    source_model.train()
    target_model.train()

    for epoch in range(num_epochs):
        # {{ Replace get_dummy_data with text_completion_train }}
        prompts = [
            # For these prompts, the expected answer is the natural continuation of the prompt
            "I believe the meaning of life is",
            "Simply put, the theory of relativity states that ",
            """A brief message congratulating the team on the launch:

            Hi everyone,

            I just """,
            # Few shot prompt (providing a few examples before asking model to complete more);
            """Translate English to French:

            sea otter => loutre de mer
            peppermint => menthe poivrée
            plush girafe => girafe peluche
            cheese =>""",
        ] 
        max_gen_len: int = 64
        temperature: float = 0.6
        top_p: float = 0.9
        print(f"prompts: {prompts}")
        labels, decoded_prompts = source_model.text_completion_train(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )
        # Convert labels (tokens) from list to tensor
        #print(f"labels: {labels.shape}")
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass through target model
        outputs = target_model.text_completion_train(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p)  # {{ Update to match new forward signature }}
        #print(f"outputs: {outputs.shape}")
        
        # Convert outputs to tensor if they are not already a tensor
        if isinstance(outputs, list):
            outputs = torch.tensor(outputs, dtype=torch.float, requires_grad=True)

        # Convert outputs to tensor if they are not already a tensor
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.float, requires_grad=True)

        loss = criterion(outputs, labels)  # Compute loss with correctly shaped tensors
        prompts = decoded_prompts
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Modify according to your task
optimizer = optim.Adam(source_model.parameters(), lr=0.001)

# Train and save the new weights into the target model
train_model(source_model, target_model, criterion, optimizer)

# Save the new target model weights
torch.save(target_model.state_dict(), "new_mamba_weights.pth")
print("New weights saved to new_mamba_weights.pth")