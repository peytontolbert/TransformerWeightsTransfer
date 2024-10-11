from models.mamba.generation import LlamaMamba
import torch
from typing import List
import fire
import os  # Add this import if not already present
import sentencepiece as spm  # Assuming you're using SentencePiece for tokenization
from models.llama.generation import Llama as OriginalLlama
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def train(
    model: LlamaMamba,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    device: str = "cuda:0",
    epochs: int = 3,
):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    train: bool = False,  # {{ Added training flag }}
    dataset_path: str = "data/train.json",  # {{ Added dataset path }}
    epochs: int = 3,  # {{ Added epochs parameter }}
):
    # Debug: List all files in the checkpoint directory
    print(f"Checking checkpoint directory: {ckpt_dir}")
    try:
        checkpoint_files = os.listdir(ckpt_dir)
        print(f"Found {len(checkpoint_files)} files in the checkpoint directory:")
        for file in checkpoint_files:
            print(f" - {file}")
    except FileNotFoundError:
        print(f"Error: The directory {ckpt_dir} does not exist.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while accessing {ckpt_dir}: {e}")
        return

    # {{ Updated pad_id based on the tokenizer }}
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    pad_id = sp.pad_id() if sp.pad_id() != -1 else 0  # Use tokenizer's pad_id or default to 0

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer path {tokenizer_path} does not exist.")
        return
    oldllama = OriginalLlama(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Initialize the model using LlamaMamba instead of LlamaMambaModel
    model = LlamaMamba(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    model.to("cuda:0")  # Move the model to GPU

    # Ensure pad_id is within the valid range
    if pad_id < 0 or pad_id >= model.vocab_size:
        print(f"Error: pad_id {pad_id} is out of range for vocab size {model.vocab_size}.")
        return

    if train:
        from datasets import load_dataset  # {{ Added dataset import }}

        class TextDataset(Dataset):  # {{ Added dataset class }}
            def __init__(self, texts):
                self.texts = texts

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx], self.texts[idx]  # Simple autoencoding

        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path)["train"]
        texts = [item["text"] for item in dataset]
        train_dataset = TextDataset(texts)
        train_loader = DataLoader(train_dataset, batch_size=max_batch_size, shuffle=True)

        # Define optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        # Start training
        train(model, train_loader, optimizer, criterion, epochs=epochs)

    else:
        prompts: List[str] = [
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

        results = model.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for prompt, result in zip(prompts, results):
            print(prompt)
            print(f"> {result['generation']}")
            print("\n==================================\n")

if __name__ == "__main__":
    # Updated checkpoint directory to the absolute path where checkpoint files are located
    ckpt_dir: str = "checkpoints/llama-3.2-3b/"
    
    # Updated tokenizer path to match the absolute checkpoint directory
    tokenizer_path: str = "checkpoints/llama-3.2-3b/tokenizer.model"
    
    temperature: float = 0.6   
    top_p: float = 0.9
    max_seq_len: int = 500
    max_gen_len: int = 64
    max_batch_size: int = 4
    
    main(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_gen_len, max_batch_size, train=True, dataset_path="path/to/your/dataset.json", epochs=5)