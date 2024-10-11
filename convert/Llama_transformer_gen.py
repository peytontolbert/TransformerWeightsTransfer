from models.llama.generation import Llama
import torch
from typing import List
import fire
import os  # Add this import if not already present


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
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

    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


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

    #model.to("cuda:0")  # Moved LLaMATransformer to GPU



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
    
    main(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_gen_len, max_batch_size)