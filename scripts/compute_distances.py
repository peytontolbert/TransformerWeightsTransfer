import numpy as np
import torch
from scipy.spatial.distance import euclidean
from scripts.extract_weights import extract_weights
from scripts.unifiedmodel import UnifiedModel
from scripts.dataset import SyntheticSequenceDataset
from torch.utils.data import DataLoader
import torch.nn as nn

def compute_fisher_information(model, dataloader, criterion, device='cpu'):
    model.eval()
    fisher_info = {}
    for name, param in model.named_parameters():
        fisher_info[name] = torch.zeros_like(param)

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device).long()
        outputs = model(inputs)[0]  # Assuming the first output is relevant
        loss = criterion(outputs, targets)
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_info[name] += param.grad.data ** 2

    for name in fisher_info:
        fisher_info[name] /= len(dataloader)

    return fisher_info

def main():
    # Load extracted weights
    all_weights = np.load('extracted_weights.npy', allow_pickle=True).item()

    # Placeholder: Initialize your model architecture
    unified_model = UnifiedModel(
        input_dim=10,
        shared_embedding_dim=50,
        projection_dim=50,
        model_params={...}  # Use your existing model_params
    )

    # Example DataLoader (replace with actual data)
    dataset = SyntheticSequenceDataset(num_samples=100, seq_len=100, input_dim=10, output_dim=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unified_model.to(device)

    # Compute Fisher Information for each checkpoint
    fisher_infos = {}
    for epoch, weights in all_weights.items():
        unified_model.load_state_dict(weights, strict=True)
        fisher_info = compute_fisher_information(unified_model, dataloader, criterion, device)
        # Flatten Fisher Information
        fisher_flat = np.concatenate([fisher_info[k].cpu().numpy().flatten() for k in sorted(fisher_info.keys())])
        fisher_infos[epoch] = fisher_flat
        print(f"Computed Fisher Information for epoch {epoch}")

    # Save Fisher Information
    np.save('fisher_information.npy', fisher_infos)
    print("Fisher Information computed and saved.")

if __name__ == "__main__":
    main()