
import torch
import os
import numpy as np

def extract_weights(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    weights = {}
    for key, value in state_dict.items():
        weights[key] = value.numpy().flatten()
    return weights

def main():
    checkpoints_dir = 'checkpoints'
    checkpoints = sorted([os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith('.pth')])

    all_weights = {}
    for cp in checkpoints:
        epoch = int(cp.split('_')[-1].split('.pth')[0])
        all_weights[epoch] = extract_weights(cp)
        print(f"Extracted weights from epoch {epoch}")

    # Save extracted weights for further processing
    np.save('extracted_weights.npy', all_weights)
    print("All weights extracted and saved to extracted_weights.npy")

if __name__ == "__main__":
    main()