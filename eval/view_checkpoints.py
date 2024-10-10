import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from scripts.dataset import SyntheticSequenceDataset
from scripts.unifiedmodel import UnifiedModel, FeatureProjection
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

# Parameters (should match those used during training)
num_samples = 100
seq_len = 1000
input_dim = 10
shared_embedding_dim = 50
projection_dim = 50
output_dim = 5  # Number of classes
batch_size = 32

# Model parameters for each individual model
model_params = {
    'transformer': {
        'model_dim': 20,
        'output_dim': 5,
        'num_heads': 2,
        'num_layers': 4
    },
    'mamba': {
        'hidden_dim': 1,
        'output_dim': 5,
        'num_layers': 2
    },
    'lstm': {
        'hidden_dim': 20,
        'output_dim': 5,
        'num_layers': 2
    },
    'liquid_s4': {
        'state_dim': 20,
        'output_dim': 5,
        'liquid_order': 2
    }
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
test_dataset = SyntheticSequenceDataset(num_samples, seq_len, input_dim, output_dim)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the Unified Model
unified_model = UnifiedModel(
    input_dim=input_dim,
    shared_embedding_dim=shared_embedding_dim,
    projection_dim=projection_dim,
    model_params=model_params
)

# Load the checkpoints
checkpoint_files = [
    'checkpoints/checkpoint_epoch_100.pth',
    'checkpoints/checkpoint_epoch_500.pth',
    'checkpoints/checkpoint_epoch_1000.pth'
]

all_weights = {}

# Function to extract and flatten model weights by architecture
def extract_architecture_weights(model):
    arch_weights = {
        'transformer': [],
        'mamba': [],
        'lstm': [],
        'liquid_s4': []
    }
    
    for name, param in model.named_parameters():
        if 'transformer' in name:
            arch_weights['transformer'].append(param.detach().cpu().numpy().flatten())
        elif 'mamba' in name:
            arch_weights['mamba'].append(param.detach().cpu().numpy().flatten())
        elif 'lstm' in name:
            arch_weights['lstm'].append(param.detach().cpu().numpy().flatten())
        elif 'liquid_s4' in name:
            arch_weights['liquid_s4'].append(param.detach().cpu().numpy().flatten())
    
    # Concatenate the weights for each architecture
    for arch in arch_weights:
        arch_weights[arch] = np.concatenate(arch_weights[arch])
    
    return arch_weights

# Load weights from each checkpoint and organize by architecture
architecture_weights = {arch: [] for arch in model_params.keys()}

for ckpt_file in checkpoint_files:
    ckpt = torch.load(ckpt_file, map_location=device)
    unified_model.load_state_dict(ckpt['model_state_dict'])
    arch_weights = extract_architecture_weights(unified_model)
    
    for arch, weights in arch_weights.items():
        architecture_weights[arch].append(weights)

# Visualize convergence with PCA
plt.figure(figsize=(10, 8))

for arch, weights_list in architecture_weights.items():
    # Perform PCA on the weights across checkpoints
    weights_pca = PCA(n_components=2).fit_transform(weights_list)
    
    # Plot the weights in 2D space for each architecture
    plt.plot(weights_pca[:, 0], weights_pca[:, 1], marker='o', label=f'{arch}', markersize=8)
    
    # Annotate each point with the corresponding checkpoint epoch
    for i, ckpt_file in enumerate(checkpoint_files):
        plt.text(weights_pca[i, 0], weights_pca[i, 1], ckpt_file.split('_')[-1].replace('.pth', ''))

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Convergence of Architecture Weights Across Checkpoints')
plt.legend()
plt.grid(True)
plt.show()
