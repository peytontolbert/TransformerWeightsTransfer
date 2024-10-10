import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from scripts.dataset import SyntheticSequenceDataset
from scripts.unifiedmodel import UnifiedModel, FeatureProjection
import torch.nn as nn

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

# Load the trained weights with weights_only=True to address the FutureWarning
"""
#######    IF USING FINAL WEIGHTS #####
unified_model.load_state_dict(torch.load('checkpoints/unified_model_final_weights.pth', map_location=device, weights_only=True))
"""
#######    IF USING CHECKPOINTS      #####
checkpoint = torch.load('checkpoints/checkpoint_epoch_100.pth', map_location=device)
unified_model.load_state_dict(checkpoint['model_state_dict'])

unified_model.to(device)
unified_model.eval()

# Define loss criterion
criterion = nn.CrossEntropyLoss()

# Dictionaries to store results
results = {
    'transformer': {'predictions': [], 'targets': [], 'loss': 0.0, 'differences': []},
    'mamba': {'predictions': [], 'targets': [], 'loss': 0.0, 'differences': []},
    'lstm': {'predictions': [], 'targets': [], 'loss': 0.0, 'differences': []},
    'liquid_s4': {'predictions': [], 'targets': [], 'loss': 0.0, 'differences': []}
}

with torch.no_grad():
    total_batches = len(test_dataloader)
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        if targets.max() >= output_dim or targets.min() < 0:
            raise ValueError(f"Targets should be in the range [0, {output_dim - 1}]. Found min: {targets.min()}, max: {targets.max()}")

        # Use the unified model's forward pass
        outputs, projected_features = unified_model(inputs)
        print("after outputs")
        # Iterate through each architecture and analyze outputs
        architectures = ['transformer', 'mamba', 'lstm', 'liquid_s4']
        for i, arch in enumerate(architectures):
            output = outputs[i]
            print("architecture", arch)
            print("output", output.shape)
            # Reshape output and targets to match CrossEntropyLoss requirements
            loss = criterion(output.view(-1, output_dim), targets.view(-1))
            print("loss", loss)
            results[arch]['loss'] += loss.item()

            # Compute differences between model outputs
            if i > 0:
                difference = output - outputs[0]
                results[arch]['differences'].append(difference.cpu().numpy())

            _, predicted = torch.max(output, 1)
            
            # {{ Ensure that predictions and targets have the same number of samples }}
            # Flatten the predictions
            pred_flat = predicted.cpu().numpy().flatten()
            # Flatten the targets
            target_flat = targets.cpu().numpy().flatten()
            
            # Check if lengths match
            if len(pred_flat) == len(target_flat):
                results[arch]['predictions'].extend(pred_flat)
                results[arch]['targets'].extend(target_flat)
            else:
                # {{ Adjust targets to match predictions if necessary }}
                min_length = min(len(pred_flat), len(target_flat))
                results[arch]['predictions'].extend(pred_flat[:min_length])
                results[arch]['targets'].extend(target_flat[:min_length])
                print(f"Warning: Mismatched lengths for {arch}. Truncated to {min_length} samples.")

# Compute average loss and accuracy for each architecture
for arch in results:
    avg_loss = results[arch]['loss'] / total_batches
    accuracy = accuracy_score(results[arch]['targets'], results[arch]['predictions'])
    print(f"Architecture: {arch}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(results[arch]['targets'], results[arch]['predictions']))
    print("-" * 50)

# Optional: Compare the performance between architectures
print("Performance Comparison Between Architectures:")
for arch in results:
    avg_loss = results[arch]['loss'] / total_batches
    accuracy = accuracy_score(results[arch]['targets'], results[arch]['predictions'])
    print(f"{arch.capitalize():<12} | Loss: {avg_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
