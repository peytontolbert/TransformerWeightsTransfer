import torch
from torch.utils.data import DataLoader
from scripts.dataset import SyntheticSequenceDataset
from scripts.unifiedmodel import UnifiedModel, FeatureProjection
import torch.nn as nn


# Parameters
num_samples = 1000
seq_len = 100
input_dim = 10
shared_embedding_dim = 50
projection_dim = 50
output_dim = 5  # Number of classes
batch_size = 32

# Define model parameters for each individual model
model_params = {
    'transformer': {
        'model_dim': 20,
        'output_dim': 5,
        'num_heads': 2,
        'num_layers': 2
    },
    'mamba': {
        'hidden_dim': 1,
        'output_dim': 5,
        'num_layers': 1
    },
    'lstm': {
        'hidden_dim': 20,
        'output_dim': 5,
        'num_layers': 1
    },
    'liquid_s4': {
        'state_dim': 20,
        'output_dim': 5,
        'liquid_order': 2
    }
}

dataset = SyntheticSequenceDataset(num_samples, seq_len, input_dim, output_dim)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Gradient clipping function
def apply_gradient_clipping(models, max_norm=1.0):
    for model in models:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def train_unified_model(unified_model, dataloader, criterion, optimizer, target_loss, max_epochs=100, max_norm=1.0, consistency_weight_output=0.1, consistency_weight_feature=0.05):
    epoch = 0
    avg_loss = float('inf')
    losses = []
    
    while avg_loss > target_loss and epoch < max_epochs:
        total_loss = 0
        unified_model.train()
        for inputs, targets in dataloader:
            # Move inputs and targets to the same device as the model
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Ensure targets are of type long for CrossEntropyLoss
            targets = targets.long()
            
            # Forward pass
            outputs, projected_features = unified_model(inputs)

            # Compute primary losses
            primary_losses = []
            for output in outputs:
                # Reshape outputs and targets for loss computation
                output = output.view(-1, output.size(-1))  # (batch_size * seq_len, output_dim)
                target = targets.view(-1)  # (batch_size * seq_len)
                
                # Check target values are within the correct range
                if target.max() >= output.size(-1) or target.min() < 0:
                    raise ValueError(f"Target values {target.min()} to {target.max()} are out of range [0, {output.size(-1)-1}]")
                
                primary_loss = criterion(output, target)
                primary_losses.append(primary_loss)

            # Compute output consistency losses
            consistency_losses = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    # Ensure outputs are detached to prevent gradient flow through consistency loss
                    consistency_loss = nn.functional.mse_loss(outputs[i].detach(), outputs[j].detach())
                    consistency_losses.append(consistency_loss)

            # Compute feature consistency losses
            feature_consistency_losses = []
            num_layers = 2  # Define the number of layers (adjust as needed)
            for layer_idx in range(num_layers):  # Assuming same number of layers or adjust accordingly
                current_layer_features = [feat[layer_idx] for feat in projected_features]
                # Compute pairwise consistency loss
                for i in range(len(current_layer_features)):
                    for j in range(i + 1, len(current_layer_features)):
                        feature_loss = nn.functional.mse_loss(current_layer_features[i].detach(), current_layer_features[j].detach())
                        feature_consistency_losses.append(feature_loss)

            # Total loss with scaled consistency losses
            total_primary_loss = sum(primary_losses)
            total_consistency_loss = (sum(consistency_losses) * consistency_weight_output) + (sum(feature_consistency_losses) * consistency_weight_feature)
            loss = total_primary_loss + total_consistency_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            apply_gradient_clipping([unified_model], max_norm=max_norm)

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        epoch += 1
        print(f"Epoch [{epoch}], Loss: {avg_loss:.4f}")
        # Early stopping if loss plateaus
        if len(losses) > 5 and abs(losses[-1] - losses[-2]) < 1e-4:
            print("Early stopping due to minimal loss improvement.")
            break
    return avg_loss

#### Initialize the Unified Model ####

unified_model = UnifiedModel(
    input_dim=input_dim,
    shared_embedding_dim=shared_embedding_dim,
    projection_dim=projection_dim,
    model_params=model_params
)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unified_model.parameters(), lr=0.0001)  # Reduced learning rate from 0.001 to 0.0001

# Move model to device if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unified_model.to(device)

# Train the unified model
print("Training Unified Model with Consistency Loss")
target_loss = 1.611  # Adjust based on initial observations
max_epochs = 100   # Maximum epochs to prevent infinite loops
consistency_weight_output = 0.1
consistency_weight_feature = 0.05  # May require experimentation

final_loss = train_unified_model(
    unified_model,
    dataloader,
    criterion,
    optimizer,
    target_loss,
    max_epochs,
    max_norm=1.0,
    consistency_weight_output=consistency_weight_output,
    consistency_weight_feature=consistency_weight_feature
)

print(f"\nFinal Loss: {final_loss:.4f}")

# Save the unified model
torch.save(unified_model.state_dict(), 'unified_model_weights.pth')