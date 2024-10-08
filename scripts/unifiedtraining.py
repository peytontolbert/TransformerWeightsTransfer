import torch
from torch.utils.data import DataLoader
from scripts.dataset import SyntheticSequenceDataset
from scripts.unifiedmodel import UnifiedModel, FeatureProjection
import torch.nn as nn


# Parameters
num_samples = 10
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

dataset = SyntheticSequenceDataset(num_samples, seq_len, input_dim, output_dim)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Gradient clipping function
def apply_gradient_clipping(models, max_norm=1.0):
    for model in models:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Early stopping function with patience
def should_early_stop(losses, patience=5, delta=1e-4):
    if len(losses) < patience:
        return False
    return all(abs(losses[-i] - losses[-i-1]) < delta for i in range(1, patience))

def train_unified_model(unified_model, dataloader, criterion, optimizer, target_loss, max_epochs=100, max_norm=1.0, consistency_weight_output=0.1, consistency_weight_feature=0.05):
    epoch = 0
    avg_loss = float('inf')
    losses = []
    
    while avg_loss > target_loss and epoch < max_epochs:
        total_loss = 0
        primary_loss_total = 0
        consistency_loss_total = 0
        unified_model.train()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move inputs and targets to the same device as the model
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            # Ensure targets are within the valid range
            if targets.max() >= output_dim or targets.min() < 0:
                raise ValueError(f"Targets should be in the range [0, {output_dim - 1}]. Found min: {targets.min()}, max: {targets.max()}")
            
            
            # **Validate Input Data**
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                raise ValueError("Input contains NaN or Inf values.")
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                raise ValueError("Targets contain NaN or Inf values.")
            
            # Forward pass
            outputs, projected_features = unified_model(inputs)

            #for i, output in enumerate(outputs):
                #print(f"Model {i} output shape: {output.shape}")
                #print(f"Model {i} output sample: {output[0, 0]}")

            # Compute primary losses
            primary_losses = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    consistency_loss = nn.functional.mse_loss(outputs[i], outputs[j])
                    primary_losses.append(consistency_loss)

            # Compute output consistency losses
            consistency_losses = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    # Remove .detach() to allow gradient flow through consistency loss
                    consistency_loss = nn.functional.mse_loss(outputs[i], outputs[j])
                    consistency_losses.append(consistency_loss)

            # Compute feature consistency losses
            feature_consistency_losses = []
            num_layers = 2  # Define the number of layers (adjust as needed)
            for layer_idx in range(num_layers):  # Assuming same number of layers or adjust accordingly
                current_layer_features = [feat[layer_idx] for feat in projected_features]
                # Compute pairwise consistency loss
                for i in range(len(current_layer_features)):
                    for j in range(i + 1, len(current_layer_features)):
                        # Remove .detach() to allow gradient flow through feature consistency loss
                        feature_loss = nn.functional.mse_loss(current_layer_features[i], current_layer_features[j])
                        feature_consistency_losses.append(feature_loss)

            # Total loss with scaled consistency losses
            total_primary_loss = sum(primary_losses)
            total_consistency_loss = (sum(consistency_losses) * consistency_weight_output) + (sum(feature_consistency_losses) * consistency_weight_feature)
            loss = total_primary_loss + total_consistency_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(unified_model.parameters(), max_norm=1)  # Reduced max_norm from 1.0 to 0.5

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()
            primary_loss_total += total_primary_loss.item()
            consistency_loss_total += total_consistency_loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        avg_primary_loss = primary_loss_total / len(dataloader)
        avg_consistency_loss = consistency_loss_total / len(dataloader)
        losses.append(avg_loss)
        epoch += 1
        print(f"Epoch [{epoch + 1}/{max_epochs}], Avg Loss: {avg_loss:.4f}, Primary Loss: {avg_primary_loss:.4f}, Consistency Loss: {avg_consistency_loss:.4f}")
        
        
        # Step the Scheduler
        scheduler.step(avg_loss)
        # Early stopping if loss plateaus
        if len(losses) > 5 and abs(losses[-1] - losses[-2]) < 1e-4:
            print("Early stopping due to minimal loss improvement.")
            break
        if should_early_stop(losses):
            print("Early stopping due to lack of progress.")
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
optimizer = torch.optim.AdamW(unified_model.parameters(), lr=0.005, weight_decay=1e-5)  # Corrected learning rate from 0.05 to 0.0001

# **Added Learning Rate Scheduler**
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Move model to device if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unified_model.to(device)

# Train the unified model
print("Training Unified Model with Consistency Loss")
target_loss = .001  # Adjust based on initial observations
max_epochs = 1000   # Maximum epochs to prevent infinite loops
consistency_weight_output = 0.1
consistency_weight_feature = 0.5  # May require experimentation

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