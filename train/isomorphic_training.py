import torch
from torch.utils.data import DataLoader
from scripts.dataset import SyntheticSequenceDataset
from scripts.unifiedmodel import UnifiedModel, FeatureProjection
import torch.nn as nn
import os
from torch.amp import GradScaler, autocast  # Updated mixed precision support
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR  # Import LinearLR for warmup

# Parameters
num_samples = 1000
seq_len = 10
input_dim = 10
shared_embedding_dim = 50
projection_dim = 50
output_dim = 5  # Number of classes
batch_size = 64
num_checkpoints = 10  # Number of checkpoints to save during training
checkpoint_dir = './checkpoints/'  # Directory to save checkpoints
dropout = nn.Dropout(p=0.05)  # Apply dropout after certain layers

# Define warmup_epochs globally to fix Undefined name `warmup_epochs`
warmup_epochs = 10

# Create directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# Define hidden_dim dynamically (e.g., through a configuration or user input)
hidden_dim = 4  # Increased from 2 to 4

# Define model parameters for each individual model
model_params = {
    'transformer': {
        'model_dim': 80,         # Increased from 40 to 80
        'output_dim': 20,        # Increased from 10 to 20
        'num_heads': 8,          # Increased from 4 to 8
        'num_layers': 16         # Increased from 8 to 16
    },
    'mamba': {
        'hidden_dim': 80,        # Changed from 40 to 80
        'output_dim': 20,        # Changed from 10 to 20
        'num_layers': 8          # Changed from 4 to 8
    },
    'lstm': {
        'hidden_dim': 80,        # Changed from 40 to 80
        'output_dim': 20,        # Changed from 10 to 20
        'num_layers': 8          # Changed from 4 to 8
    },
    'liquid_s4': {
        'state_dim': 80,         # Changed from 40 to 80
        'output_dim': 20,        # Changed from 10 to 20
        'liquid_order': 8        # Changed from 4 to 8
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

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)

def train_unified_model(unified_model, dataloader, criterion, optimizer, target_loss, max_epochs=2000, max_norm=0.5, consistency_weight_output=0.0, consistency_weight_feature=0.0):
    # Set consistency weights to zero to disable consistency losses
    epoch = 0
    avg_loss = float('inf')
    losses = []
    checkpoints_saved = 0
    
    scaler = GradScaler()  # Initialize GradScaler for mixed precision
    
    # Implement learning rate warmup using the globally defined warmup_epochs
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6)
    
    # Initialize individual loss tracking for each model
    model_names = ['transformer', 'mamba', 'lstm', 'liquid_s4']
    model_loss_totals = { name: 0.0 for name in model_names }
    
    device_type = "cuda:0" if device.type == "cuda:0" else "cpu"  # Define device_type based on device

    while avg_loss > target_loss and epoch < max_epochs:
        total_loss = 0
        primary_loss_total = 0
        consistency_loss_total = 0
        unified_model.train()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            
            with autocast(device_type):  # Enable mixed precision with device_type
                # Forward pass
                outputs, projected_features = unified_model(inputs)
                
                # Compute primary losses
                primary_losses = [nn.functional.cross_entropy(output.permute(0, 2, 1), targets) for output in outputs]
                
                # Track individual model losses
                for name, loss in zip(model_names, primary_losses):
                    model_loss_totals[name] += loss.item()
    
                # Compute output consistency losses
                consistency_losses = []
                for i in range(len(outputs)):
                    for j in range(i + 1, len(outputs)):
                        consistency_loss = nn.functional.mse_loss(outputs[i], outputs[j])
                        consistency_losses.append(consistency_loss)
    
                # Compute feature consistency losses
                feature_consistency_losses = []
                num_layers = 2  # Define the number of layers (adjust as needed)
                for layer_idx in range(num_layers):  # Assuming same number of layers
                    current_layer_features = [feat[layer_idx] for feat in projected_features]
                    # Compute pairwise consistency loss
                    for i in range(len(current_layer_features)):
                        for j in range(i + 1, len(current_layer_features)):
                            feature_loss = nn.functional.mse_loss(current_layer_features[i], current_layer_features[j])
                            feature_consistency_losses.append(feature_loss)
    
                # Dynamically adjust consistency weights based on epoch
                consistency_weight_output = min(0.005 + epoch * 0.002, 0.05)  # Adjusted increment
                consistency_weight_feature = min(0.002 + epoch * 0.001, 0.005)  # Adjusted increment
    
                # Total loss with scaled consistency losses
                total_primary_loss = sum(primary_losses)
                total_consistency_loss = (sum(consistency_losses) * consistency_weight_output) + (sum(feature_consistency_losses) * consistency_weight_feature)
                loss = total_primary_loss + total_consistency_loss
    
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at epoch {epoch}, batch {batch_idx}. Stopping training.")
                return float('inf')  # or handle as appropriate
    
            # Backward pass with mixed precision
            optimizer.zero_grad()
            scaler.scale(loss).backward()
    
            # Apply gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unified_model.parameters(), max_norm=1)
    
            # Step the optimizer with dynamic learning rates
            scaler.step(optimizer)
            scaler.update()
    
            # Update learning rate schedulers
            if epoch < warmup_epochs:
                scheduler_warmup.step()
            else:
                scheduler_cosine.step()
    
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
        
        # Compute average individual model losses
        avg_model_losses = { name: total / len(dataloader) for name, total in model_loss_totals.items() }
        
        # Print average losses for the current epoch
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Primary Loss = {avg_primary_loss:.4f}, Consistency Loss = {avg_consistency_loss:.4f}")
        for name, loss in avg_model_losses.items():
            print(f" - {name} Loss: {loss:.4f}")
        
        # Reset model_loss_totals for next epoch
        model_loss_totals = { name: 0.0 for name in model_names }
        
        # Save checkpoints
        if epoch % ((max_epochs - warmup_epochs) // num_checkpoints) == 0 and checkpoints_saved < num_checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(unified_model, optimizer, epoch, checkpoint_path)
            checkpoints_saved += 1
            print(f"Checkpoint saved at epoch {epoch}.")
    
        # Early stopping condition
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

max_epochs = 2000   # Increased from 1000 to allow more training
# Define criterion with reduced label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced from 0.1 to 0.05
# **Adjust optimizer with dynamic learning rates for each sub-model**
optimizer = torch.optim.AdamW([
    {'params': unified_model.shared_embedding.parameters(), 'lr': 1e-2},  # Increased learning rate
    {'params': unified_model.transformer.parameters(), 'lr': 1e-1},
    {'params': unified_model.mamba.parameters(), 'lr': 1e-1},
    {'params': unified_model.lstm.parameters(), 'lr': 1e-1},
    {'params': unified_model.liquid_s4.parameters(), 'lr': 1e-1},
    {'params': unified_model.projections.parameters(), 'lr': 1e-1},
], weight_decay=1e-4)  # Base lr is overridden by parameter groups

# **Modify learning rate schedulers to handle multiple parameter groups**
scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6)

# Move model to device if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unified_model.to(device)

# Train the unified model
target_loss = 1e-6  # Target loss
consistency_weight_output = 0.0  # Disabled consistency losses
consistency_weight_feature = 0.0  # Disabled consistency losses

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Move model to device with mixed precision
unified_model.to(device)

# Train the unified model with scaler
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

# Save the final model weights
torch.save(unified_model.state_dict(), 'checkpoints/unified_model_final_weight.pth')

if not torch.backends.cudnn.enabled:
    print("Warning: cudnn is not enabled. Flash Attention may not be available.")