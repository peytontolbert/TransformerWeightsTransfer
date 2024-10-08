import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from models.transformer import SimpleTransformer
from models.mamba import MambaModel
from models.lstm import SimpleLSTM
from models.s4model import LiquidS4
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Parameters
num_samples = 1000
test_split = 0.2  # 20% for testing

seq_len = 100
input_dim = 10
output_dim = 5  # Number of classes
batch_size = 32
class SyntheticSequenceDataset(Dataset):
    def __init__(self, num_samples, seq_len, input_dim, output_dim, scaler=None):
        super(SyntheticSequenceDataset, self).__init__()
        self.inputs = torch.randn(num_samples, seq_len, input_dim)
        self.targets = torch.randint(0, output_dim, (num_samples, seq_len))
        
        if scaler is None:
            self.scaler = StandardScaler()
            all_data = self.inputs.view(-1, input_dim).numpy()
            self.scaler.fit(all_data)
        else:
            self.scaler = scaler
        
        # Apply scaling
        self.inputs = torch.tensor(
            self.scaler.transform(self.inputs.view(-1, input_dim)), dtype=torch.float32
        ).view(num_samples, seq_len, input_dim)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]
        return input_seq, target_seq

# Create the full dataset
full_dataset = SyntheticSequenceDataset(num_samples, seq_len, input_dim, output_dim)

# Calculate split sizes
test_size = int(num_samples * test_split)
train_size = num_samples - test_size

# Split the dataset
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    
# Initialize the scaler with training data
train_dataset = SyntheticSequenceDataset(train_size, seq_len, input_dim, output_dim)
scaler = train_dataset.scaler

# Initialize the test dataset with the same scaler
test_dataset = SyntheticSequenceDataset(test_size, seq_len, input_dim, output_dim, scaler=scaler)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            # outputs: [batch_size, seq_len, output_dim]
            # targets: [batch_size, seq_len]
            
            # Get the predicted class by taking the argmax
            _, preds = torch.max(outputs, dim=-1)  # preds: [batch_size, seq_len]
            
            # Append to the lists
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')  # You can choose 'macro', 'micro', etc.
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return accuracy, f1, conf_matrix

# Initialize model
transformer_model = SimpleTransformer(input_dim=10, model_dim=20, output_dim=5, num_heads=2, num_layers=2)
# Initialize model
mamba_model = MambaModel(input_dim=10, hidden_dim=20, output_dim=5, num_layers=1)
# Initialize model
lstm_model = SimpleLSTM(input_dim=10, hidden_dim=20, output_dim=5, num_layers=1)
# Assuming LiquidS4 model is already defined as per your code
liquid_model = LiquidS4(state_dim=20, input_dim=10, output_dim=5, liquid_order=2)

# Example of loading models (if not already in memory)
transformer_model.load_state_dict(torch.load('transformer_weights.pth'))
mamba_model.load_state_dict(torch.load('mamba_weights.pth'))
lstm_model.load_state_dict(torch.load('lstm_weights.pth'))
liquid_model.load_state_dict(torch.load('liquid_weights.pth'))
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to device
transformer_model.to(device)
mamba_model.to(device)
lstm_model.to(device)
liquid_model.to(device)

# Evaluate Transformer Model
print("Evaluating Transformer Model...")
transformer_acc, transformer_f1, transformer_cm = evaluate_model(transformer_model, test_dataloader, device)
print(f"Transformer Accuracy: {transformer_acc:.4f}")
print(f"Transformer F1 Score: {transformer_f1:.4f}")
print("Transformer Confusion Matrix:")
print(transformer_cm)

# Evaluate Mamba Model
print("\nEvaluating Mamba Model...")
mamba_acc, mamba_f1, mamba_cm = evaluate_model(mamba_model, test_dataloader, device)
print(f"Mamba Model Accuracy: {mamba_acc:.4f}")
print(f"Mamba Model F1 Score: {mamba_f1:.4f}")
print("Mamba Model Confusion Matrix:")
print(mamba_cm)

# Evaluate LSTM Model
print("\nEvaluating LSTM Model...")
lstm_acc, lstm_f1, lstm_cm = evaluate_model(lstm_model, test_dataloader, device)
print(f"LSTM Model Accuracy: {lstm_acc:.4f}")
print(f"LSTM Model F1 Score: {lstm_f1:.4f}")
print("LSTM Model Confusion Matrix:")
print(lstm_cm)

# Evaluate LiquidS4 Model
print("\nEvaluating LiquidS4 Model...")
liquid_acc, liquid_f1, liquid_cm = evaluate_model(liquid_model, test_dataloader, device)
print(f"LiquidS4 Model Accuracy: {liquid_acc:.4f}")
print(f"LiquidS4 Model F1 Score: {liquid_f1:.4f}")
print("LiquidS4 Model Confusion Matrix:")
print(liquid_cm)
