import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticSequenceDataset(Dataset):
    def __init__(self, num_samples, seq_len, input_dim, num_classes):
        super(SyntheticSequenceDataset, self).__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Generate random input sequences
        self.inputs = torch.randn(num_samples, seq_len, input_dim)

        # Generate random target sequences (classes from 0 to num_classes-1)
        self.targets = torch.randint(0, num_classes, (num_samples, seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
