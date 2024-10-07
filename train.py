from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from main import LSSM
import torch.nn as nn
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")



# Load WikiText dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

# Preprocess dataset: tokenize the raw text
def preprocess_function(examples):
    texts = examples['text']
    
    # Set or add padding token
    tokenizer.pad_token = tokenizer.eos_token  # Or you can use tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Tokenize the raw text
    return tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")


hidden_size = 3072
vocab_size = 128256
epochs = 2
# Load QA dataset (example: SQuAD)
lssm_model = LSSM(input_dim=hidden_size, state_dim=hidden_size, output_dim=vocab_size, A_init=torch.empty(hidden_size, hidden_size), B_init=torch.empty(hidden_size, hidden_size), C_init=torch.empty(vocab_size, hidden_size), D_init=torch.empty(vocab_size, hidden_size))
lssm_model.load_state_dict(torch.load("lssm_weights.pth"))

tokenizer.pad_token = tokenizer.eos_token  # Or you can use a custom token like '[PAD]'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# Apply preprocessing
train_dataset = dataset['train'].map(preprocess_function, batched=True)
train_dataloader = DataLoader(train_dataset, batch_size=8)
optimizer = torch.optim.Adam(lssm_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch in train_dataloader:
        inputs = batch['input_ids']
        labels = batch['labels']  # Adjust according to the dataset format
        
        # Get input embeddings from LLaMA (or your tokenizer/model)
        input_embeddings = tokenizer.embed_tokens(inputs)
        
        # Pass embeddings through LSSM
        outputs = lssm_model(input_embeddings)
        
        # Calculate loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.save(lssm_model.state_dict(), "lssm_wiki_weights.pth")
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
