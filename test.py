import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


# Define input text
input_text = "What is the capital of France?"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')
print(f"Input IDs: {input_ids}")

# Generate text using the model
output_ids = model.generate(input_ids, max_new_tokens=500, num_return_sequences=1)
print(f"Output IDs: {output_ids}")

# Decode the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")
