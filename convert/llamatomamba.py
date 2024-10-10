from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# Load the LlamaForCausalLM model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
model.to("cuda:0")  # Move model to GPU

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", legacy=False)

input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

# Generate output
generated_ids = model.generate(
    inputs["input_ids"],
    max_length=500,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Decode the generated tokens
decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(decoded_output)
