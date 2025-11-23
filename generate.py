import torch
from transformers import AutoTokenizer
from model import BigramLanguageModel

# Load the tokenizer
print("Loading GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def decode(numbers):
    return tokenizer.decode(numbers, skip_special_tokens=True)

# Load the saved model
checkpoint = torch.load("shakespeare_model.pt", weights_only=False)
vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]

# Create model and load weights
model = BigramLanguageModel(vocab_size, block_size, n_embd=384)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate text
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=1000)
print(decode(generated[0].tolist()))
