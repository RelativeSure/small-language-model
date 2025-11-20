import torch

# Load character mappings
from main import chars, decode
from model import BigramLanguageModel

# Load the saved model
checkpoint = torch.load("shakespeare_model.pt")
vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]

# Create model and load weights
model = BigramLanguageModel(vocab_size, block_size)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=1000)
print(decode(generated[0].tolist()))
