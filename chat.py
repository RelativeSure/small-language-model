import torch
from transformers import AutoTokenizer
from model import BigramLanguageModel

print("Loading model and tokenizer...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load the trained model
checkpoint = torch.load("wikitext_model.pt", weights_only=False)
vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]

# Create model and load weights
model = BigramLanguageModel(vocab_size, block_size, n_embd=384)

# Handle torch.compile prefix if present
state_dict = checkpoint["model_state_dict"]
if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
    # Remove _orig_mod. prefix added by torch.compile
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded! (Perplexity: {checkpoint['val_perplexity']:.2f})")
print(f"Using device: {device}\n")
print("=" * 80)
print("Chat with your WikiText-2 trained model!")
print("=" * 80)
print("Commands:")
print("  - Type your prompt and press Enter to generate text")
print("  - Type 'quit' or 'exit' to stop")
print("  - Type 'clear' for a fresh start")
print("=" * 80)

def generate_text(prompt, max_tokens=200, temperature=0.8):
    """Generate text continuation from a prompt"""
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Limit input length to block_size
    if input_ids.shape[1] > block_size:
        input_ids = input_ids[:, -block_size:]

    # Generate
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=max_tokens)

    # Decode and return
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

# Chat loop
while True:
    print("\n" + "─" * 80)
    user_input = input("You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break

    if user_input.lower() == 'clear':
        print("\n" * 50)  # Clear screen
        print("=" * 80)
        print("Starting fresh!")
        print("=" * 80)
        continue

    if not user_input:
        print("Please enter a prompt!")
        continue

    print("\nGenerating...\n")
    print("─" * 80)

    try:
        response = generate_text(user_input, max_tokens=200, temperature=0.8)
        print(f"Model: {response}")
    except Exception as e:
        print(f"Error generating text: {e}")

    print("─" * 80)
