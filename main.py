import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from model import BigramLanguageModel

# Load pre-trained tokenizer from sentence-transformers
print("Loading tokenizer from sentence-transformers/all-MiniLM-L6-v2...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

dataset = load_dataset("text", data_files="data/tiny_shakespeare.txt")
text = "\n".join(dataset["train"]["text"])

# Use the pre-trained tokenizer
def encode(text):
    return tokenizer.encode(text, add_special_tokens=False)


def decode(numbers):
    return tokenizer.decode(numbers, skip_special_tokens=True)


# For backward compatibility, keep char mappings (used in generate.py)
chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}


encoded_text = encode(text)
torch.tensor(encoded_text)

n = int(0.9 * len(encoded_text))

train_encoded_text = encoded_text[:n]
val_encoded_text = encoded_text[n:]


def get_batch(split):
    data = train_encoded_text if split == "train" else val_encoded_text
    block_size = 256
    batch_size = 128

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i : i + block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i + 1 : i + block_size + 1]) for i in ix])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device), y.to(device)


if __name__ == "__main__":
    # Use the tokenizer's vocab size
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    block_size = 256
    model = BigramLanguageModel(vocab_size, block_size, use_pretrained_embeddings=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for steps in range(10000):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if steps % 500 == 0:
            print(f"Step {steps}: loss {loss.item()}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)
    print(decode(generated[0].tolist()))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "block_size": block_size,
        },
        "shakespeare_model.pt",
    )

    print("\nModel saved to shakespeare_model.pt")
