import torch
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import math

from model import BigramLanguageModel

# Configuration
USE_WIKITEXT = True  # Set to False to use Shakespeare dataset

# Load pre-trained tokenizer from sentence-transformers
print("Loading tokenizer from sentence-transformers/all-MiniLM-L6-v2...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load dataset
if USE_WIKITEXT:
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join([text for text in dataset["train"]["text"] if text.strip()])
    val_text = "\n".join([text for text in dataset["validation"]["text"] if text.strip()])
    text = train_text  # For char mappings
else:
    print("Loading Shakespeare dataset...")
    dataset = load_dataset("text", data_files="data/tiny_shakespeare.txt")
    text = "\n".join(dataset["train"]["text"])
    train_text = text
    val_text = None

# Use the pre-trained tokenizer
def encode(text):
    return tokenizer.encode(text, add_special_tokens=False)


def decode(numbers):
    return tokenizer.decode(numbers, skip_special_tokens=True)


# For backward compatibility, keep char mappings (used in generate.py)
chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}


# Encode texts
if USE_WIKITEXT and val_text:
    print("Encoding train and validation texts...")
    train_encoded_text = encode(train_text)
    val_encoded_text = encode(val_text)
else:
    print("Encoding text...")
    encoded_text = encode(text)
    n = int(0.9 * len(encoded_text))
    train_encoded_text = encoded_text[:n]
    val_encoded_text = encoded_text[n:]

print(f"Train tokens: {len(train_encoded_text):,}")
print(f"Validation tokens: {len(val_encoded_text):,}")


def get_batch(split, block_size=256, batch_size=128, device="cuda"):
    data = train_encoded_text if split == "train" else val_encoded_text

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i : i + block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i + 1 : i + block_size + 1]) for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, eval_iters=100):
    """Evaluate model and compute perplexity"""
    model.eval()
    losses = {"train": [], "val": []}

    for split in ["train", "val"]:
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[split].append(loss.item())

    model.train()

    # Calculate mean losses and perplexity
    train_loss = sum(losses["train"]) / len(losses["train"])
    val_loss = sum(losses["val"]) / len(losses["val"])
    train_perplexity = math.exp(train_loss)
    val_perplexity = math.exp(val_loss)

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_perplexity": train_perplexity,
        "val_perplexity": val_perplexity,
    }


if __name__ == "__main__":
    # Hyperparameters
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    block_size = 256
    batch_size = 128
    max_steps = 10000
    learning_rate = 3e-4
    eval_interval = 500
    warmup_steps = 500

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model
    # Using embedding dimension of 384 (same as sentence-transformers/all-MiniLM-L6-v2)
    model = BigramLanguageModel(vocab_size, block_size, n_embd=384)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )

    # Setup mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    use_amp = device == "cuda"

    print(f"\nTraining Configuration:")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Max Steps: {max_steps}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Block Size: {block_size}\n")

    # Training loop
    model.train()
    for steps in range(max_steps):
        xb, yb = get_batch("train", block_size=block_size, batch_size=batch_size, device=device)

        # Mixed precision training
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluation
        if steps % eval_interval == 0:
            metrics = evaluate(model)
            lr = scheduler.get_last_lr()[0]
            print(f"Step {steps:5d} | LR: {lr:.2e} | "
                  f"Train Loss: {metrics['train_loss']:.4f} | Val Loss: {metrics['val_loss']:.4f} | "
                  f"Train PPL: {metrics['train_perplexity']:.2f} | Val PPL: {metrics['val_perplexity']:.2f}")

    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation:")
    final_metrics = evaluate(model, eval_iters=200)
    print(f"  Train Loss: {final_metrics['train_loss']:.4f} | Train Perplexity: {final_metrics['train_perplexity']:.2f}")
    print(f"  Val Loss: {final_metrics['val_loss']:.4f} | Val Perplexity: {final_metrics['val_perplexity']:.2f}")
    print("="*80 + "\n")

    # Generate sample text
    print("Generating sample text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)
    print("\n" + "="*80)
    print("Sample Generated Text:")
    print("="*80)
    print(decode(generated[0].tolist()))
    print("="*80 + "\n")

    # Save model
    model_name = "wikitext_model.pt" if USE_WIKITEXT else "shakespeare_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "block_size": block_size,
            "train_loss": final_metrics['train_loss'],
            "val_loss": final_metrics['val_loss'],
            "train_perplexity": final_metrics['train_perplexity'],
            "val_perplexity": final_metrics['val_perplexity'],
        },
        model_name,
    )

    print(f"Model saved to {model_name}")
