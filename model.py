import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=384, use_pretrained_embeddings=True):
        super().__init__()
        self.block_size = block_size
        self.use_pretrained_embeddings = use_pretrained_embeddings

        # Initialize with pre-trained embeddings from sentence-transformers
        if use_pretrained_embeddings:
            print("Loading pre-trained embeddings from sentence-transformers/all-MiniLM-L6-v2...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # Get the embedding dimension from the pre-trained model
            n_embd = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Using embedding dimension: {n_embd}")

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4, block_size=block_size),
            Block(n_embd, n_head=4, block_size=block_size),
            Block(n_embd, n_head=4, block_size=block_size),
            Block(n_embd, n_head=4, block_size=block_size),
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # print(
        #     f"Debug: T={T}, pos_table_size={self.position_embedding_table.num_embeddings}"
        # )

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            if i % 10 == 0:  # Print every 10 tokens
                print(f"Generating token {i}/{max_new_tokens}")
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
