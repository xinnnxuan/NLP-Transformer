"""
Basic Transformer Implementation for a^n b^n a^n Language

This module implements basic Transformer components for the a^n b^n a^n language
classification task, including positional encoding, custom embeddings, and
multi-head self-attention mechanisms.

Key Skills: PyTorch, Transformers, Attention Mechanisms, Language Classification
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Please install: pip install torch")

import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        """
        Initialize positional encoding
        d_model: dimension of the model
        max_seq_length: maximum length of sequences
        """
        super().__init__()

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer makes the tensor a part of the model but not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: input tensor of shape [batch_size, seq_length, d_model]
        """
        return x + self.pe[:x.size(1)]

    def get_encoding(self, position):
        """
        Get the positional encoding for a specific position
        position: int, the position to get encoding for
        Returns: tensor of shape [d_model]
        """
        return self.pe[position]

class AnBnAnEmbedding(nn.Module):
    def __init__(self, d_model=64, max_seq_length=100):
        """
        Custom embedding for a^n b^n a^n language

        Args:
            d_model: dimension of the embedding vectors
                    - 32: Minimal size, might work for very small n
                    - 64: Default, good balance for n up to ~20
                    - 128: Better for longer sequences or more complex patterns
            max_seq_length: maximum length of input sequences

        The choice of d_model considers:
        1. Minimum information needed:
           - Position information (log2(max_seq_length) bits)
           - Token type (2 bits for a/b/pad/unk)
           - Section information (2 bits for first_a/b/last_a)
        2. Need for redundancy in representation for better learning
        3. Computational efficiency (powers of 2)
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Validate d_model is sufficient for the task
        min_dim = int(np.ceil(np.log2(max_seq_length))) + 4  # position + token + section bits
        if d_model < min_dim:
            raise ValueError(f"d_model={d_model} is too small for max_seq_length={max_seq_length}. "
                           f"Minimum recommended dimension is {min_dim}")

        # Create learnable embeddings
        self.token_embedding = nn.Embedding(4, d_model)  # 0=pad, 1=a, 2=b, 3=unk
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Initialize embeddings to help distinguish token types
        with torch.no_grad():
            # Initialize 'a' tokens to have higher values in first quarter of dimensions
            self.token_embedding.weight[1, :d_model//4] = 1.0
            # Initialize 'b' tokens to have higher values in second quarter
            self.token_embedding.weight[2, d_model//4:d_model//2] = 1.0

    def tokenize(self, text):
        """
        Convert text to token indices
        Returns: tensor of token indices
        """
        # Find the first character (will be 'a')
        a_char = text[0]
        # Find the first different character (will be 'b')
        b_char = next(c for c in text if c != a_char)

        # Convert characters to tokens
        tokens = []
        for char in text:
            if char == a_char:
                tokens.append(1)  # 'a' token
            elif char == b_char:
                tokens.append(2)  # 'b' token
            else:
                tokens.append(3)  # unknown token

        return torch.tensor(tokens)

    def forward(self, text_batch):
        """
        Convert batch of strings to embedded tensors
        text_batch: list of strings
        Returns: tensor of shape [batch_size, seq_length, d_model]
        """
        # Convert texts to token indices
        token_indices = [self.tokenize(text) for text in text_batch]

        # Pad sequences to same length
        max_len = max(len(tokens) for tokens in token_indices)
        padded_indices = torch.zeros((len(text_batch), max_len), dtype=torch.long)
        for i, tokens in enumerate(token_indices):
            padded_indices[i, :len(tokens)] = tokens

        # Get token embeddings
        embeddings = self.token_embedding(padded_indices) * np.sqrt(self.d_model)

        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)

        return embeddings

def test_embedding():
    # Create embedding layer
    embedding = AnBnAnEmbedding(d_model=64)

    # Test cases
    test_strings = [
        "aaabbbaaaa",  # Valid string
        "zzzYYYzzz",   # Valid string with different characters
        "aabbaa",      # Valid string
        "aabaa"        # Invalid string
    ]

    # Process test strings
    embeddings = embedding(test_strings)

    print(f"Input shape: {embeddings.shape}")
    print("\nTest cases:")
    for i, text in enumerate(test_strings):
        tokens = embedding.tokenize(text)
        print(f"\nString: {text}")
        print(f"Tokens: {tokens.tolist()}")
        print(f"Embedding shape for this string: {embeddings[i].shape}")
        print(f"First few values of embedding: {embeddings[i][0][:5]}")

def test_embedding_sizes():
    """
    Test different embedding sizes and analyze their properties
    """
    test_strings = [
        "aaabbbaaa",   # n=3
        "aaaabbbbaaaa" # n=4
    ]

    dimensions = [32, 64, 128]

    for d_model in dimensions:
        print(f"\nTesting d_model = {d_model}")
        print("-" * 40)

        try:
            embedding = AnBnAnEmbedding(d_model=d_model)
            embeddings = embedding(test_strings)

            # Analyze embedding properties
            print(f"Embedding shape: {embeddings.shape}")

            # Calculate average distance between token types
            # Process each string separately to avoid shape mismatch
            for i, text in enumerate(test_strings):
                tokens = embedding.tokenize(text)
                a_mask = (tokens == 1)
                b_mask = (tokens == 2)

                # Make sure we have both token types before calculating
                if a_mask.any() and b_mask.any():
                    a_embeds = embeddings[i, :len(tokens)][a_mask]
                    b_embeds = embeddings[i, :len(tokens)][b_mask]

                    a_b_dist = torch.norm(a_embeds.mean(0) - b_embeds.mean(0))
                    print(f"String '{text}' - Average a-b token distance: {a_b_dist:.4f}")

            # Memory usage
            memory_mb = embeddings.element_size() * embeddings.nelement() / (1024 * 1024)
            print(f"Memory usage: {memory_mb:.2f} MB")

        except ValueError as e:
            print(f"Error: {e}")

def demonstrate_positional_encoding():
    """
    Demonstrate positional encoding with 10 examples
    Shows the token, embedding, positional encoding, and combined representation
    """
    print("\n==== DEMONSTRATING POSITIONAL ENCODING ====")

    # Create embedding layer
    d_model = 64
    embedding = AnBnAnEmbedding(d_model=d_model)
    pos_encoder = embedding.positional_encoding

    # Test with a varied a^n b^n a^n string
    test_string = "aaabbbaaaa"  # a^3 b^3 a^4

    # Get raw token IDs
    tokens = embedding.tokenize(test_string)
    first_char = test_string[0]
    second_char = next(c for c in test_string if c != first_char)

    print(f"\nExample string: {test_string}")
    print(f"Token mapping: '{first_char}' → 1, '{second_char}' → 2")
    print(f"Token sequence: {tokens.tolist()}")

    # Get raw token embeddings (without positional encoding)
    token_embeddings = embedding.token_embedding(tokens)

    # Show examples for each position
    print("\n--- 10 Examples of Positional Encoding ---")
    for pos in range(min(10, len(tokens))):
        token_id = tokens[pos].item()
        token_char = test_string[pos]

        # Get base embedding for this token
        token_emb = token_embeddings[pos]

        # Get positional encoding for this position
        pos_encoding = pos_encoder.get_encoding(pos)

        # Get combined representation
        combined = token_emb + pos_encoding

        # Print information in a clear format
        print(f"\n[Position {pos}] Token: '{token_char}' (ID: {token_id})")
        print(f"  Token Embedding     [:5]: {token_emb[:5].detach().numpy().round(4)}")
        print(f"  + Positional Encoding [:5]: {pos_encoding[:5].detach().numpy().round(4)}")
        print(f"  = Combined Representation [:5]: {combined[:5].detach().numpy().round(4)}")

    # Demonstrate consistency of token embeddings across different strings
    print("\n==== TOKEN CONSISTENCY CHECK ====")
    # Create another string with different characters but same pattern
    test_string2 = "xxxxyyyyxxxx"  # x^4 y^4 x^4
    tokens2 = embedding.tokenize(test_string2)

    # Compare 'a' from first string and 'x' from second string (both map to token 1)
    # Also compare 'b' from first string and 'y' from second (both map to token 2)
    a_embedding = embedding.token_embedding(torch.tensor([1]))[0]
    b_embedding = embedding.token_embedding(torch.tensor([2]))[0]

    print(f"Token 1 ('{first_char}' or 'x') embedding [:5]: {a_embedding[:5].detach().numpy().round(4)}")
    print(f"Token 2 ('{second_char}' or 'y') embedding [:5]: {b_embedding[:5].detach().numpy().round(4)}")

    # Show how position affects the same token
    print("\n==== POSITION EFFECT ON SAME TOKEN ====")
    print(f"Token '{first_char}' at different positions:")

    # Find all positions of the first token type ('a')
    a_positions = [i for i, t in enumerate(tokens) if t.item() == 1][:5]  # limit to 5 examples

    for pos in a_positions:
        pos_encoding = pos_encoder.get_encoding(pos)
        combined = a_embedding + pos_encoding
        print(f"  Position {pos}: {combined[:5].detach().numpy().round(4)}")

if __name__ == "__main__":
    test_embedding()
    test_embedding_sizes()
    demonstrate_positional_encoding()
