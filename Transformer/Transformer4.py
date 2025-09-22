"""
Advanced Transformer Implementation for a^n b^n a^n Language

This module implements advanced Transformer components for the a^n b^n a^n language
classification task, including positional encoding, custom embeddings, multi-head
self-attention, feed-forward networks, and complete model architecture.

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

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        """
        Multi-head self-attention mechanism

        Args:
            d_model: dimension of the model (embedding dimension)
            num_heads: number of attention heads
            dropout: dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Check if d_model is divisible by num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x, mask=None):
        """
        Apply self-attention mechanism

        Args:
            x: input tensor of shape [batch_size, seq_length, d_model]
            mask: optional mask to prevent attention to certain positions
                  shape [batch_size, seq_length, seq_length]

        Returns:
            output tensor of shape [batch_size, seq_length, d_model]
        """
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        # (batch_size, num_heads, seq_length, head_dim) @ (batch_size, num_heads, head_dim, seq_length)
        # -> (batch_size, num_heads, seq_length, seq_length)
        energy = torch.matmul(q, k.transpose(-2, -1)) / self.scale.to(x.device)

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)

        # Apply dropout
        attention = self.dropout(attention)

        # Compute weighted sum of values
        # (batch_size, num_heads, seq_length, seq_length) @ (batch_size, num_heads, seq_length, head_dim)
        # -> (batch_size, num_heads, seq_length, head_dim)
        output = torch.matmul(attention, v)

        # Reshape back to original dimensions
        # (batch_size, num_heads, seq_length, head_dim) -> (batch_size, seq_length, d_model)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)

        # Final linear projection
        output = self.out_proj(output)

        return output

    def get_attention_weights(self, x, mask=None):
        """
        Compute and return attention weights (for visualization/analysis)

        Args:
            x: input tensor of shape [batch_size, seq_length, d_model]
            mask: optional mask to prevent attention to certain positions

        Returns:
            attention weights of shape [batch_size, num_heads, seq_length, seq_length]
        """
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        energy = torch.matmul(q, k.transpose(-2, -1)) / self.scale.to(x.device)

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)

        return attention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        """
        Feed-forward network used in Transformer

        Args:
            d_model: dimension of the model (input and output)
            d_ff: dimension of the hidden layer (usually larger than d_model)
            dropout: dropout probability
        """
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Apply feed-forward network

        Args:
            x: input tensor of shape [batch_size, seq_length, d_model]

        Returns:
            output tensor of shape [batch_size, seq_length, d_model]
        """
        # First linear layer + activation
        x = self.activation(self.fc1(x))

        # Dropout
        x = self.dropout(x)

        # Second linear layer
        x = self.fc2(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=256, dropout=0.1):
        """
        Complete Transformer block including self-attention and feed-forward network

        Args:
            d_model: dimension of the model (embedding dimension)
            num_heads: number of attention heads
            d_ff: dimension of the feed-forward network's hidden layer
            dropout: dropout probability
        """
        super().__init__()

        # Self-attention layer
        self.self_attention = SelfAttention(d_model, num_heads, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Apply full transformer block

        Args:
            x: input tensor of shape [batch_size, seq_length, d_model]
            mask: optional mask for self-attention

        Returns:
            output tensor of shape [batch_size, seq_length, d_model]
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

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

def demonstrate_self_attention():
    """
    Demonstrate self-attention with 10 examples of a^n b^n a^n strings
    Shows the input embeddings, positional encodings, and self-attention outputs
    """
    print("\n==== DEMONSTRATING SELF-ATTENTION ====")

    # Create 10 examples of a^n b^n a^n strings
    examples = [
        "aabaa",          # a^2 b^1 a^2
        "aaabbbaaa",      # a^3 b^3 a^3
        "aaaabbbbaaaa",   # a^4 b^4 a^4
        "zzzYYYzzz",      # z^3 Y^3 z^3 (using different chars)
        "111222111",      # 1^3 2^3 1^3 (using digits)
        "xxyxx",          # x^2 y^1 x^2
        "gghhhhggg",      # g^2 h^4 g^3
        "aaaabaaaa",      # a^4 b^1 a^4
        "aaabbbbbbbaaa",  # a^3 b^7 a^3
        "xyyyx"           # x^1 y^3 x^1
    ]

    print("\n10 Examples of a^n b^n a^n strings:")
    for i, example in enumerate(examples):
        print(f"{i+1}. {example}")

    # Create models
    d_model = 64
    embedding = AnBnAnEmbedding(d_model=d_model)

    # New self-attention module
    self_attention = SelfAttention(d_model=d_model, num_heads=4)

    # Process all examples as a batch
    print("\nProcessing all examples as a batch...")

    # Get raw token embeddings without positional encoding
    token_embeddings_batch = []
    for example in examples:
        tokens = embedding.tokenize(example)
        # Get raw token embeddings
        token_embeddings = embedding.token_embedding(tokens) * np.sqrt(d_model)
        token_embeddings_batch.append(token_embeddings)

    # Pad embeddings to the same length
    max_len = max(emb.shape[0] for emb in token_embeddings_batch)
    padded_embeddings = torch.zeros((len(examples), max_len, d_model))
    for i, emb in enumerate(token_embeddings_batch):
        padded_embeddings[i, :emb.shape[0]] = emb

    print(f"\nRaw Token Embeddings (without positional encoding):")
    print(f"Shape: {padded_embeddings.shape}")
    # Print the first 5 values of the first token of the first example
    print(f"First example, first token (first 5 values): {padded_embeddings[0, 0, :5].detach().numpy().round(4)}")

    # Apply positional encoding
    pos_encoded = embedding.positional_encoding(padded_embeddings)

    print(f"\nEmbeddings with Positional Encoding:")
    print(f"Shape: {pos_encoded.shape}")
    print(f"First example, first token (first 5 values): {pos_encoded[0, 0, :5].detach().numpy().round(4)}")

    # Apply self-attention
    attention_output = self_attention(pos_encoded)

    print(f"\nOutput after Self-Attention:")
    print(f"Shape: {attention_output.shape}")
    print(f"First example, first token (first 5 values): {attention_output[0, 0, :5].detach().numpy().round(4)}")

    # Retrieve and analyze attention weights for the first example
    attention_weights = self_attention.get_attention_weights(pos_encoded)
    print(f"\nAttention Weights:")
    print(f"Shape: {attention_weights.shape}")
    print(f"Number of attention heads: {attention_weights.shape[1]}")

    # Show the first attention head for the first example
    first_head = attention_weights[0, 0].detach().numpy()
    seq_len = min(8, first_head.shape[0])  # Use at most 8 tokens to keep the output readable

    print(f"\nAttention weights for first example, first head (up to {seq_len}x{seq_len}):")
    for i in range(seq_len):
        row = [f"{v:.2f}" for v in first_head[i, :seq_len]]
        print(f"  {row}")

def demonstrate_transformer_block():
    """
    Demonstrate the complete transformer block (self-attention + feed-forward)
    with 10 examples of a^n b^n a^n strings
    """
    print("\n==== DEMONSTRATING TRANSFORMER BLOCK ====")

    # Create 10 examples of a^n b^n a^n strings
    examples = [
        "aabaa",          # a^2 b^1 a^2
        "aaabbbaaa",      # a^3 b^3 a^3
        "aaaabbbbaaaa",   # a^4 b^4 a^4
        "zzzYYYzzz",      # z^3 Y^3 z^3 (using different chars)
        "111222111",      # 1^3 2^3 1^3 (using digits)
        "xxyxx",          # x^2 y^1 x^2
        "gghhhhggg",      # g^2 h^4 g^3
        "aaaabaaaa",      # a^4 b^1 a^4
        "aaabbbbbbbaaa",  # a^3 b^7 a^3
        "xyyyx"           # x^1 y^3 x^1
    ]

    print("\n10 Examples of a^n b^n a^n strings:")
    for i, example in enumerate(examples):
        print(f"{i+1}. {example}")

    # Create models
    d_model = 64
    embedding = AnBnAnEmbedding(d_model=d_model)

    # Self-attention module and feed-forward network combined in transformer block
    transformer_block = TransformerBlock(d_model=d_model, num_heads=4, d_ff=128)

    # Process all examples as a batch
    print("\nProcessing all examples as a batch through the transformer block...")

    # Get raw token embeddings without positional encoding
    token_embeddings_batch = []
    for example in examples:
        tokens = embedding.tokenize(example)
        # Get raw token embeddings
        token_embeddings = embedding.token_embedding(tokens) * np.sqrt(d_model)
        token_embeddings_batch.append(token_embeddings)

    # Pad embeddings to the same length
    max_len = max(emb.shape[0] for emb in token_embeddings_batch)
    padded_embeddings = torch.zeros((len(examples), max_len, d_model))
    for i, emb in enumerate(token_embeddings_batch):
        padded_embeddings[i, :emb.shape[0]] = emb

    # 1. Display raw token embeddings
    print(f"\n1. Raw Token Embeddings (without positional encoding):")
    print(f"   Shape: {padded_embeddings.shape}")
    print(f"   First example, first token (first 5 values): {padded_embeddings[0, 0, :5].detach().numpy().round(4)}")

    # 2. Apply positional encoding
    pos_encoded = embedding.positional_encoding(padded_embeddings)

    print(f"\n2. Embeddings with Positional Encoding:")
    print(f"   Shape: {pos_encoded.shape}")
    print(f"   First example, first token (first 5 values): {pos_encoded[0, 0, :5].detach().numpy().round(4)}")

    # 3. Extract just the self-attention layer from the transformer block
    self_attention = transformer_block.self_attention
    attention_output = self_attention(pos_encoded)

    print(f"\n3. Output after Self-Attention Layer:")
    print(f"   Shape: {attention_output.shape}")
    print(f"   First example, first token (first 5 values): {attention_output[0, 0, :5].detach().numpy().round(4)}")

    # 4. Extract just the feed-forward network from the transformer block
    feed_forward = transformer_block.feed_forward
    ff_output = feed_forward(attention_output)

    print(f"\n4. Output after Feed-Forward Network:")
    print(f"   Shape: {ff_output.shape}")
    print(f"   First example, first token (first 5 values): {ff_output[0, 0, :5].detach().numpy().round(4)}")

    # 5. Apply the complete transformer block (including residual connections and layer norms)
    transformer_output = transformer_block(pos_encoded)

    print(f"\n5. Output after Complete Transformer Block:")
    print(f"   Shape: {transformer_output.shape}")
    print(f"   First example, first token (first 5 values): {transformer_output[0, 0, :5].detach().numpy().round(4)}")

    # Analyze how the representations change through the network
    print("\n==== REPRESENTATION CHANGES THROUGH THE NETWORK ====")

    # Take the first example and show how a token changes through the network
    example_idx = 0
    token_idx = 0  # First token

    first_token_orig = padded_embeddings[example_idx, token_idx]
    first_token_pos = pos_encoded[example_idx, token_idx]
    first_token_attn = attention_output[example_idx, token_idx]
    first_token_ff = ff_output[example_idx, token_idx]
    first_token_transformer = transformer_output[example_idx, token_idx]

    print(f"\nExample: '{examples[example_idx]}', First Token")
    print(f"Initial embedding norm: {torch.norm(first_token_orig).item():.4f}")
    print(f"After positional encoding norm: {torch.norm(first_token_pos).item():.4f}")
    print(f"After self-attention norm: {torch.norm(first_token_attn).item():.4f}")
    print(f"After feed-forward norm: {torch.norm(first_token_ff).item():.4f}")
    print(f"After complete transformer block norm: {torch.norm(first_token_transformer).item():.4f}")

    # Calculate changes in representation at each step
    pos_change = torch.norm(first_token_pos - first_token_orig).item()
    attn_change = torch.norm(first_token_attn - first_token_pos).item()
    ff_change = torch.norm(first_token_ff - first_token_attn).item()
    transformer_change = torch.norm(first_token_transformer - first_token_orig).item()

    print(f"\nChanges in representation:")
    print(f"Change due to positional encoding: {pos_change:.4f}")
    print(f"Change due to self-attention: {attn_change:.4f}")
    print(f"Change due to feed-forward: {ff_change:.4f}")
    print(f"Overall change: {transformer_change:.4f}")

if __name__ == "__main__":
    test_embedding()
    test_embedding_sizes()
    demonstrate_positional_encoding()
    demonstrate_self_attention()
    demonstrate_transformer_block()