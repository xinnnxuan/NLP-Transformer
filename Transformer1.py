import torch
import torch.nn as nn
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

class AnBnAnEmbedding(nn.Module):
    def __init__(self, d_model=64):
        """
        Custom embedding for a^n b^n a^n language
        d_model: dimension of the embedding vectors
        """
        super().__init__()
        self.d_model = d_model

        # Create learnable embeddings for 'a' and 'b' tokens
        # Adding a padding token (0) and unknown token (3)
        self.token_embedding = nn.Embedding(4, d_model)  # 0=pad, 1=a, 2=b, 3=unk
        self.positional_encoding = PositionalEncoding(d_model)

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

if __name__ == "__main__":
    test_embedding()
