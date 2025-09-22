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
        if not text:
            return torch.tensor([])  # Return empty tensor for empty text

        # Find the first character (will be 'a')
        a_char = text[0]

        # Try to find the first different character (will be 'b')
        # If not found, default to a placeholder character
        try:
            b_char = next(c for c in text if c != a_char)
        except StopIteration:
            # If all characters are the same, use a default second character
            # This will only be used for the theoretical mapping but won't appear in the tokens
            b_char = '!' if a_char != '!' else '@'

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

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads=4, d_ff=128, num_layers=2, dropout=0.1):
        """
        Transformer Encoder for a^n b^n a^n language

        Args:
            d_model: dimension of the model
            num_heads: number of attention heads
            d_ff: dimension of feed-forward network
            num_layers: number of transformer blocks
            dropout: dropout rate
        """
        super().__init__()

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Apply transformer encoder

        Args:
            x: input tensor [batch_size, seq_length, d_model]
            mask: optional mask for padding

        Returns:
            output tensor [batch_size, seq_length, d_model]
        """
        # Apply each transformer block in sequence
        for layer in self.layers:
            x = layer(x, mask)

        # Apply final normalization
        x = self.norm(x)

        return x

class AnBnAnClassifier(nn.Module):
    def __init__(self, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout=0.1, max_seq_length=100):
        """
        Complete model for classifying if a string belongs to a^n b^n a^n language

        Args:
            d_model: dimension of the model
            num_heads: number of attention heads
            d_ff: dimension of feed-forward network
            num_layers: number of transformer blocks
            dropout: dropout rate
            max_seq_length: maximum sequence length
        """
        super().__init__()

        # Embedding layer
        self.embedding = AnBnAnEmbedding(d_model, max_seq_length)

        # Encoder stack
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)

        # Classification head
        # First use mean pooling of all token representations
        # Then project to a single output (binary classification)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, text_batch):
        """
        Classify if strings belong to a^n b^n a^n language

        Args:
            text_batch: list of strings to classify

        Returns:
            probabilities tensor [batch_size]
        """
        # Special case for empty strings
        if not text_batch or all(not text for text in text_batch):
            return torch.zeros(len(text_batch) if isinstance(text_batch, list) else 1)

        # Get token embeddings with positional encoding
        x = self.embedding(text_batch)

        # Apply encoder
        encoder_output = self.encoder(x)

        # Mean pooling across sequence length
        # This combines information from all positions into a single vector
        pooled = encoder_output.mean(dim=1)

        # Apply classifier to get probabilities
        probs = self.classifier(pooled).squeeze(-1)

        return probs

def generate_training_data(num_samples=1000, max_n=10, valid_ratio=0.5):
    """
    Generate training data for a^n b^n a^n language

    Args:
        num_samples: total number of samples to generate
        max_n: maximum value of n
        valid_ratio: ratio of valid samples

    Returns:
        texts: list of strings
        labels: list of binary labels (1 for valid, 0 for invalid)
    """
    texts = []
    labels = []

    # Generate valid samples of form a^n b^n a^n
    num_valid = int(num_samples * valid_ratio)
    for _ in range(num_valid):
        n = np.random.randint(1, max_n + 1)
        text = 'a' * n + 'b' * n + 'a' * n
        texts.append(text)
        labels.append(1)

    # Generate invalid samples using various patterns
    num_invalid = num_samples - num_valid
    for i in range(num_invalid):
        n = np.random.randint(1, max_n + 1)
        m = np.random.randint(1, max_n + 1)
        k = np.random.randint(1, max_n + 1)

        # Make sure n, m, k are not all the same (would be valid)
        while n == m and m == k:
            m = np.random.randint(1, max_n + 1)
            k = np.random.randint(1, max_n + 1)

        # Different invalid patterns to ensure variety
        pattern_type = i % 4
        if pattern_type == 0:
            # a^n b^m a^k where n, m, k are not all the same
            text = 'a' * n + 'b' * m + 'a' * k
        elif pattern_type == 1:
            # a^n b^m a^n where m != n
            text = 'a' * n + 'b' * m + 'a' * n
        elif pattern_type == 2:
            # a^n a^n b^n (wrong order)
            text = 'a' * n + 'a' * n + 'b' * n
        else:
            # a^n b^n b^n (wrong order)
            text = 'a' * n + 'b' * n + 'b' * n

        texts.append(text)
        labels.append(0)

    # Shuffle the data
    indices = np.random.permutation(num_samples)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels

def train_model(model, num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train the a^n b^n a^n classifier

    Args:
        model: model to train
        num_epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer

    Returns:
        trained model and training history
    """
    # Set model to training mode
    model.train()

    # Generate training data
    train_texts, train_labels = generate_training_data(num_samples=2000, max_n=15)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Binary cross entropy loss
    criterion = nn.BCELoss()

    # Training history
    history = {
        'loss': [],
        'accuracy': []
    }

    # Number of batches
    num_batches = len(train_texts) // batch_size

    print(f"Training model on {len(train_texts)} examples for {num_epochs} epochs...")

    # Training loop
    for epoch in range(num_epochs):
        # Shuffle data for each epoch
        indices = np.random.permutation(len(train_texts))
        shuffled_texts = [train_texts[i] for i in indices]
        shuffled_labels = [train_labels[i] for i in indices]

        # Track metrics
        epoch_loss = 0
        epoch_correct = 0
        total_samples = 0

        # Process each batch
        for i in range(0, len(shuffled_texts), batch_size):
            # Get batch
            batch_texts = shuffled_texts[i:i+batch_size]
            batch_labels = torch.tensor(shuffled_labels[i:i+batch_size], dtype=torch.float32)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_texts)

            # Calculate loss
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item() * len(batch_texts)
            preds = (outputs > 0.5).float()
            epoch_correct += (preds == batch_labels).sum().item()
            total_samples += len(batch_texts)

        # Calculate epoch metrics
        epoch_loss /= total_samples
        epoch_accuracy = epoch_correct / total_samples

        # Save history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    print("Training complete!")
    return model, history

def generate_test_data(num_samples=2000, max_n=20):
    """
    Generate a large set of test data with various patterns

    Args:
        num_samples: total number of test samples
        max_n: maximum value of n

    Returns:
        texts: list of strings
        labels: list of binary labels (1 for valid, 0 for invalid)
    """
    texts = []
    labels = []

    # Generate 50% valid samples
    num_valid = num_samples // 2
    for _ in range(num_valid):
        n = np.random.randint(1, max_n + 1)
        text = 'a' * n + 'b' * n + 'a' * n
        texts.append(text)
        labels.append(1)

    # Generate invalid samples with various patterns
    remaining = num_samples - num_valid
    pattern_types = 8  # Number of different invalid patterns
    samples_per_pattern = remaining // pattern_types

    for pattern in range(pattern_types):
        for _ in range(samples_per_pattern):
            n = np.random.randint(1, max_n + 1)
            m = np.random.randint(1, max_n + 1)
            k = np.random.randint(1, max_n + 1)

            # Ensure we don't accidentally create valid strings
            while n == m and m == k:
                m = np.random.randint(1, max_n + 1)
                k = np.random.randint(1, max_n + 1)

            if pattern == 0:
                # a^n b^m a^k (all different lengths)
                text = 'a' * n + 'b' * m + 'a' * k
            elif pattern == 1:
                # a^n b^m a^n (middle section wrong)
                text = 'a' * n + 'b' * m + 'a' * n
            elif pattern == 2:
                # a^n a^n b^n (wrong order)
                text = 'a' * n + 'a' * n + 'b' * n
            elif pattern == 3:
                # b^n a^n b^n (wrong order)
                text = 'b' * n + 'a' * n + 'b' * n
            elif pattern == 4:
                # a^n b^n (missing section)
                text = 'a' * n + 'b' * n
            elif pattern == 5:
                # a^n b^n a^n b^n (extra section)
                text = 'a' * n + 'b' * n + 'a' * n + 'b' * n
            elif pattern == 6:
                # a^n (single section)
                text = 'a' * n
            else:
                # Random interleaving
                text = 'a' * (n//2) + 'b' + 'a' * (n//2) + 'b' * (m-1) + 'a' * k

            texts.append(text)
            labels.append(0)

    # Add some edge cases
    edge_cases = ['', 'a', 'b', 'ab', 'ba', 'aaa', 'bbb', 'aba', 'bab']
    texts.extend(edge_cases)
    labels.extend([0] * len(edge_cases))

    # Shuffle the data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels

def evaluate_model_large_scale(model, batch_size=64):
    """
    Evaluate model on a large number of test cases

    Args:
        model: trained model
        batch_size: batch size for evaluation
    """
    model.eval()

    # Generate large test set
    test_texts, test_labels = generate_test_data(num_samples=2000, max_n=20)

    # Track metrics
    total_correct = 0
    total_samples = 0

    # Track performance by string length
    length_metrics = {}

    # Track performance by pattern type
    pattern_metrics = {
        'valid': {'correct': 0, 'total': 0},
        'invalid': {'correct': 0, 'total': 0}
    }

    print("\n==== LARGE-SCALE MODEL EVALUATION ====")
    print(f"Testing on {len(test_texts)} examples...")

    # Process in batches
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            batch_labels = torch.tensor(test_labels[i:i+batch_size], dtype=torch.float32)

            # Get predictions
            outputs = model(batch_texts)
            predictions = (outputs > 0.5).float()

            # Update metrics
            correct = (predictions == batch_labels).float()
            total_correct += correct.sum().item()
            total_samples += len(batch_texts)

            # Update pattern metrics
            for j, (text, label, pred) in enumerate(zip(batch_texts, batch_labels, predictions)):
                pattern_type = 'valid' if label == 1 else 'invalid'
                pattern_metrics[pattern_type]['total'] += 1
                if label == pred:
                    pattern_metrics[pattern_type]['correct'] += 1

                # Track by length
                length = len(text)
                if length not in length_metrics:
                    length_metrics[length] = {'correct': 0, 'total': 0}
                length_metrics[length]['total'] += 1
                if label == pred:
                    length_metrics[length]['correct'] += 1

    # Calculate overall accuracy
    accuracy = total_correct / total_samples

    # Print results
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    print("\nPerformance by Pattern Type:")
    for pattern in pattern_metrics:
        correct = pattern_metrics[pattern]['correct']
        total = pattern_metrics[pattern]['total']
        pat_accuracy = correct / total if total > 0 else 0
        print(f"{pattern.capitalize()}: {pat_accuracy:.4f} ({correct}/{total})")

    print("\nPerformance by String Length:")
    lengths = sorted(length_metrics.keys())
    for length in lengths:
        correct = length_metrics[length]['correct']
        total = length_metrics[length]['total']
        length_accuracy = correct / total if total > 0 else 0
        print(f"Length {length}: {length_accuracy:.4f} ({correct}/{total})")

    return accuracy

def generate_test_data_for_n(n, num_samples=100):
    """
    Generate test data specifically for strings with length parameter n

    Args:
        n: the specific length parameter for a^n b^n a^n
        num_samples: total number of samples (half valid, half invalid)

    Returns:
        texts: list of strings
        labels: list of binary labels (1 for valid, 0 for invalid)
    """
    texts = []
    labels = []

    # Generate valid samples (all with exactly n)
    num_valid = num_samples // 2
    valid_string = 'a' * n + 'b' * n + 'a' * n
    texts.extend([valid_string] * num_valid)
    labels.extend([1] * num_valid)

    # Generate invalid samples
    num_invalid = num_samples - num_valid
    invalid_patterns = [
        lambda n: 'a' * (n+1) + 'b' * n + 'a' * n,      # First section too long
        lambda n: 'a' * n + 'b' * (n+1) + 'a' * n,      # Middle section too long
        lambda n: 'a' * n + 'b' * n + 'a' * (n+1),      # Last section too long
        lambda n: 'a' * (n-1) + 'b' * n + 'a' * n if n > 1 else 'ba',  # First section too short
        lambda n: 'a' * n + 'b' * (n-1) + 'a' * n if n > 1 else 'ab',  # Middle section too short
        lambda n: 'a' * n + 'b' * n + 'a' * (n-1) if n > 1 else 'aa',  # Last section too short
        lambda n: 'b' * n + 'a' * n + 'b' * n,          # Wrong order
        lambda n: 'a' * n + 'a' * n + 'b' * n,          # Wrong order
        lambda n: 'a' * n + 'b' * n,                    # Missing section
        lambda n: 'a' * n + 'b' * n + 'a' * n + 'b' * n # Extra section
    ]

    samples_per_pattern = num_invalid // len(invalid_patterns)
    remaining = num_invalid % len(invalid_patterns)

    for pattern in invalid_patterns:
        num_samples = samples_per_pattern + (1 if remaining > 0 else 0)
        remaining -= 1
        texts.extend([pattern(n)] * num_samples)
        labels.extend([0] * num_samples)

    # Shuffle the data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels

def test_N(model, n, batch_size=32):
    """
    Test the model specifically on strings with length parameter n

    Args:
        model: trained AnBnAnClassifier model
        n: the specific length parameter for a^n b^n a^n
        batch_size: batch size for evaluation
    """
    model.eval()

    # Generate test data for specific n
    test_texts, test_labels = generate_test_data_for_n(n)

    print(f"\n==== TESTING MODEL ON STRINGS WITH n={n} ====")
    print(f"Testing on {len(test_texts)} examples ({len(test_texts)//2} valid, {len(test_texts)//2} invalid)")

    # Track metrics
    total_correct = 0

    # Process in batches
    results = []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            batch_labels = torch.tensor(test_labels[i:i+batch_size], dtype=torch.float32)

            # Get predictions
            outputs = model(batch_texts)
            predictions = (outputs > 0.5).float()

            # Store results
            for text, label, pred, conf in zip(batch_texts, batch_labels, predictions, outputs):
                results.append({
                    'text': text,
                    'true_label': int(label.item()),
                    'predicted': int(pred.item()),
                    'confidence': float(conf.item()),
                    'correct': label.item() == pred.item()
                })
                if label.item() == pred.item():
                    total_correct += 1

    # Calculate accuracy
    accuracy = total_correct / len(test_texts)

    # Print results
    print("\nDetailed Results:")
    print(f"{'String':<{3*n+10}} {'True':<6} {'Pred':<6} {'Conf':<8} {'Result'}")
    print("-" * (3*n + 35))

    for result in results:
        text_display = result['text'][:40] + "..." if len(result['text']) > 40 else result['text']
        status = "✓" if result['correct'] else "✗"
        print(f"{text_display:<{3*n+10}} {result['true_label']:<6} {result['predicted']:<6} "
              f"{result['confidence']:.4f}  {status}")

    # Print summary
    print(f"\nSummary for n={n}:")
    print(f"Overall Accuracy: {accuracy:.4f} ({total_correct}/{len(test_texts)})")

    # Calculate separate accuracies for valid and invalid examples
    valid_results = [r for r in results if r['true_label'] == 1]
    invalid_results = [r for r in results if r['true_label'] == 0]

    valid_accuracy = sum(1 for r in valid_results if r['correct']) / len(valid_results)
    invalid_accuracy = sum(1 for r in invalid_results if r['correct']) / len(invalid_results)

    print(f"Valid String Accuracy: {valid_accuracy:.4f} "
          f"({sum(1 for r in valid_results if r['correct'])}/{len(valid_results)})")
    print(f"Invalid String Accuracy: {invalid_accuracy:.4f} "
          f"({sum(1 for r in invalid_results if r['correct'])}/{len(invalid_results)})")

    return accuracy

def demonstrate_anbnan_classifier():
    """
    Demonstrate the complete a^n b^n a^n language classifier with improved architecture
    """
    print("\n==== IMPROVED a^n b^n a^n LANGUAGE CLASSIFIER ====")

    # Create model with enhanced architecture
    model = AnBnAnClassifier(
        d_model=128,          # Increased from 64 to 128
        num_heads=8,          # Increased from 4 to 8 heads
        d_ff=256,            # Increased from 128 to 256
        num_layers=3,        # Increased from 2 to 3 layers
        dropout=0.2          # Increased dropout for better regularization
    )

    # Train model with more epochs and smaller batch size
    model, history = train_model(
        model=model,
        num_epochs=10,       # Increased from 5 to 10
        batch_size=16,       # Decreased from 32 to 16
        learning_rate=0.0005  # Decreased from 0.001 to 0.0005
    )

    # Evaluate model on large-scale test set
    print("\nEvaluating on large-scale test set:")
    large_scale_accuracy = evaluate_model_large_scale(model)

    # Test model on specific values of n
    test_ns = [1, 2, 3, 5, 10]
    n_accuracies = {}
    for n in test_ns:
        n_accuracies[n] = test_N(model, n)

    # Print final summary
    print("\n==== FINAL SUMMARY ====")
    print(f"Training Accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Large-Scale Test Accuracy: {large_scale_accuracy:.4f}")
    print("\nAccuracy by n value:")
    for n, acc in n_accuracies.items():
        print(f"n={n}: {acc:.4f}")

    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        # Training metrics
        plt.subplot(1, 3, 1)
        plt.plot(history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 2)
        plt.plot(history['accuracy'])
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        # Accuracy by n
        plt.subplot(1, 3, 3)
        ns = list(n_accuracies.keys())
        accs = list(n_accuracies.values())
        plt.plot(ns, accs, 'bo-')
        plt.title('Accuracy by n')
        plt.xlabel('n')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")

if __name__ == "__main__":
    demonstrate_anbnan_classifier()

    model = AnBnAnClassifier(d_model=64, num_heads=4, num_layers=1)
    model, history = train_model(model, num_epochs=3, batch_size=16)
    evaluate_model_large_scale(model)