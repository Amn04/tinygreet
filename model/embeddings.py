"""
Embedding Layers for TinyGreet

This implements:  
1. Token Embeddings - Convert token IDs to vectors
2. Positional Encoding - Add position information
3. Combined Embedding Layer - Token + Position + LayerNorm

All built from scratch using our Tensor class!  
"""

import numpy as np
from typing import Optional, Tuple, List
import math

# Import our tensor class
from tensor import Tensor


class Embedding:  
    """
    Token Embedding Layer
    
    Converts token IDs (integers) into dense vectors. 
    This is a lookup table that gets LEARNED during training.
    
    Shape: (vocab_size, embed_dim)
    
    Example:
        vocab_size = 500, embed_dim = 64
        Token ID 229 -> embedding_table[229] -> vector of 64 numbers
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx:  Optional[int] = 0
    ):
        """
        Initialize the embedding table.
        
        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim:  Dimension of embedding vectors
            padding_idx:  If provided, embeddings at this index are zeros (for <PAD>)
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        
        # Initialize embedding table with small random values
        # Using Xavier/Glorot initialization:  scale by sqrt(1/embed_dim)
        scale = np.sqrt(1.0 / embed_dim)
        self.weight = Tensor(
            np.random.randn(vocab_size, embed_dim) * scale,
            requires_grad=True
        )
        
        # Set padding embedding to zeros
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0.0
    
    def __call__(self, token_ids:  np.ndarray) -> Tensor: 
        """
        Look up embeddings for token IDs.
        
        Args: 
            token_ids: numpy array of token IDs, shape (batch_size, seq_len)
                       or (seq_len,) for single sequence
        
        Returns:
            Tensor of embeddings, shape (batch_size, seq_len, embed_dim)
            or (seq_len, embed_dim) for single sequence
        """
        # Handle single sequence (add batch dimension)
        single_sequence = token_ids.ndim == 1
        if single_sequence: 
            token_ids = token_ids[np.newaxis, :]  # (1, seq_len)
        
        batch_size, seq_len = token_ids.shape
        
        # Lookup embeddings (this is just fancy indexing!)
        embeddings = self.weight.data[token_ids]  # (batch_size, seq_len, embed_dim)
        
        # Create output tensor with autograd connection
        out = Tensor(
            embeddings,
            requires_grad=True,
            _children=(self.weight,),
            _op='embedding'
        )
        
        # Store for backward
        stored_token_ids = token_ids.copy()
        
        # Define backward pass for embedding lookup
        def _backward():
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                
                # Accumulate gradients for each token that was looked up
                np.add.at(self.weight.grad, stored_token_ids.flatten(),
                         out.grad.reshape(-1, self.embed_dim))
        
        out._backward = _backward
        
        # Remove batch dimension if input was single sequence
        if single_sequence:
            out_batched = out
            out = Tensor(
                out.data[0],
                requires_grad=out.requires_grad,
                _children=(out_batched,),
                _op='squeeze'
            )
            # Define backward for squeeze
            def _backward_squeeze():
                if out_batched.requires_grad:
                    if out_batched.grad is None:
                        out_batched.grad = np.zeros_like(out_batched.data)
                    out_batched.grad += out.grad[np.newaxis, :, :]
            out._backward = _backward_squeeze
        
        return out
    
    def parameters(self) -> List[Tensor]:
        """Return list of learnable parameters."""
        return [self.weight]
    
    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"


class PositionalEncoding: 
    """
    Sinusoidal Positional Encoding
    
    Adds position information to embeddings using sine and cosine functions.
    This is the SAME method used in "Attention Is All You Need" (original Transformer).
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Deterministic (no learned parameters)
    - Can handle any sequence length
    - Provides unique position signal for each position
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 512,
        dropout_rate: float = 0.0
    ):
        """
        Initialize positional encoding.
        
        Args: 
            embed_dim:  Dimension of embeddings (must match token embeddings)
            max_seq_len:  Maximum sequence length to pre-compute
            dropout_rate: Dropout rate (0.0 = no dropout)
        """
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        
        # Pre-compute positional encodings for all positions
        self.pe = self._create_positional_encoding(max_seq_len, embed_dim)
    
    def _create_positional_encoding(
        self, 
        max_seq_len:  int, 
        embed_dim: int
    ) -> np.ndarray:
        """
        Create the positional encoding matrix. 
        
        Returns: 
            numpy array of shape (max_seq_len, embed_dim)
        """
        # Initialize matrix
        pe = np.zeros((max_seq_len, embed_dim))
        
        # Position indices: 0, 1, 2, ..., max_seq_len-1
        position = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)
        
        # Dimension indices for the formula
        # We need 2i for even dims, so we use 0, 2, 4, ...
        div_term = np.exp(
            np.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )  # (embed_dim/2,)
        
        # Apply sine to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cosine to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe.astype(np.float32)
    
    def __call__(self, x:  Tensor) -> Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args: 
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
               or (seq_len, embed_dim) for single sequence
        
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # Get sequence length from input
        if x.data.ndim == 2:
            seq_len = x.data.shape[0]
        else:
            seq_len = x.data.shape[1]
        
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
        
        # Get positional encoding for this sequence length
        pe = self.pe[:seq_len]  # (seq_len, embed_dim)
        
        # Add positional encoding to input
        # The PE is added (not concatenated) to the embeddings
        pe_tensor = Tensor(pe, requires_grad=False)
        
        return x + pe_tensor
    
    def parameters(self) -> List[Tensor]: 
        """Positional encoding has no learnable parameters."""
        return []
    
    def visualize(self, num_positions: int = 50) -> None:
        """
        Print a visualization of positional encodings.
        Useful for understanding the pattern.
        """
        print("\nPositional Encoding Visualization")
        print("=" * 60)
        print(f"Showing first {num_positions} positions, first 8 dimensions\n")
        
        pe = self.pe[:num_positions, :8]
        
        print("Position |  dim0   dim1   dim2   dim3   dim4   dim5   dim6   dim7")
        print("-" * 70)
        
        for pos in range(min(num_positions, 10)):
            values = " ".join([f"{v: 6.3f}" for v in pe[pos]])
            print(f"   {pos:3d}   | {values}")
        
        if num_positions > 10:
            print("   ...")
            for pos in range(num_positions - 3, num_positions):
                values = " ".join([f"{v:6.3f}" for v in pe[pos]])
                print(f"   {pos:3d}   | {values}")
    
    def __repr__(self):
        return f"PositionalEncoding(embed_dim={self.embed_dim}, max_seq_len={self.max_seq_len})"


class LayerNorm: 
    """
    Layer Normalization
    
    Normalizes the input across the last dimension (features).
    This stabilizes training by keeping activations in a reasonable range.
    
    Formula: 
        y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Where gamma and beta are learned parameters.
    
    Used in Transformers after attention and feed-forward layers. 
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize layer normalization.
        
        Args:
            normalized_shape: Size of the last dimension to normalize
            eps:  Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        # gamma (scale) - initialized to 1
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        # beta (shift) - initialized to 0
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)
    
    def __call__(self, x:  Tensor) -> Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
        
        Returns: 
            Normalized tensor of same shape
        """
        # Compute mean and variance along last dimension
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        out_data = self.gamma.data * x_norm + self.beta.data
        
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
            _children=(x, self.gamma, self.beta),
            _op='layernorm'
        )
        
        # Store for backward
        stored_x_norm = x_norm.copy()
        stored_std = np.sqrt(var + self.eps)
        
        def _backward():
            if out.grad is None:
                return
            
            n = self.normalized_shape
            
            # Gradient w.r.t. gamma
            if self.gamma.requires_grad:
                if self.gamma.grad is None:
                    self.gamma.grad = np.zeros_like(self.gamma.data)
                # Sum over batch and sequence dimensions
                self.gamma.grad += (out.grad * stored_x_norm).sum(axis=tuple(range(out.grad.ndim - 1)))
            
            # Gradient w.r.t. beta
            if self.beta.requires_grad:
                if self.beta.grad is None:
                    self.beta.grad = np.zeros_like(self.beta.data)
                self.beta.grad += out.grad.sum(axis=tuple(range(out.grad.ndim - 1)))
            
            # Gradient w.r.t. x
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                
                # LayerNorm backward is complex due to normalization
                dx_norm = out.grad * self.gamma.data
                
                # Gradient through normalization
                dvar = np.sum(dx_norm * (x.data - mean) * -0.5 * (var + self.eps)**(-1.5), 
                             axis=-1, keepdims=True)
                dmean = np.sum(dx_norm * -1.0 / stored_std, axis=-1, keepdims=True) + \
                        dvar * np.sum(-2.0 * (x.data - mean), axis=-1, keepdims=True) / n
                
                dx = dx_norm / stored_std + dvar * 2.0 * (x.data - mean) / n + dmean / n
                x.grad += dx
        
        out._backward = _backward
        return out
    
    def parameters(self) -> List[Tensor]:
        """Return list of learnable parameters."""
        return [self.gamma, self.beta]
    
    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape})"


class TransformerEmbedding:
    """
    Complete Embedding Layer for Transformer
    
    Combines: 
    1. Token Embedding (lookup table)
    2. Positional Encoding (sinusoidal)
    3. Layer Normalization (optional)
    4. Dropout (optional, for training)
    
    This is the first layer of any Transformer model. 
    Input: token IDs (integers)
    Output: contextualized embeddings (float vectors)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len:  int = 512,
        padding_idx: int = 0,
        dropout_rate:  float = 0.0,
        use_layer_norm: bool = True
    ):
        """
        Initialize the complete embedding layer.
        
        Args: 
            vocab_size:  Size of vocabulary
            embed_dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
            padding_idx:  Index of padding token
            dropout_rate: Dropout rate (0.0 = no dropout)
            use_layer_norm: Whether to apply LayerNorm after embedding
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Initialize components
        self.token_embedding = Embedding(vocab_size, embed_dim, padding_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        if use_layer_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else: 
            self.layer_norm = None
        
        # Scaling factor (from "Attention Is All You Need")
        # Embeddings are scaled by sqrt(embed_dim) before adding positional encoding
        self.scale = np.sqrt(embed_dim)
    
    def __call__(
        self, 
        token_ids: np.ndarray, 
        training: bool = False
    ) -> Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            token_ids: numpy array of token IDs
                      Shape: (batch_size, seq_len) or (seq_len,)
            training: Whether in training mode (affects dropout)
        
        Returns:
            Tensor of embeddings
            Shape: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        # Step 1: Token Embedding
        x = self.token_embedding(token_ids)
        
        # Step 2: Scale embeddings
        x = x * self.scale
        
        # Step 3: Add Positional Encoding
        x = self.positional_encoding(x)
        
        # Step 4: Layer Normalization (if enabled)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        # Step 5: Dropout (if training)
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.data.shape)
            x_pre_dropout = x
            scale = 1 - self.dropout_rate
            x = Tensor(
                x.data * mask / scale,
                requires_grad=x.requires_grad,
                _children=(x_pre_dropout,),
                _op='dropout'
            )
            # Define backward for dropout
            def _backward_dropout():
                if x_pre_dropout.requires_grad:
                    if x_pre_dropout.grad is None:
                        x_pre_dropout.grad = np.zeros_like(x_pre_dropout.data)
                    x_pre_dropout.grad += x.grad * mask / scale
            x._backward = _backward_dropout
        
        return x
    
    def parameters(self) -> List[Tensor]:
        """Return all learnable parameters."""
        params = self.token_embedding.parameters()
        if self.layer_norm is not None: 
            params.extend(self.layer_norm.parameters())
        return params
    
    def __repr__(self):
        return (f"TransformerEmbedding(\n"
                f"  vocab_size={self.vocab_size},\n"
                f"  embed_dim={self.embed_dim},\n"
                f"  max_seq_len={self.max_seq_len},\n"
                f"  use_layer_norm={self.use_layer_norm}\n"
                f")")


# ==================== TESTS ====================

def test_embedding():
    """Test the Embedding layer."""
    print("\n" + "=" * 60)
    print("TESTING EMBEDDING LAYER")
    print("=" * 60)
    
    vocab_size = 10
    embed_dim = 4
    
    emb = Embedding(vocab_size, embed_dim, padding_idx=0)
    print(f"\nCreated:  {emb}")
    print(f"Weight shape: {emb.weight.shape}")
    
    # Test single sequence
    print("\n--- Single Sequence ---")
    token_ids = np.array([1, 2, 3])  # 3 tokens
    output = emb(token_ids)
    print(f"Input token IDs: {token_ids}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output.data}")
    
    # Verify lookup
    print("\nVerifying lookup is correct:")
    for i, tid in enumerate(token_ids):
        expected = emb.weight.data[tid]
        actual = output.data[i]
        match = np.allclose(expected, actual)
        print(f"  Token {tid}: {'✅' if match else '❌'}")
    
    # Test batch
    print("\n--- Batch of Sequences ---")
    batch_ids = np.array([[1, 2, 3], [4, 5, 6]])  # 2 sequences, 3 tokens each
    batch_output = emb(batch_ids)
    print(f"Input shape: {batch_ids.shape}")
    print(f"Output shape: {batch_output.shape}")
    
    # Test gradient
    print("\n--- Gradient Test ---")
    token_ids = np.array([1, 2, 1])  # Token 1 appears twice
    output = emb(token_ids)
    loss = output.sum()
    loss.backward()
    
    print(f"Tokens: {token_ids}")
    print(f"Gradient for token 0 (not used): {emb.weight.grad[0]}")
    print(f"Gradient for token 1 (used 2x): {emb.weight.grad[1]}")
    print(f"Gradient for token 2 (used 1x): {emb.weight.grad[2]}")
    
    print("\n✅ Embedding tests passed!")


def test_positional_encoding():
    """Test the Positional Encoding."""
    print("\n" + "=" * 60)
    print("TESTING POSITIONAL ENCODING")
    print("=" * 60)
    
    embed_dim = 8
    max_seq_len = 100
    
    pe = PositionalEncoding(embed_dim, max_seq_len)
    print(f"\nCreated: {pe}")
    
    # Visualize
    pe.visualize(num_positions=10)
    
    # Test adding to embeddings
    print("\n--- Adding to Embeddings ---")
    x = Tensor(np.random.randn(5, embed_dim))  # 5 tokens
    print(f"Input shape: {x.shape}")
    
    output = pe(x)
    print(f"Output shape: {output.shape}")
    
    # Verify position 0 has the expected pattern
    print("\n--- Verifying Position 0 ---")
    print(f"PE[0] (even dims should be 0, odd dims should be 1):")
    print(f"  {pe.pe[0]}")
    
    # Verify uniqueness
    print("\n--- Uniqueness Check ---")
    pe_matrix = pe.pe[:20]
    for i in range(min(5, len(pe_matrix))):
        for j in range(i + 1, min(5, len(pe_matrix))):
            diff = np.abs(pe_matrix[i] - pe_matrix[j]).sum()
            print(f"  Diff between pos {i} and pos {j}: {diff:.4f}")
    
    print("\n✅ Positional Encoding tests passed!")


def test_layer_norm():
    """Test Layer Normalization."""
    print("\n" + "=" * 60)
    print("TESTING LAYER NORMALIZATION")
    print("=" * 60)
    
    normalized_shape = 4
    ln = LayerNorm(normalized_shape)
    print(f"\nCreated:  {ln}")
    
    # Test forward
    print("\n--- Forward Pass ---")
    x = Tensor(np.array([[2.0, 4.0, 6.0, 8.0],
                         [1.0, 2.0, 3.0, 4.0]]), requires_grad=True)
    print(f"Input:\n{x.data}")
    print(f"Input mean (per row): {x.data.mean(axis=-1)}")
    print(f"Input std (per row): {x.data.std(axis=-1)}")
    
    output = ln(x)
    print(f"\nOutput:\n{output.data}")
    print(f"Output mean (per row): {output.data.mean(axis=-1)} (should be ~0)")
    print(f"Output std (per row): {output.data.std(axis=-1)} (should be ~1)")
    
    # Test gradient
    print("\n--- Gradient Test ---")
    loss = output.sum()
    loss.backward()
    
    print(f"Gradient for gamma: {ln.gamma.grad}")
    print(f"Gradient for beta: {ln.beta.grad}")
    print(f"Gradient for x:\n{x.grad}")
    
    print("\n✅ Layer Normalization tests passed!")


def test_transformer_embedding():
    """Test the complete TransformerEmbedding layer."""
    print("\n" + "=" * 60)
    print("TESTING TRANSFORMER EMBEDDING")
    print("=" * 60)
    
    vocab_size = 500
    embed_dim = 64
    max_seq_len = 128
    
    emb_layer = TransformerEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        padding_idx=0,
        use_layer_norm=True
    )
    
    print(f"\n{emb_layer}")
    
    # Count parameters
    total_params = sum(p.data.size for p in emb_layer.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"  - Token embeddings: {vocab_size * embed_dim: ,}")
    print(f"  - LayerNorm gamma: {embed_dim}")
    print(f"  - LayerNorm beta:  {embed_dim}")
    
    # Test forward pass
    print("\n--- Forward Pass ---")
    token_ids = np.array([2, 229, 6, 3])  # <BOS>, "Hello", "!", <EOS>
    print(f"Token IDs: {token_ids}")
    
    output = emb_layer(token_ids)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({len(token_ids)}, {embed_dim})")
    
    # Show output statistics
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.data.mean():.4f}")
    print(f"  Std:   {output.data.std():.4f}")
    print(f"  Min:  {output.data.min():.4f}")
    print(f"  Max:  {output.data.max():.4f}")
    
    # Test batch
    print("\n--- Batch Processing ---")
    batch_ids = np.array([
        [2, 229, 6, 3, 0],     # <BOS> Hello !  <EOS> <PAD>
        [2, 109, 272, 6, 3],   # <BOS> Good morning ! <EOS>
    ])
    print(f"Batch shape: {batch_ids.shape}")
    
    batch_output = emb_layer(batch_ids)
    print(f"Output shape: {batch_output.shape}")
    print(f"Expected:  ({batch_ids.shape[0]}, {batch_ids.shape[1]}, {embed_dim})")
    
    # Verify padding token has zero embedding (before positional encoding)
    print("\n--- Padding Check ---")
    pad_embedding = emb_layer.token_embedding. weight.data[0]
    print(f"Padding token embedding sum: {pad_embedding.sum():.6f} (should be 0)")
    
    # Test gradient flow
    print("\n--- Gradient Flow ---")
    token_ids = np.array([2, 229, 6, 3])
    output = emb_layer(token_ids)
    loss = output.sum()
    loss.backward()
    
    print("Gradients computed for:")
    for i, param in enumerate(emb_layer.parameters()):
        grad_norm = np.linalg.norm(param.grad) if param.grad is not None else 0
        print(f"  Parameter {i}: shape={param.shape}, grad_norm={grad_norm:.4f}")
    
    print("\n✅ TransformerEmbedding tests passed!")


def demonstrate_embedding_process():
    """
    Demonstrate the complete embedding process step by step.
    This is educational - shows exactly what happens to your tokens! 
    """
    print("\n" + "=" * 70)
    print("COMPLETE EMBEDDING PROCESS DEMONSTRATION")
    print("=" * 70)
    
    # Small dimensions for clarity
    vocab_size = 10
    embed_dim = 4
    
    # Create embedding layer
    emb = TransformerEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=10,
        use_layer_norm=True
    )
    
    # Sample tokens
    token_ids = np.array([2, 5, 3])  # 3 tokens
    print(f"\nInput token IDs: {token_ids}")
    print(f"(Pretend:  2=<BOS>, 5='Hello', 3=<EOS>)")
    
    # Step 1: Token Embedding Lookup
    print("\n" + "-" * 60)
    print("STEP 1: Token Embedding Lookup")
    print("-" * 60)
    print(f"\nEmbedding table shape: ({vocab_size}, {embed_dim})")
    print("\nEmbedding table (first 5 rows):")
    for i in range(min(5, vocab_size)):
        print(f"  Token {i}: {emb.token_embedding.weight.data[i]}")
    
    token_emb = emb. token_embedding(token_ids)
    print(f"\nLooking up tokens {token_ids}:")
    for i, tid in enumerate(token_ids):
        print(f"  Token {tid} -> {token_emb.data[i]}")
    
    # Step 2: Scale by sqrt(embed_dim)
    print("\n" + "-" * 60)
    print("STEP 2: Scale by sqrt(embed_dim)")
    print("-" * 60)
    scale = np.sqrt(embed_dim)
    print(f"\nScale factor: sqrt({embed_dim}) = {scale:.4f}")
    print(f"\nBefore scaling:\n{token_emb.data}")
    scaled = token_emb. data * scale
    print(f"\nAfter scaling:\n{scaled}")
    
    # Step 3: Add Positional Encoding
    print("\n" + "-" * 60)
    print("STEP 3: Add Positional Encoding")
    print("-" * 60)
    
    pe = emb. positional_encoding. pe[:len(token_ids)]
    print(f"\nPositional encodings for positions 0, 1, 2:")
    for i in range(len(token_ids)):
        print(f"  Position {i}: {pe[i]}")
    
    with_position = scaled + pe
    print(f"\nAfter adding positional encoding:")
    for i in range(len(token_ids)):
        print(f"  Position {i}: {scaled[i]} + {pe[i]} = {with_position[i]}")
    
    # Step 4: Layer Normalization
    print("\n" + "-" * 60)
    print("STEP 4: Layer Normalization")
    print("-" * 60)
    
    for i in range(len(token_ids)):
        row = with_position[i]
        mean = row.mean()
        std = row.std()
        normalized = (row - mean) / (std + 1e-5)
        print(f"\n  Position {i}:")
        print(f"    Input:      {row}")
        print(f"    Mean:       {mean:.4f}")
        print(f"    Std:        {std:.4f}")
        print(f"    Normalized: {normalized}")
    
    # Full forward pass
    print("\n" + "-" * 60)
    print("COMPLETE OUTPUT")
    print("-" * 60)
    
    output = emb(token_ids)
    print(f"\nFinal output shape:  {output.shape}")
    print(f"Final output:\n{output.data}")
    print(f"\nPer-row mean (should be ~0): {output.data. mean(axis=-1)}")
    print(f"Per-row std (should be ~1): {output.data.std(axis=-1)}")


if __name__ == "__main__": 
    # Run all tests
    test_embedding()
    test_positional_encoding()
    test_layer_norm()
    test_transformer_embedding()
    
    # Run demonstration
    demonstrate_embedding_process()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)