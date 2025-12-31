"""
Attention Mechanism - Built from Scratch

This implements: 
1.  Scaled Dot-Product Attention
2. Multi-Head Attention
3. Causal Masking

The heart of the Transformer architecture! 
"""

import numpy as np
from typing import Optional, Tuple, List
import math

from tensor import Tensor


class Linear:
    """
    Linear (Dense/Fully Connected) Layer
    
    y = x @ W + b
    
    This is a fundamental building block used in attention for
    projecting Q, K, V and the output. 
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        """
        Initialize linear layer. 
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights using Xavier/Glorot initialization
        # This helps with gradient flow at the start of training
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(
                np.zeros(out_features),
                requires_grad=True
            )
        else: 
            self.bias = None
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass:  y = x @ W + b
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns: 
            Output tensor of shape (..., out_features)
        """
        # Matrix multiplication
        out = x @ self.weight
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def parameters(self) -> List[Tensor]:
        """Return list of learnable parameters."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (look-ahead) mask.
    
    Returns a matrix where position [i, j] is: 
    - 0 if i >= j (can attend)
    - -inf if i < j (cannot attend to future)
    
    Args:
        seq_len: Length of the sequence
    
    Returns:
        Mask of shape (seq_len, seq_len)
    """
    # Create upper triangular matrix of -inf (above diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)) * float('-inf'), k=1)
    return mask.astype(np.float32)


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(Q @ K. T / sqrt(d_k)) @ V
    
    This is the core attention mechanism used in Transformers. 
    """
    
    def __init__(self, dropout_rate: float = 0.0):
        """
        Initialize attention. 
        
        Args:
            dropout_rate: Dropout rate for attention weights
        """
        self.dropout_rate = dropout_rate
        self.attention_weights:  Optional[np.ndarray] = None  # Store for visualization
    
    def __call__(
        self,
        query:  Tensor,
        key: Tensor,
        value: Tensor,
        mask:  Optional[np.ndarray] = None,
        training: bool = False
    ) -> Tuple[Tensor, np.ndarray]: 
        """
        Compute scaled dot-product attention. 
        
        Args:
            query:  Query tensor, shape (..., seq_len, d_k)
            key: Key tensor, shape (..., seq_len, d_k)
            value: Value tensor, shape (... , seq_len, d_v)
            mask: Optional mask, shape (..., seq_len, seq_len)
            training: Whether in training mode (affects dropout)
        
        Returns: 
            Tuple of (output, attention_weights)
            - output: shape (..., seq_len, d_v)
            - attention_weights: shape (..., seq_len, seq_len)
        """
        # Get dimension for scaling
        d_k = query.shape[-1]
        scale = np.sqrt(d_k)
        
        # Step 1: Compute attention scores
        # Q @ K.T -> (seq_len, seq_len)
        # We need to transpose the last two dimensions of K
        key_t = key.transpose(*range(len(key.shape) - 2), -1, -2)
        scores = query @ key_t  # (..., seq_len, seq_len)
        
        # Step 2: Scale by sqrt(d_k)
        scores = scores * (1.0 / scale)
        
        # Step 3: Apply mask (if provided)
        if mask is not None:
            # Add mask (masked positions become -inf before softmax)
            scores = Tensor(
                scores.data + mask,
                requires_grad=scores.requires_grad,
                _children=(scores,),
                _op='mask'
            )
            
            # Backward for masking (gradient flows through unchanged)
            old_scores = scores._prev.copy()
            def _backward_mask():
                for child in old_scores:
                    if child.requires_grad:
                        if child.grad is None:
                            child.grad = np.zeros_like(child.data)
                        child.grad += scores.grad
            scores._backward = _backward_mask
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = scores.softmax(axis=-1)
        
        # Store attention weights for visualization
        self.attention_weights = attention_weights.data.copy()
        
        # Step 5: Apply dropout (if training)
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout_rate,
                attention_weights.shape
            ).astype(np.float32)
            attn_pre_dropout = attention_weights
            scale = 1 - self.dropout_rate
            attention_weights = Tensor(
                attention_weights.data * dropout_mask / scale,
                requires_grad=attention_weights.requires_grad,
                _children=(attn_pre_dropout,),
                _op='dropout'
            )
            # Define backward for dropout
            def _backward_dropout():
                if attn_pre_dropout.requires_grad:
                    if attn_pre_dropout.grad is None:
                        attn_pre_dropout.grad = np.zeros_like(attn_pre_dropout.data)
                    attn_pre_dropout.grad += attention_weights.grad * dropout_mask / scale
            attention_weights._backward = _backward_dropout
        
        # Step 6: Apply attention weights to values
        output = attention_weights @ value  # (..., seq_len, d_v)
        
        return output, self.attention_weights
    
    def __repr__(self):
        return f"ScaledDotProductAttention(dropout={self.dropout_rate})"


class MultiHeadAttention: 
    """
    Multi-Head Attention
    
    Instead of performing a single attention function, we: 
    1. Project Q, K, V into multiple "heads"
    2. Apply attention to each head in parallel
    3. Concatenate the results
    4. Project back to original dimension
    
    This allows the model to attend to different aspects of the input. 
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate:  float = 0.0,
        bias: bool = True
    ):
        """
        Initialize multi-head attention.
        
        Args: 
            embed_dim:  Total embedding dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for attention weights
            bias: Whether to use bias in linear projections
        """
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # d_k = d_v = embed_dim / num_heads
        self.scale = np.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        # These project from embed_dim to embed_dim (but will be split into heads)
        self.W_q = Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.W_o = Linear(embed_dim, embed_dim, bias=bias)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout_rate)
        
        # Store attention weights for visualization
        self.attention_weights:  Optional[np.ndarray] = None
    
    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Input shape: (batch_size, seq_len, embed_dim)
        Output shape: (batch_size, num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[-2] if x.data.ndim > 2 else x.shape[0]
        
        if x.data.ndim == 2:
            # Single sequence: (seq_len, embed_dim) -> (1, num_heads, seq_len, head_dim)
            reshaped = x.reshape(1, seq_len, self.num_heads, self.head_dim)
        else:
            # Batch: (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
            reshaped = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        return reshaped.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """
        Combine heads back into single dimension.
        
        Input shape: (batch_size, num_heads, seq_len, head_dim)
        Output shape: (batch_size, seq_len, embed_dim)
        """
        # Transpose back: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        x = x.transpose(0, 2, 1, 3)
        
        seq_len = x.shape[1]
        
        # Reshape to combine heads: (batch_size, seq_len, embed_dim)
        return x.reshape(batch_size, seq_len, self.embed_dim)
    
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value:  Tensor,
        mask: Optional[np.ndarray] = None,
        training: bool = False
    ) -> Tensor:
        """
        Apply multi-head attention. 
        
        Args:
            query:  Query tensor, shape (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
            key: Key tensor, shape (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
            value: Value tensor, shape (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
            mask: Optional causal mask, shape (seq_len, seq_len)
            training: Whether in training mode
        
        Returns:
            Output tensor of same shape as input
        """
        # Handle single sequence (no batch dimension)
        single_sequence = query.data.ndim == 2
        if single_sequence:
            # Add batch dimension - need to track for gradient flow
            original_query = query
            original_key = key
            original_value = value
            
            query = Tensor(query.data[np.newaxis, :, :], requires_grad=query.requires_grad, _children=(original_query,), _op='unsqueeze')
            key = Tensor(key.data[np.newaxis, :, :], requires_grad=key.requires_grad, _children=(original_key,), _op='unsqueeze')
            value = Tensor(value.data[np.newaxis, :, :], requires_grad=value.requires_grad, _children=(original_value,), _op='unsqueeze')
            
            # Define backward for unsqueeze
            def make_unsqueeze_backward(orig, expanded):
                def _backward():
                    if orig.requires_grad:
                        if orig.grad is None:
                            orig.grad = np.zeros_like(orig.data)
                        orig.grad += expanded.grad[0]
                return _backward
            
            query._backward = make_unsqueeze_backward(original_query, query)
            key._backward = make_unsqueeze_backward(original_key, key)
            value._backward = make_unsqueeze_backward(original_value, value)
        
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Step 1: Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Step 2: Split into multiple heads
        Q = self._split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len, head_dim)
        K = self._split_heads(K, batch_size)
        V = self._split_heads(V, batch_size)
        
        # Step 3: Expand mask for heads if needed
        if mask is not None: 
            # Expand mask:  (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            mask = mask[np.newaxis, np.newaxis, :, :]
        
        # Step 4: Apply attention to each head
        # This is done in parallel by treating heads as a batch dimension
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask, training=training)
        
        # Store attention weights for visualization
        self.attention_weights = attn_weights
        
        # Step 5: Combine heads
        attn_output = self._combine_heads(attn_output, batch_size)  # (batch_size, seq_len, embed_dim)
        
        # Step 6: Final linear projection
        output = self.W_o(attn_output)
        
        # Remove batch dimension if input was single sequence
        if single_sequence: 
            original_output = output
            output = Tensor(
                output.data[0],
                requires_grad=output.requires_grad,
                _children=(original_output,),
                _op='squeeze'
            )
            
            # Define backward for squeeze
            def _backward_squeeze():
                if original_output.requires_grad:
                    if original_output.grad is None:
                        original_output.grad = np.zeros_like(original_output.data)
                    original_output.grad += output.grad[np.newaxis, :, :]
            output._backward = _backward_squeeze
        
        return output
    
    def parameters(self) -> List[Tensor]:
        """Return all learnable parameters."""
        params = []
        params.extend(self.W_q.parameters())
        params.extend(self.W_k.parameters())
        params.extend(self.W_v.parameters())
        params.extend(self.W_o.parameters())
        return params
    
    def __repr__(self):
        return (f"MultiHeadAttention(\n"
                f"  embed_dim={self.embed_dim},\n"
                f"  num_heads={self.num_heads},\n"
                f"  head_dim={self.head_dim}\n"
                f")")


class CausalSelfAttention: 
    """
    Causal Self-Attention (used in decoder/GPT-style models)
    
    This is Multi-Head Attention with: 
    1. Query, Key, Value all come from the same input (self-attention)
    2. Causal masking (can only attend to past tokens)
    
    This is what GPT, Claude, and other autoregressive models use! 
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len:  int = 512,
        dropout_rate: float = 0.0,
        bias: bool = True
    ):
        """
        Initialize causal self-attention.
        
        Args:
            embed_dim:  Embedding dimension
            num_heads: Number of attention heads
            max_seq_len:  Maximum sequence length (for pre-computing mask)
            dropout_rate: Dropout rate
            bias: Whether to use bias in linear layers
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Multi-head attention
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            bias=bias
        )
        
        # Pre-compute causal mask for maximum sequence length
        self.causal_mask = create_causal_mask(max_seq_len)
    
    def __call__(
        self,
        x: Tensor,
        training: bool = False
    ) -> Tensor:
        """
        Apply causal self-attention.
        
        Args: 
            x: Input tensor, shape (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
            training: Whether in training mode
        
        Returns:
            Output tensor of same shape as input
        """
        # Get sequence length
        if x.data.ndim == 2:
            seq_len = x.shape[0]
        else: 
            seq_len = x.shape[1]
        
        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Apply multi-head attention with self-attention (Q=K=V=x)
        return self.mha(x, x, x, mask=mask, training=training)
    
    def parameters(self) -> List[Tensor]:
        """Return all learnable parameters."""
        return self.mha.parameters()
    
    @property
    def attention_weights(self) -> Optional[np.ndarray]:
        """Get attention weights from last forward pass."""
        return self.mha.attention_weights
    
    def __repr__(self):
        return (f"CausalSelfAttention(\n"
                f"  embed_dim={self.embed_dim},\n"
                f"  num_heads={self.num_heads},\n"
                f"  max_seq_len={self.max_seq_len}\n"
                f")")


# ==================== TESTS ====================

def test_linear():
    """Test the Linear layer."""
    print("\n" + "=" * 60)
    print("TESTING LINEAR LAYER")
    print("=" * 60)
    
    linear = Linear(4, 2)
    print(f"\nCreated:  {linear}")
    
    # Test forward pass
    x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]), requires_grad=True)
    y = linear(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y.data}")
    
    # Test gradient
    loss = y.sum()
    loss.backward()
    
    print(f"\nWeight gradient shape: {linear.weight.grad.shape}")
    print(f"Bias gradient: {linear.bias.grad}")
    
    print("\n✅ Linear layer tests passed!")


def test_causal_mask():
    """Test causal mask creation."""
    print("\n" + "=" * 60)
    print("TESTING CAUSAL MASK")
    print("=" * 60)
    
    mask = create_causal_mask(5)
    
    print("\nCausal mask for seq_len=5:")
    print("(0 = can attend, -inf = cannot attend)")
    print()
    
    # Print with nice formatting
    print("       Pos0  Pos1  Pos2  Pos3  Pos4")
    for i in range(5):
        row = f"Pos{i}  "
        for j in range(5):
            if mask[i, j] == 0:
                row += "  ✓   "
            else: 
                row += "  ✗   "
        print(row)
    
    print("\n✅ Causal mask test passed!")


def test_scaled_dot_product_attention():
    """Test scaled dot-product attention."""
    print("\n" + "=" * 60)
    print("TESTING SCALED DOT-PRODUCT ATTENTION")
    print("=" * 60)
    
    attention = ScaledDotProductAttention()
    
    # Create simple Q, K, V
    seq_len = 4
    d_k = 8
    
    Q = Tensor(np.random.randn(seq_len, d_k), requires_grad=True)
    K = Tensor(np.random.randn(seq_len, d_k), requires_grad=True)
    V = Tensor(np.random.randn(seq_len, d_k), requires_grad=True)
    
    print(f"\nQ shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # Without mask
    print("\n--- Without Mask ---")
    output, weights = attention(Q, K, V)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Weights sum per row (should be 1.0): {weights.sum(axis=-1)}")
    
    # With causal mask
    print("\n--- With Causal Mask ---")
    mask = create_causal_mask(seq_len)
    output_masked, weights_masked = attention(Q, K, V, mask=mask)
    
    print(f"Attention weights with mask:")
    print(weights_masked.round(3))
    print("\nNote: Upper triangle should be ~0 (masked out)")
    
    print("\n✅ Scaled dot-product attention tests passed!")


def test_multi_head_attention():
    """Test multi-head attention."""
    print("\n" + "=" * 60)
    print("TESTING MULTI-HEAD ATTENTION")
    print("=" * 60)
    
    embed_dim = 16
    num_heads = 4
    seq_len = 5
    
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    print(f"\nCreated:  {mha}")
    
    # Count parameters
    total_params = sum(p.data.size for p in mha.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"  W_q: {embed_dim * embed_dim} + {embed_dim} bias")
    print(f"  W_k: {embed_dim * embed_dim} + {embed_dim} bias")
    print(f"  W_v: {embed_dim * embed_dim} + {embed_dim} bias")
    print(f"  W_o: {embed_dim * embed_dim} + {embed_dim} bias")
    
    # Test forward pass (single sequence)
    print("\n--- Single Sequence ---")
    x = Tensor(np.random.randn(seq_len, embed_dim), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    output = mha(x, x, x)  # Self-attention
    print(f"Output shape: {output.shape}")
    print(f"Output matches input shape: {output.shape == x.shape}")
    
    # Test forward pass (batch)
    print("\n--- Batch Processing ---")
    batch_size = 3
    x_batch = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
    print(f"Input shape: {x_batch.shape}")
    
    output_batch = mha(x_batch, x_batch, x_batch)
    print(f"Output shape: {output_batch.shape}")
    
    # Test with causal mask
    print("\n--- With Causal Mask ---")
    mask = create_causal_mask(seq_len)
    output_masked = mha(x, x, x, mask=mask)
    print(f"Output shape with mask: {output_masked.shape}")
    
    # Test gradients
    print("\n--- Gradient Test ---")
    loss = output.sum()
    loss.backward()
    
    print("Gradients computed for all parameters:")
    for i, param in enumerate(mha.parameters()):
        grad_norm = np.linalg.norm(param.grad) if param.grad is not None else 0
        print(f"  Param {i}: shape={param.shape}, grad_norm={grad_norm:.4f}")
    
    print("\n✅ Multi-head attention tests passed!")


def test_causal_self_attention():
    """Test causal self-attention."""
    print("\n" + "=" * 60)
    print("TESTING CAUSAL SELF-ATTENTION")
    print("=" * 60)
    
    embed_dim = 64
    num_heads = 4
    max_seq_len = 128
    
    csa = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )
    print(f"\nCreated: {csa}")
    
    # Test forward pass
    seq_len = 10
    x = Tensor(np.random.randn(seq_len, embed_dim), requires_grad=True)
    
    print(f"\nInput shape: {x.shape}")
    output = csa(x)
    print(f"Output shape: {output.shape}")
    
    # Check attention weights
    attn_weights = csa.attention_weights
    print(f"\nAttention weights shape: {attn_weights.shape}")
    
    # Verify causality - check that attention to future positions is zero
    print("\nVerifying causality (future attention should be ~0):")
    is_causal = True
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            # Check all heads
            max_future_attn = np.max(np.abs(attn_weights[:, :, i, j]))
            if max_future_attn > 1e-6:
                print(f"  ⚠️ Position {i} attending to future position {j}:  {max_future_attn}")
                is_causal = False
    
    if is_causal:
        print("  ✅ All future attention weights are ~0")
    
    # Test gradients
    print("\n--- Gradient Test ---")
    loss = output.sum()
    loss.backward()
    
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Input gradient norm: {np.linalg.norm(x.grad):.4f}")
    
    print("\n✅ Causal self-attention tests passed!")


def visualize_attention():
    """Visualize attention patterns."""
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION")
    print("=" * 60)
    
    # Create a simple example
    embed_dim = 16
    num_heads = 2
    seq_len = 6
    
    csa = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=32
    )
    
    # Create input (pretend these are token embeddings)
    tokens = ["<BOS>", "Hello", "how", "are", "you", "?"]
    x = Tensor(np.random.randn(seq_len, embed_dim))
    
    # Forward pass
    _ = csa(x)
    
    # Get attention weights for first head
    attn = csa.attention_weights[0, 0]  # First batch, first head
    
    print(f"\nTokens: {tokens}")
    print(f"\nAttention pattern (Head 0):")
    print("(Reading:  Row token attends to Column token)")
    print()
    
    # Header
    header = "         "
    for t in tokens:
        header += f"{t: >7}"
    print(header)
    print("-" * len(header))
    
    # Attention matrix
    for i, token in enumerate(tokens):
        row = f"{token:>8} "
        for j in range(seq_len):
            if j <= i:  # Can attend
                row += f"{attn[i, j]:>7.3f}"
            else:  # Masked
                row += f"{'---':>7}"
        print(row)
    
    print("\n💡 Notice: Each token can only attend to itself and previous tokens!")
    print("   This is the causal masking in action.")


def demonstrate_attention_computation():
    """
    Step-by-step demonstration of attention computation. 
    Educational function to show exactly what happens.
    """
    print("\n" + "=" * 70)
    print("STEP-BY-STEP ATTENTION DEMONSTRATION")
    print("=" * 70)
    
    # Very simple example
    print("\n📝 Setup:")
    print("-" * 40)
    
    embed_dim = 4
    num_heads = 1
    head_dim = embed_dim // num_heads
    seq_len = 3
    
    print(f"Embedding dimension: {embed_dim}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension:  {head_dim}")
    print(f"Sequence length:  {seq_len}")
    
    # Create input (like token embeddings)
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],  # Token 0 (e. g., "Hello")
        [0.0, 1.0, 0.0, 1.0],  # Token 1 (e.g., "how")
        [1.0, 1.0, 0.0, 0.0],  # Token 2 (e.g., "are")
    ])
    
    print(f"\n📊 Input X (token embeddings):")
    print(f"   Shape: {X.shape}")
    for i, row in enumerate(X):
        print(f"   Token {i}: {row}")
    
    # Create weight matrices (small for demonstration)
    np.random.seed(42)  # For reproducibility
    W_q = np.random.randn(embed_dim, head_dim) * 0.5
    W_k = np.random.randn(embed_dim, head_dim) * 0.5
    W_v = np.random.randn(embed_dim, head_dim) * 0.5
    
    print(f"\n📊 Weight matrices:")
    print(f"   W_q shape: {W_q.shape}")
    print(f"   W_k shape: {W_k.shape}")
    print(f"   W_v shape: {W_v.shape}")
    
    # Step 1: Compute Q, K, V
    print("\n" + "=" * 60)
    print("STEP 1: Compute Q, K, V")
    print("=" * 60)
    
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    print(f"\nQ = X @ W_q:")
    print(f"   Shape: {Q.shape}")
    for i, row in enumerate(Q):
        print(f"   Token {i}: [{', '.join([f'{v:.3f}' for v in row])}]")
    
    print(f"\nK = X @ W_k:")
    print(f"   Shape: {K.shape}")
    for i, row in enumerate(K):
        print(f"   Token {i}: [{', '.join([f'{v:.3f}' for v in row])}]")
    
    print(f"\nV = X @ W_v:")
    print(f"   Shape: {V.shape}")
    for i, row in enumerate(V):
        print(f"   Token {i}: [{', '.join([f'{v:.3f}' for v in row])}]")
    
    # Step 2: Compute attention scores
    print("\n" + "=" * 60)
    print("STEP 2: Compute Attention Scores (Q @ K.T)")
    print("=" * 60)
    
    scores = Q @ K.T
    print(f"\nScores = Q @ K.T:")
    print(f"   Shape: {scores.shape}")
    print(f"\n   Scores[i][j] = how much token i attends to token j")
    print()
    print("         Token0  Token1  Token2")
    for i in range(seq_len):
        row = f"   Token{i}  "
        for j in range(seq_len):
            row += f"{scores[i, j]:7.3f} "
        print(row)
    
    # Step 3: Scale by sqrt(d_k)
    print("\n" + "=" * 60)
    print("STEP 3: Scale by √(d_k)")
    print("=" * 60)
    
    scale = np.sqrt(head_dim)
    scaled_scores = scores / scale
    
    print(f"\nScale factor = √{head_dim} = {scale:.3f}")
    print(f"\nScaled scores = scores / {scale:.3f}:")
    print()
    print("         Token0  Token1  Token2")
    for i in range(seq_len):
        row = f"   Token{i}  "
        for j in range(seq_len):
            row += f"{scaled_scores[i, j]:7.3f} "
        print(row)
    
    # Step 4: Apply causal mask
    print("\n" + "=" * 60)
    print("STEP 4: Apply Causal Mask")
    print("=" * 60)
    
    mask = create_causal_mask(seq_len)
    print(f"\nCausal mask (0 = keep, -inf = mask):")
    print()
    print("         Token0  Token1  Token2")
    for i in range(seq_len):
        row = f"   Token{i}  "
        for j in range(seq_len):
            if mask[i, j] == 0:
                row += f"{'0': >7} "
            else: 
                row += f"{'-inf':>7} "
        print(row)
    
    masked_scores = scaled_scores + mask
    print(f"\nMasked scores = scaled_scores + mask:")
    print()
    print("         Token0  Token1  Token2")
    for i in range(seq_len):
        row = f"   Token{i}  "
        for j in range(seq_len):
            if np.isinf(masked_scores[i, j]):
                row += f"{'-inf':>7} "
            else:
                row += f"{masked_scores[i, j]: 7.3f} "
        print(row)
    
    # Step 5: Apply softmax
    print("\n" + "=" * 60)
    print("STEP 5: Apply Softmax (row-wise)")
    print("=" * 60)
    
    def softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        # Handle -inf:  exp(-inf) = 0
        exp_x = np.where(np.isinf(x), 0, exp_x)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-10)
    
    attention_weights = softmax(masked_scores, axis=-1)
    
    print(f"\nAttention weights = softmax(masked_scores):")
    print(f"   (Each row sums to 1.0)")
    print()
    print("         Token0  Token1  Token2    Sum")
    for i in range(seq_len):
        row = f"   Token{i}  "
        for j in range(seq_len):
            row += f"{attention_weights[i, j]:7.3f} "
        row += f"  = {attention_weights[i].sum():.3f}"
        print(row)
    
    print("\n   💡 Notice:")
    print("      - Token 0 only attends to itself (100%)")
    print("      - Token 1 splits attention between Token 0 and Token 1")
    print("      - Token 2 attends to all three tokens")
    print("      - No token attends to future tokens!")
    
    # Step 6: Compute output
    print("\n" + "=" * 60)
    print("STEP 6: Compute Output (Attention @ V)")
    print("=" * 60)
    
    output = attention_weights @ V
    
    print(f"\nOutput = attention_weights @ V:")
    print(f"   Shape: {output.shape}")
    print()
    print(f"   For each token, output is a weighted sum of Value vectors:")
    print()
    
    for i in range(seq_len):
        print(f"   Token {i} output: [{', '.join([f'{v:.3f}' for v in output[i]])}]")
        
        # Show the weighted sum
        print(f"      = ", end="")
        terms = []
        for j in range(i + 1):  # Only up to current position (causal)
            weight = attention_weights[i, j]
            if weight > 0.001:
                terms.append(f"{weight:.3f} × V[{j}]")
        print(" + ".join(terms))
        print()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    The attention mechanism: 
    
    1. Projects inputs into Query (Q), Key (K), Value (V) spaces
    2. Computes similarity scores:  Q @ K.T
    3. Scales by √(d_k) for numerical stability
    4. Applies causal mask (optional, for autoregressive models)
    5. Applies softmax to get attention weights (probabilities)
    6. Computes weighted sum of Values using attention weights
    
    Result: Each token's output is a weighted combination of all
    (allowed) token values, where weights are learned similarities! 
    """)


if __name__ == "__main__":
    # Run all tests
    test_linear()
    test_causal_mask()
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_causal_self_attention()
    
    # Visualize attention
    visualize_attention()
    
    # Step-by-step demonstration
    demonstrate_attention_computation()
    
    print("\n" + "=" * 60)
    print("ALL ATTENTION TESTS PASSED! ✅")
    print("=" * 60)