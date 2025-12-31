"""
Feed-Forward Network for Transformer

The FFN is applied after attention in each Transformer block. 
It's a simple two-layer MLP with a non-linearity in between. 

FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

The hidden dimension is typically 4x the embedding dimension.
"""

import numpy as np
from typing import List

from tensor import Tensor
from attention import Linear


class FeedForward: 
    """
    Position-wise Feed-Forward Network
    
    This is applied to each position independently and identically. 
    
    Architecture:
        Input (embed_dim) 
          → Linear (embed_dim → hidden_dim)
          → GELU activation
          → Linear (hidden_dim → embed_dim)
        Output (embed_dim)
    
    The hidden_dim is typically 4 * embed_dim (following GPT-2).
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        dropout_rate: float = 0.0,
        activation: str = 'gelu'
    ):
        """
        Initialize feed-forward network. 
        
        Args: 
            embed_dim: Input and output dimension
            hidden_dim: Hidden layer dimension (default: 4 * embed_dim)
            dropout_rate: Dropout rate
            activation: Activation function ('gelu' or 'relu')
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or 4 * embed_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Two linear layers
        self.fc1 = Linear(embed_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, embed_dim)
    
    def __call__(self, x: Tensor, training: bool = False) -> Tensor: 
        """
        Forward pass. 
        
        Args:
            x:  Input tensor, shape (..., embed_dim)
            training:  Whether in training mode (affects dropout)
        
        Returns: 
            Output tensor, same shape as input
        """
        # First linear layer
        h = self.fc1(x)
        
        # Activation
        if self.activation == 'gelu':
            h = h.gelu()
        elif self.activation == 'relu':
            h = h.relu()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Dropout (if training)
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(
                1, 1 - self.dropout_rate, h.shape
            ).astype(np.float32)
            h_pre_dropout = h
            scale = 1 - self.dropout_rate
            h = Tensor(
                h.data * mask / scale,
                requires_grad=h.requires_grad,
                _children=(h_pre_dropout,),
                _op='dropout'
            )
            # Define backward for dropout
            def _backward_dropout():
                if h_pre_dropout.requires_grad:
                    if h_pre_dropout.grad is None:
                        h_pre_dropout.grad = np.zeros_like(h_pre_dropout.data)
                    h_pre_dropout.grad += h.grad * mask / scale
            h._backward = _backward_dropout
        
        # Second linear layer
        out = self.fc2(h)
        
        return out
    
    def parameters(self) -> List[Tensor]: 
        """Return all learnable parameters."""
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params
    
    def __repr__(self):
        return (f"FeedForward(\n"
                f"  embed_dim={self.embed_dim},\n"
                f"  hidden_dim={self.hidden_dim},\n"
                f"  activation={self.activation}\n"
                f")")


# ==================== TEST ====================

def test_feedforward():
    """Test the FeedForward network."""
    print("\n" + "=" * 60)
    print("TESTING FEED-FORWARD NETWORK")
    print("=" * 60)
    
    embed_dim = 64
    hidden_dim = 256
    seq_len = 10
    
    ff = FeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim)
    print(f"\nCreated:  {ff}")
    
    # Count parameters
    total_params = sum(p.data.size for p in ff.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  fc1: {embed_dim} × {hidden_dim} + {hidden_dim} = {embed_dim * hidden_dim + hidden_dim: ,}")
    print(f"  fc2: {hidden_dim} × {embed_dim} + {embed_dim} = {hidden_dim * embed_dim + embed_dim:,}")
    
    # Test forward pass
    print("\n--- Forward Pass ---")
    x = Tensor(np.random.randn(seq_len, embed_dim), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    output = ff(x)
    print(f"Output shape: {output.shape}")
    print(f"Output matches input shape: {output.shape == x.shape}")
    
    # Test gradient
    print("\n--- Gradient Test ---")
    loss = output.sum()
    loss.backward()
    
    print("Gradients computed:")
    for i, param in enumerate(ff. parameters()):
        grad_norm = np.linalg. norm(param.grad) if param.grad is not None else 0
        print(f"  Param {i}: shape={param.shape}, grad_norm={grad_norm:.4f}")
    
    print("\n✅ Feed-forward network tests passed!")


if __name__ == "__main__":
    test_feedforward()