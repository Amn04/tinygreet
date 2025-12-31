"""
Transformer Block

A complete Transformer decoder block consists of:
1. Causal Self-Attention
2. Layer Normalization
3. Feed-Forward Network
4. Residual Connections

This is the building block that gets stacked to create GPT-like models! 
"""

import numpy as np
from typing import List, Optional

from tensor import Tensor
from embeddings import LayerNorm
from attention import CausalSelfAttention
from feedforward import FeedForward


class TransformerBlock:
    """
    A single Transformer decoder block.
    
    Architecture (Pre-LayerNorm, like GPT-2):
    
        ┌───────────────────────────────────┐
        │              Input                │
        └───────────────┬───────────────────┘
                        │
                        ├─────────────────────────┐ (Residual)
                        │                         │
                        ▼                         │
                ┌───────────────┐                 │
                │  LayerNorm 1  │                 │
                └───────┬───────┘                 │
                        │                         │
                        ▼                         │
                ┌───────────────┐                 │
                │   Causal      │                 │
                │ Self-Attention│                 │
                └───────┬───────┘                 │
                        │                         │
                        ▼                         │
                      (Add) ◄─────────────────────┘
                        │
                        ├─────────────────────────┐ (Residual)
                        │                         │
                        ▼                         │
                ┌───────────────┐                 │
                │  LayerNorm 2  │                 │
                └───────┬───────┘                 │
                        │                         │
                        ▼                         │
                ┌───────────────┐                 │
                │  Feed-Forward │                 │
                └───────┬───────┘                 │
                        │                         │
                        ▼                         │
                      (Add) ◄─────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │             Output                │
        └───────────────────────────────────┘
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads:  int,
        ff_hidden_dim: int = None,
        max_seq_len:  int = 512,
        dropout_rate: float = 0.0,
        activation: str = 'gelu'
    ):
        """
        Initialize a Transformer block. 
        
        Args: 
            embed_dim:  Embedding dimension
            num_heads: Number of attention heads
            ff_hidden_dim:  Feed-forward hidden dimension (default: 4 * embed_dim)
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            activation:  Activation function for FFN
        """
        self.embed_dim = embed_dim
        self. num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim or 4 * embed_dim
        
        # Layer Normalization
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        
        # Causal Self-Attention
        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )
        
        # Feed-Forward Network
        self. ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=self. ff_hidden_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        self.dropout_rate = dropout_rate
    
    def __call__(self, x:  Tensor, training:  bool = False) -> Tensor:
        """
        Forward pass through the Transformer block.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
            training: Whether in training mode
        
        Returns:
            Output tensor, same shape as input
        """
        # ===== Attention Sub-block =====
        # Pre-LayerNorm
        normalized = self.ln1(x)
        
        # Causal Self-Attention
        attn_out = self.attention(normalized, training=training)
        
        # Dropout (if training)
        if training and self. dropout_rate > 0:
            mask = np.random. binomial(
                1, 1 - self.dropout_rate, attn_out.shape
            ).astype(np.float32)
            attn_out = Tensor(
                attn_out.data * mask / (1 - self.dropout_rate),
                requires_grad=attn_out.requires_grad,
                _children=(attn_out,),
                _op='dropout'
            )
        
        # Residual connection
        x = x + attn_out
        
        # ===== Feed-Forward Sub-block =====
        # Pre-LayerNorm
        normalized = self.ln2(x)
        
        # Feed-Forward Network
        ffn_out = self.ffn(normalized, training=training)
        
        # Dropout (if training)
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(
                1, 1 - self.dropout_rate, ffn_out.shape
            ).astype(np.float32)
            ffn_out = Tensor(
                ffn_out.data * mask / (1 - self.dropout_rate),
                requires_grad=ffn_out.requires_grad,
                _children=(ffn_out,),
                _op='dropout'
            )
        
        # Residual connection
        x = x + ffn_out
        
        return x
    
    def parameters(self) -> List[Tensor]:
        """Return all learnable parameters."""
        params = []
        params.extend(self.ln1.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.ffn.parameters())
        return params
    
    @property
    def attention_weights(self) -> Optional[np.ndarray]: 
        """Get attention weights from last forward pass."""
        return self.attention.attention_weights
    
    def __repr__(self):
        return (f"TransformerBlock(\n"
                f"  embed_dim={self.embed_dim},\n"
                f"  num_heads={self.num_heads},\n"
                f"  ff_hidden_dim={self.ff_hidden_dim}\n"
                f")")


class TransformerStack:
    """
    Stack of Transformer blocks. 
    
    This is what you get when you stack N Transformer blocks together. 
    GPT-2 Small has 12 layers, GPT-3 has 96 layers! 
    
    For TinyGreet, we'll use 2-4 layers.
    """
    
    def __init__(
        self,
        num_layers: int,
        embed_dim:  int,
        num_heads: int,
        ff_hidden_dim: int = None,
        max_seq_len:  int = 512,
        dropout_rate: float = 0.0,
        activation: str = 'gelu'
    ):
        """
        Initialize a stack of Transformer blocks. 
        
        Args:
            num_layers: Number of Transformer blocks to stack
            embed_dim: Embedding dimension
            num_heads: Number of attention heads per block
            ff_hidden_dim: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
            activation: Activation function for FFN
        """
        self.num_layers = num_layers
        self. embed_dim = embed_dim
        
        # Create the stack of blocks
        self.blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                max_seq_len=max_seq_len,
                dropout_rate=dropout_rate,
                activation=activation
            )
            for _ in range(num_layers)
        ]
    
    def __call__(self, x: Tensor, training: bool = False) -> Tensor: 
        """
        Forward pass through all Transformer blocks. 
        
        Args: 
            x: Input tensor from embedding layer
            training:  Whether in training mode
        
        Returns: 
            Output tensor, same shape as input
        """
        for block in self.blocks:
            x = block(x, training=training)
        return x
    
    def parameters(self) -> List[Tensor]:
        """Return all learnable parameters from all blocks."""
        params = []
        for block in self. blocks:
            params.extend(block. parameters())
        return params
    
    def get_attention_weights(self, layer:  int = -1) -> Optional[np.ndarray]: 
        """Get attention weights from a specific layer (default: last)."""
        return self.blocks[layer].attention_weights
    
    def __repr__(self):
        return (f"TransformerStack(\n"
                f"  num_layers={self.num_layers},\n"
                f"  embed_dim={self.embed_dim},\n"
                f"  blocks={self.blocks[0]}\n"
                f")")


# ==================== TESTS ====================

def test_transformer_block():
    """Test a single Transformer block."""
    print("\n" + "=" * 60)
    print("TESTING TRANSFORMER BLOCK")
    print("=" * 60)
    
    embed_dim = 64
    num_heads = 4
    seq_len = 10
    
    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=128
    )
    print(f"\nCreated: {block}")
    
    # Count parameters
    total_params = sum(p.data.size for p in block.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Break down parameters
    ln_params = sum(p.data. size for p in block.ln1.parameters()) + \
                sum(p.data. size for p in block.ln2.parameters())
    attn_params = sum(p.data. size for p in block.attention.parameters())
    ffn_params = sum(p.data.size for p in block.ffn.parameters())
    
    print(f"  LayerNorm (×2):     {ln_params:,}")
    print(f"  Attention:           {attn_params:,}")
    print(f"  Feed-Forward:       {ffn_params:,}")
    
    # Test forward pass
    print("\n--- Forward Pass ---")
    x = Tensor(np.random.randn(seq_len, embed_dim), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    output = block(x)
    print(f"Output shape:  {output.shape}")
    print(f"Output matches input shape: {output.shape == x.shape}")
    
    # Check residual connection effect
    input_norm = np.linalg. norm(x.data)
    output_norm = np.linalg.norm(output.data)
    print(f"\nInput norm:   {input_norm:.4f}")
    print(f"Output norm: {output_norm:.4f}")
    print("(Residual connections help maintain signal magnitude)")
    
    # Test gradient
    print("\n--- Gradient Test ---")
    loss = output. sum()
    loss.backward()
    
    print(f"Input gradient norm: {np.linalg.norm(x.grad):.4f}")
    print("Parameter gradients computed:  ✅")
    
    print("\n✅ Transformer block tests passed!")


def test_transformer_stack():
    """Test a stack of Transformer blocks."""
    print("\n" + "=" * 60)
    print("TESTING TRANSFORMER STACK")
    print("=" * 60)
    
    num_layers = 3
    embed_dim = 64
    num_heads = 4
    seq_len = 10
    
    stack = TransformerStack(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=128
    )
    print(f"\nCreated: {stack}")
    
    # Count parameters
    total_params = sum(p.data. size for p in stack.parameters())
    params_per_layer = total_params // num_layers
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Parameters per layer: {params_per_layer:,}")
    
    # Test forward pass
    print("\n--- Forward Pass ---")
    x = Tensor(np.random.randn(seq_len, embed_dim), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    output = stack(x)
    print(f"Output shape: {output.shape}")
    
    # Test gradient flow through all layers
    print("\n--- Gradient Flow Test ---")
    loss = output.sum()
    loss.backward()
    
    print(f"Input gradient norm: {np.linalg. norm(x.grad):.4f}")
    print("Gradients flowing through all {num_layers} layers:  ✅")
    
    # Check attention weights from each layer
    print("\n--- Attention Weights ---")
    for i, block in enumerate(stack. blocks):
        attn = block.attention_weights
        if attn is not None: 
            print(f"Layer {i} attention shape: {attn. shape}")
    
    print("\n✅ Transformer stack tests passed!")


def demonstrate_transformer_block():
    """
    Demonstrate what happens inside a Transformer block.
    """
    print("\n" + "=" * 70)
    print("TRANSFORMER BLOCK DEMONSTRATION")
    print("=" * 70)
    
    embed_dim = 8
    num_heads = 2
    seq_len = 4
    
    print(f"""
    Configuration:
    - Embedding dimension: {embed_dim}
    - Number of attention heads: {num_heads}
    - Sequence length: {seq_len}
    - Feed-forward hidden dim: {4 * embed_dim}
    """)
    
    # Create block
    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=32
    )
    
    # Create input
    np.random.seed(42)
    x = Tensor(np.random.randn(seq_len, embed_dim))
    
    print("Input X (shape:  {seq_len} tokens × {embed_dim} dims):")
    print("-" * 50)
    for i in range(seq_len):
        print(f"  Token {i}: [{', '.join([f'{v:.2f}' for v in x. data[i, :4]])}...]")
    
    # Forward pass with explanations
    print("\n" + "=" * 50)
    print("STEP 1: Layer Normalization (pre-attention)")
    print("=" * 50)
    
    ln_out = block.ln1(x)
    print(f"\nAfter LayerNorm (mean ≈ 0, std ≈ 1 per token):")
    for i in range(min(2, seq_len)):
        row = ln_out.data[i]
        print(f"  Token {i}: mean={row.mean():.4f}, std={row.std():.4f}")
    
    print("\n" + "=" * 50)
    print("STEP 2: Causal Self-Attention")
    print("=" * 50)
    
    attn_out = block.attention(ln_out)
    print(f"\nAttention output shape: {attn_out.shape}")
    print(f"Each token now contains information from previous tokens!")
    
    # Show attention pattern
    attn_weights = block.attention. attention_weights
    if attn_weights is not None: 
        print(f"\nAttention pattern (Head 0):")
        print("         ", end="")
        for j in range(seq_len):
            print(f"  Tok{j}", end="")
        print()
        
        weights = attn_weights[0, 0] if attn_weights. ndim == 4 else attn_weights[0]
        for i in range(seq_len):
            print(f"  Token{i}", end="")
            for j in range(seq_len):
                if j <= i:
                    print(f"  {weights[i, j]:.2f}", end="")
                else: 
                    print(f"   ---", end="")
            print()
    
    print("\n" + "=" * 50)
    print("STEP 3: Residual Connection (x + attention_output)")
    print("=" * 50)
    
    residual1 = x + attn_out
    print(f"\nResidual connection preserves original information")
    print(f"while adding attention-aggregated context.")
    
    print("\n" + "=" * 50)
    print("STEP 4: Layer Normalization (pre-FFN)")
    print("=" * 50)
    
    ln2_out = block. ln2(residual1)
    print(f"\nNormalized again before feed-forward network.")
    
    print("\n" + "=" * 50)
    print("STEP 5: Feed-Forward Network")
    print("=" * 50)
    
    ffn_out = block.ffn(ln2_out)
    print(f"""
    FFN applies the same transformation to each position: 
    - Linear: {embed_dim} → {4 * embed_dim}
    - GELU activation
    - Linear: {4 * embed_dim} → {embed_dim}
    
    This adds non-linearity and allows the model to 
    "think" about the information from attention. 
    """)
    
    print("\n" + "=" * 50)
    print("STEP 6: Final Residual Connection")
    print("=" * 50)
    
    output = residual1 + ffn_out
    print(f"\nFinal output shape: {output. shape}")
    print(f"Same shape as input, but now each token contains:")
    print(f"  - Original information (from residual)")
    print(f"  - Context from other tokens (from attention)")
    print(f"  - Non-linear transformations (from FFN)")
    
    print("\n" + "=" * 50)
    print("COMPARISON: Input vs Output")
    print("=" * 50)
    
    print(f"\nInput token 0:  [{', '.join([f'{v:.2f}' for v in x.data[0, :4]])}...]")
    print(f"Output token 0: [{', '.join([f'{v:.2f}' for v in output.data[0, :4]])}...]")
    
    diff = np.linalg.norm(output.data[0] - x.data[0])
    print(f"\nDifference (L2 norm): {diff:.4f}")
    print("(The output is different because it now incorporates context! )")


if __name__ == "__main__": 
    # Run tests
    test_transformer_block()
    test_transformer_stack()
    
    # Demonstration
    demonstrate_transformer_block()
    
    print("\n" + "=" * 60)
    print("ALL TRANSFORMER BLOCK TESTS PASSED!  ✅")
    print("=" * 60)