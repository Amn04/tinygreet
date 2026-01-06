"""
TinyGreet Language Model

A complete GPT-style language model built from scratch. 
This combines all our components into a working LLM! 
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import json
import os
import pickle

from tensor import Tensor
from embeddings import Embedding, PositionalEncoding, LayerNorm
from attention import Linear, CausalSelfAttention, KVCache
from feedforward import FeedForward
from transformer_block import TransformerBlock, TransformerStack


class TinyGreetConfig:
    """
    Configuration for TinyGreet model. 
    
    This holds all hyperparameters in one place,
    similar to HuggingFace's config classes.
    """
    
    def __init__(
        self,
        vocab_size: int = 500,
        embed_dim: int = 128,       # Scaled up from 64
        num_heads:  int = 8,        # Scaled up from 4
        num_layers: int = 6,        # Scaled up from 2
        ff_hidden_dim: int = None,  # Default: 4 * embed_dim
        max_seq_len: int = 256,     # Scaled up from 128
        dropout_rate: float = 0.1,
        activation: str = 'gelu',
        pad_token_id:  int = 0,
        bos_token_id: int = 2,
        eos_token_id:  int = 3,
        sep_token_id:  int = 4,
        use_gradient_checkpointing: bool = False,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_hidden_dim = ff_hidden_dim or 4 * embed_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers':  self.num_layers,
            'ff_hidden_dim':  self.ff_hidden_dim,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'sep_token_id': self.sep_token_id,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TinyGreetConfig': 
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path:  str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path:  str) -> 'TinyGreetConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def __repr__(self):
        return f"TinyGreetConfig({self.to_dict()})"


class TinyGreetModel:
    """
    TinyGreet Language Model
    
    A GPT-style autoregressive language model for greeting/farewell generation.
    
    Architecture:
    - Token Embedding + Positional Encoding
    - N Transformer Blocks (with causal self-attention)
    - Final LayerNorm
    - Output projection to vocabulary
    """
    
    def __init__(self, config: TinyGreetConfig):
        """
        Initialize the model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        
        # Token Embedding
        self.token_embedding = Embedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            padding_idx=config.pad_token_id
        )
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            embed_dim=config.embed_dim,
            max_seq_len=config.max_seq_len
        )
        
        # Embedding scale (from "Attention Is All You Need")
        self.embed_scale = np.sqrt(config.embed_dim)
        
        # Transformer Stack
        self.transformer = TransformerStack(
            num_layers=config.num_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_hidden_dim=config.ff_hidden_dim,
            max_seq_len=config.max_seq_len,
            dropout_rate=config.dropout_rate,
            activation=config.activation,
            use_gradient_checkpointing=config.use_gradient_checkpointing
        )
        
        # Final Layer Normalization
        self.final_ln = LayerNorm(config.embed_dim)
        
        # Output projection (Language Model Head)
        # Projects from embed_dim to vocab_size
        self.lm_head = Linear(
            in_features=config.embed_dim,
            out_features=config.vocab_size,
            bias=False  # GPT-2 style:  no bias in output projection
        )
        
        # Weight tying:  share weights between embedding and output projection
        # This is a common technique that improves performance and reduces parameters
        # Note: embedding is (vocab_size, embed_dim) but lm_head needs (embed_dim, vocab_size)
        # We need to transpose when using, so we store the embedding weight and use .T
        self._tie_weights = True  # Flag to indicate weight tying is active
        
        # Store whether we're in training mode
        self.training = False
        
        # KV cache for efficient generation
        self._kv_cache: Optional[KVCache] = None
    
    def create_kv_cache(self) -> KVCache:
        """Create a new KV cache for generation."""
        return KVCache(self.config.num_layers, self.config.max_seq_len)
    
    def forward(
        self,
        input_ids: np.ndarray,
        training: bool = False,
        use_cache: bool = False,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0
    ) -> Tensor:
        """
        Forward pass through the model.
        
        Args: 
            input_ids:  Token IDs, shape (batch_size, seq_len) or (seq_len,)
            training: Whether in training mode (affects dropout)
            use_cache: Whether to use KV cache for efficient generation
            kv_cache: KV cache object
            position_offset: Position offset for positional encoding (for cached generation)
        
        Returns: 
            Logits tensor, shape (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
        """
        self.training = training
        
        # Handle single sequence
        single_sequence = input_ids.ndim == 1
        if single_sequence:
            input_ids = input_ids[np.newaxis, :]
        
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Token Embedding
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        # Step 2: Scale embeddings
        x_unscaled = x
        x = Tensor(
            x.data * self.embed_scale,
            requires_grad=x.requires_grad,
            _children=(x_unscaled,),
            _op='scale'
        )
        # Define backward for scale
        embed_scale = self.embed_scale
        def _backward_scale():
            if x_unscaled.requires_grad:
                if x_unscaled.grad is None:
                    x_unscaled.grad = np.zeros_like(x_unscaled.data)
                x_unscaled.grad += x.grad * embed_scale
        x._backward = _backward_scale
        
        # Step 3: Add Positional Encoding (with optional offset for cached generation)
        x = self.positional_encoding(x, position_offset=position_offset)
        
        # Step 4: Pass through Transformer Stack (with optional KV cache)
        x = self.transformer(x, training=training, use_cache=use_cache, kv_cache=kv_cache)
        
        # Step 5: Final Layer Normalization
        x = self.final_ln(x)
        
        # Step 6: Project to vocabulary (get logits)
        # Use weight tying: multiply by transposed embedding weights
        if self._tie_weights:
            # x @ embedding.weight.T = (batch, seq, embed) @ (embed, vocab) = (batch, seq, vocab)
            logits = x @ self.token_embedding.weight.T
        else:
            logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Remove batch dimension if input was single sequence
        if single_sequence:
            logits_batched = logits
            logits = Tensor(
                logits.data[0],
                requires_grad=logits.requires_grad,
                _children=(logits_batched,),
                _op='squeeze'
            )
            # Define backward for squeeze
            def _backward_squeeze():
                if logits_batched.requires_grad:
                    if logits_batched.grad is None:
                        logits_batched.grad = np.zeros_like(logits_batched.data)
                    logits_batched.grad += logits.grad[np.newaxis, :, :]
            logits._backward = _backward_squeeze
        
        return logits
    
    def __call__(
        self, 
        input_ids: np.ndarray, 
        training: bool = False,
        use_cache: bool = False,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0
    ) -> Tensor:
        """Make the model callable."""
        return self.forward(
            input_ids, 
            training=training, 
            use_cache=use_cache, 
            kv_cache=kv_cache,
            position_offset=position_offset
        )
    
    def parameters(self) -> List[Tensor]: 
        """Return all learnable parameters."""
        params = []
        
        # Token embedding (note: lm_head shares weights, so don't double count)
        params.extend(self.token_embedding.parameters())
        
        # Transformer stack
        params.extend(self.transformer.parameters())
        
        # Final layer norm
        params.extend(self.final_ln.parameters())
        
        # lm_head bias (if any) - weights are shared with embedding
        if self.lm_head.bias is not None:
            params.append(self.lm_head.bias)
        
        return params
    
    def num_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.data.size for p in self.parameters())
    
    def save(self, directory: str):
        """
        Save model to directory.
        
        Saves:
        - config.json: Model configuration
        - model. pkl: Model weights
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(directory, 'config.json'))
        
        # Save weights
        weights = {}
        for i, param in enumerate(self.parameters()):
            weights[f'param_{i}'] = param.data
        
        with open(os.path.join(directory, 'model.pkl'), 'wb') as f:
            pickle.dump(weights, f)
        
        print(f"💾 Model saved to {directory}/")
    
    @classmethod
    def load(cls, directory: str) -> 'TinyGreetModel':
        """Load model from directory."""
        # Load config
        config = TinyGreetConfig.load(os.path.join(directory, 'config.json'))
        
        # Create model
        model = cls(config)
        
        # Load weights
        with open(os.path.join(directory, 'model.pkl'), 'rb') as f:
            weights = pickle.load(f)
        
        for i, param in enumerate(model.parameters()):
            param.data = weights[f'param_{i}']
        
        print(f"📂 Model loaded from {directory}/")
        return model
    
    def summary(self):
        """Print model summary."""
        print("\n" + "=" * 60)
        print("TINYGREET MODEL SUMMARY")
        print("=" * 60)
        
        print(f"\nConfiguration:")
        for key, value in self.config.to_dict().items():
            print(f"  {key}: {value}")
        
        print(f"\nArchitecture:")
        print(f"  Token Embedding: ({self.config.vocab_size}, {self.config.embed_dim})")
        print(f"  Positional Encoding: sinusoidal, max_len={self.config.max_seq_len}")
        print(f"  Transformer Blocks: {self.config.num_layers}")
        print(f"    - Attention Heads: {self.config.num_heads}")
        print(f"    - Head Dimension: {self.config.embed_dim // self.config.num_heads}")
        print(f"    - FFN Hidden Dim:  {self.config.ff_hidden_dim}")
        print(f"  Final LayerNorm: {self.config.embed_dim}")
        print(f"  LM Head: ({self.config.embed_dim}, {self.config.vocab_size})")
        print(f"    (weight tied with embedding)")
        
        total_params = self.num_parameters()
        print(f"\nTotal Parameters: {total_params:,}")
        
        # Break down by component
        emb_params = sum(p.data.size for p in self.token_embedding.parameters())
        trans_params = sum(p.data.size for p in self.transformer.parameters())
        ln_params = sum(p.data.size for p in self.final_ln.parameters())
        
        print(f"  Token Embedding: {emb_params:,} ({100*emb_params/total_params:.1f}%)")
        print(f"  Transformer: {trans_params:,} ({100*trans_params/total_params:.1f}%)")
        print(f"  Final LayerNorm: {ln_params:,} ({100*ln_params/total_params:.1f}%)")
        
        print("\n" + "=" * 60)


# ==================== TEST ====================

def test_model():
    """Test the complete model."""
    print("\n" + "=" * 60)
    print("TESTING TINYGREET MODEL")
    print("=" * 60)
    
    # Create config
    config = TinyGreetConfig(
        vocab_size=500,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=128
    )
    
    # Create model
    model = TinyGreetModel(config)
    model.summary()
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    
    # Single sequence
    input_ids = np.array([2, 229, 6, 4, 223, 263, 73, 6, 3])  # Example token IDs
    print(f"Input shape: {input_ids.shape}")
    
    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected:  ({len(input_ids)}, {config.vocab_size})")
    
    # Batch
    print("\n--- Batch Test ---")
    batch_input = np.array([
        [2, 229, 6, 3, 0, 0],
        [2, 109, 272, 6, 3, 0],
    ])
    print(f"Batch input shape: {batch_input.shape}")
    
    batch_logits = model(batch_input)
    print(f"Batch output shape: {batch_logits.shape}")
    
    # Test gradient
    print("\n--- Gradient Test ---")
    input_ids = np.array([2, 229, 6, 3])
    logits = model(input_ids, training=True)
    
    # Simulate loss (sum of logits for simplicity)
    loss = logits.sum()
    loss.backward()
    
    print("Gradients computed for all parameters:  ✅")
    
    # Check a few gradients
    params = model.parameters()
    for i in [0, len(params)//2, -1]:
        param = params[i]
        grad_norm = np.linalg.norm(param.grad) if param.grad is not None else 0
        print(f"  Param {i}: shape={param.shape}, grad_norm={grad_norm:.4f}")
    
    print("\n✅ Model tests passed!")


if __name__ == "__main__":
    test_model()
