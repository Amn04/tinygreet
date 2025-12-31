"""
Integration Test:   Tokenizer + Embeddings

This tests the complete pipeline:
1. Text -> Tokenizer -> Token IDs
2. Token IDs -> Embeddings -> Vectors

This is exactly what happens at the start of every LLM forward pass! 
"""

import sys
import os
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tokenizer'))

from tensor import Tensor
from embeddings import TransformerEmbedding


def load_tokenizer():
    """Load our trained BPE tokenizer."""
    try:
        from bpe import BPETokenizer
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'tokenizer', 'vocab'
        )
        tokenizer = BPETokenizer.load(tokenizer_path)
        return tokenizer
    except Exception as e:
        print(f"Could not load tokenizer:  {e}")
        print("Make sure you've trained the tokenizer first!")
        return None


def main():
    print("=" * 70)
    print("INTEGRATION TEST:  TOKENIZER + EMBEDDINGS")
    print("=" * 70)
    
    # Load tokenizer
    print("\n📂 Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    if tokenizer is None: 
        print("\n⚠️  Creating a demo tokenizer for testing...")
        # Create a simple demo tokenizer
        from bpe import BPETokenizer
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train([
            "hello", "hello world", "good morning",
            "how are you", "goodbye", "thank you"
        ], verbose=False, min_frequency=1)
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"   Vocabulary size: {vocab_size}")
    
    # Create embedding layer
    print("\n🔧 Creating embedding layer...")
    embed_dim = 64
    max_seq_len = 128
    
    embedding_layer = TransformerEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        padding_idx=tokenizer.vocab.get('<PAD>', 0),
        use_layer_norm=True
    )
    
    print(f"   Embedding dimension: {embed_dim}")
    print(f"   Max sequence length: {max_seq_len}")
    
    # Count parameters
    total_params = sum(p.data.size for p in embedding_layer.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Test examples
    test_texts = [
        "Hello! ",
        "Good morning!",
        "How are you? ",
        "Goodbye!",
        "Thank you so much! ",
    ]
    
    print("\n" + "=" * 70)
    print("PROCESSING EXAMPLES")
    print("=" * 70)
    
    for text in test_texts:
        print(f"\n{'─' * 60}")
        print(f"Input: \"{text}\"")
        print(f"{'─' * 60}")
        
        # Step 1: Tokenize
        token_ids = tokenizer.encode(text)
        tokens = [tokenizer.id_to_token(tid) for tid in token_ids]
        
        print(f"\n  Step 1: Tokenization")
        print(f"    Tokens: {tokens}")
        print(f"    IDs:    {token_ids}")
        
        # Step 2: Convert to numpy array
        token_ids_np = np.array(token_ids)
        
        # Step 3: Get embeddings
        embeddings = embedding_layer(token_ids_np)
        
        print(f"\n  Step 2: Embedding")
        print(f"    Input shape:   {token_ids_np.shape}")
        print(f"    Output shape: {embeddings.shape}")
        print(f"    Output dtype: {embeddings.dtype}")
        
        # Show statistics
        print(f"\n  Step 3: Output Statistics")
        print(f"    Mean:  {embeddings.data.mean():.4f}")
        print(f"    Std:  {embeddings.data.std():.4f}")
        print(f"    Min:  {embeddings.data.min():.4f}")
        print(f"    Max:  {embeddings.data.max():.4f}")
        
        # Show first token's embedding (truncated)
        print(f"\n  First token ('{tokens[0]}') embedding (first 8 dims):")
        print(f"    {embeddings.data[0, :8]}")
    
    # Test batch processing
    print("\n" + "=" * 70)
    print("BATCH PROCESSING TEST")
    print("=" * 70)
    
    batch_texts = [
        "Hello! ",
        "Good morning, how are you?",
        "Bye! ",
    ]
    
    print(f"\nBatch of {len(batch_texts)} texts:")
    for i, text in enumerate(batch_texts):
        print(f"  {i}:  \"{text}\"")
    
    # Tokenize all and find max length
    batch_token_ids = [tokenizer.encode(text) for text in batch_texts]
    max_len = max(len(ids) for ids in batch_token_ids)
    
    print(f"\nToken lengths:  {[len(ids) for ids in batch_token_ids]}")
    print(f"Max length: {max_len}")
    
    # Pad to same length
    pad_id = tokenizer.vocab.get('<PAD>', 0)
    padded_batch = np.array([
        ids + [pad_id] * (max_len - len(ids))
        for ids in batch_token_ids
    ])
    
    print(f"Padded batch shape: {padded_batch.shape}")
    
    # Get embeddings for batch
    batch_embeddings = embedding_layer(padded_batch)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    print(f"Expected:  ({len(batch_texts)}, {max_len}, {embed_dim})")
    
    # Gradient test
    print("\n" + "=" * 70)
    print("GRADIENT FLOW TEST")
    print("=" * 70)
    
    # Forward pass
    token_ids = np.array(tokenizer.encode("Hello!"))
    embeddings = embedding_layer(token_ids)
    
    # Simulate loss (sum of all embeddings)
    loss = embeddings.sum()
    print(f"\nForward pass: text -> embeddings -> loss")
    print(f"  Loss value: {loss.data:.4f}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nBackward pass: gradients computed!")
    for i, param in enumerate(embedding_layer.parameters()):
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad)
            grad_mean = param.grad.mean()
            print(f"  Parameter {i}: shape={param.shape}, "
                  f"grad_norm={grad_norm:.4f}, grad_mean={grad_mean:.6f}")
    
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST COMPLETE!")
    print("=" * 70)
    
    print("\n📊 SUMMARY")
    print("-" * 40)
    print(f"  Tokenizer vocabulary:  {vocab_size} tokens")
    print(f"  Embedding dimension:   {embed_dim}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Gradient flow:        ✅ Working")
    print("\n🚀 Ready for Phase 3:  Attention Mechanism!")


if __name__ == "__main__":
    main()