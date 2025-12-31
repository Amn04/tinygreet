"""
Loss Functions for Language Model Training

Cross-Entropy Loss is the standard loss for language modeling.
We predict the next token and compare with the actual next token.
"""

import numpy as np
from typing import Optional

from tensor import Tensor


class CrossEntropyLoss: 
    """
    Cross-Entropy Loss for Language Modeling
    
    For a sequence of predictions, we compute: 
    
    loss = -1/N * Σ log(p(target_token))
    
    Where p(target_token) is the predicted probability of the correct token.
    
    This is equivalent to:
    loss = -1/N * Σ (log_softmax(logits)[target])
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        """
        Initialize loss function.
        
        Args:
            ignore_index: Token ID to ignore in loss computation (e.g., padding)
            label_smoothing: Amount of label smoothing (0.0 = no smoothing)
        """
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def __call__(
        self,
        logits: Tensor,
        targets:  np.ndarray,
        mask: Optional[np. ndarray] = None
    ) -> Tensor:
        """
        Compute cross-entropy loss. 
        
        Args: 
            logits:  Model output logits, shape (batch_size, seq_len, vocab_size)
                   or (seq_len, vocab_size)
            targets: Target token IDs, shape (batch_size, seq_len) or (seq_len,)
            mask: Optional mask for valid positions (1 = valid, 0 = ignore)
        
        Returns:
            Scalar loss tensor
        """
        # Handle dimensions
        if logits.data.ndim == 2:
            # Single sequence:  (seq_len, vocab_size)
            logits_3d = logits. data[np.newaxis, : , :]
            targets_2d = targets[np.newaxis, :]
        else:
            # Batch: (batch_size, seq_len, vocab_size)
            logits_3d = logits.data
            targets_2d = targets
        
        batch_size, seq_len, vocab_size = logits_3d.shape
        
        # Compute log-softmax for numerical stability
        # log_softmax(x) = x - log(sum(exp(x)))
        logits_max = np.max(logits_3d, axis=-1, keepdims=True)
        logits_shifted = logits_3d - logits_max
        log_sum_exp = np. log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
        log_probs = logits_shifted - log_sum_exp  # (batch_size, seq_len, vocab_size)
        
        # Gather log probabilities for target tokens
        # log_probs[b, t, targets[b, t]] for each (b, t)
        batch_indices = np.arange(batch_size)[: , np.newaxis]
        seq_indices = np. arange(seq_len)[np.newaxis, :]
        target_log_probs = log_probs[batch_indices, seq_indices, targets_2d]  # (batch_size, seq_len)
        
        # Create mask for valid positions
        if mask is None:
            # Create mask from ignore_index
            mask = (targets_2d != self.ignore_index).astype(np.float32)
        
        # Apply label smoothing if specified
        if self. label_smoothing > 0:
            # Smoothed targets:  (1 - ε) * one_hot + ε / vocab_size
            smooth_loss = -log_probs. mean(axis=-1)  # Average over vocab
            nll_loss = -target_log_probs
            loss_per_token = (1 - self. label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss_per_token = -target_log_probs  # (batch_size, seq_len)
        
        # Apply mask and compute mean
        loss_per_token = loss_per_token * mask
        num_valid = mask.sum()
        
        if num_valid > 0:
            loss_value = loss_per_token.sum() / num_valid
        else:
            loss_value = 0.0
        
        # Create loss tensor with gradient support
        loss = Tensor(
            np.array(loss_value),
            requires_grad=logits.requires_grad,
            _children=(logits,),
            _op='cross_entropy'
        )
        
        # Store for backward pass
        stored_log_probs = log_probs
        stored_targets = targets_2d
        stored_mask = mask
        stored_num_valid = num_valid
        stored_batch_size = batch_size
        stored_seq_len = seq_len
        stored_vocab_size = vocab_size
        is_single = logits.data.ndim == 2
        
        def _backward():
            if logits.requires_grad:
                if logits.grad is None:
                    logits.grad = np.zeros_like(logits.data)
                
                # Gradient of cross-entropy with softmax: 
                # d_loss/d_logits = softmax(logits) - one_hot(targets)
                probs = np.exp(stored_log_probs)  # (batch_size, seq_len, vocab_size)
                
                grad = probs. copy()
                
                # Subtract 1 from target positions
                for b in range(stored_batch_size):
                    for t in range(stored_seq_len):
                        if stored_mask[b, t] > 0:
                            target_idx = stored_targets[b, t]
                            grad[b, t, target_idx] -= 1.0
                
                # Scale by mask and normalize
                grad = grad * stored_mask[: , : , np.newaxis]
                if stored_num_valid > 0:
                    grad = grad / stored_num_valid
                
                # Scale by upstream gradient
                grad = grad * loss. grad
                
                # Remove batch dimension if needed
                if is_single:
                    grad = grad[0]
                
                logits.grad += grad
        
        loss._backward = _backward
        
        return loss


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss. 
    
    Perplexity = exp(loss)
    
    Lower perplexity = better model. 
    Perplexity of 1 = perfect predictions.
    Perplexity of vocab_size = random guessing.
    """
    return np. exp(loss)


# ==================== TEST ====================

def test_cross_entropy():
    """Test cross-entropy loss."""
    print("\n" + "=" * 60)
    print("TESTING CROSS-ENTROPY LOSS")
    print("=" * 60)
    
    vocab_size = 10
    seq_len = 5
    
    # Create dummy logits and targets
    np.random.seed(42)
    logits = Tensor(np.random.randn(seq_len, vocab_size), requires_grad=True)
    targets = np.array([1, 3, 5, 7, 2])
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Targets:  {targets}")
    
    # Compute loss
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    
    print(f"\nLoss value:  {loss.data:. 4f}")
    print(f"Perplexity:  {compute_perplexity(loss. data):.4f}")
    
    # Test gradient
    loss. backward()
    print(f"\nLogits gradient shape: {logits.grad. shape}")
    print(f"Gradient sum per position (should be ~0): {logits.grad.sum(axis=-1)}")
    
    # Test with mask (ignore some positions)
    print("\n--- With Mask ---")
    mask = np.array([1, 1, 1, 0, 0]).astype(np. float32)  # Ignore last 2
    
    logits_masked = Tensor(np.random.randn(seq_len, vocab_size), requires_grad=True)
    loss_masked = loss_fn(logits_masked, targets, mask=mask)
    
    print(f"Mask: {mask}")
    print(f"Loss with mask: {loss_masked.data:.4f}")
    
    # Test with batch
    print("\n--- Batch Test ---")
    batch_size = 3
    batch_logits = Tensor(np.random.randn(batch_size, seq_len, vocab_size), requires_grad=True)
    batch_targets = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Batch logits shape: {batch_logits. shape}")
    print(f"Batch targets shape: {batch_targets.shape}")
    
    batch_loss = loss_fn(batch_logits, batch_targets)
    print(f"Batch loss: {batch_loss.data:.4f}")
    
    batch_loss.backward()
    print(f"Batch gradient shape: {batch_logits.grad. shape}")
    
    print("\n✅ Cross-entropy loss tests passed!")


if __name__ == "__main__":
    test_cross_entropy()