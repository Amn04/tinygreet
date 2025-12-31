"""
Optimizers for Training

Implementing Adam optimizer from scratch - the most popular optimizer for LLMs! 
"""

import numpy as np
from typing import List, Optional

from tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    The simplest optimizer:  θ = θ - lr * ∇θ
    """
    
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        Initialize SGD optimizer.
        
        Args: 
            parameters: List of model parameters
            lr:  Learning rate
            momentum: Momentum factor (0 = no momentum)
            weight_decay: L2 regularization factor
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Momentum buffers
        if momentum > 0:
            self.velocity = [np.zeros_like(p.data) for p in parameters]
        else: 
            self.velocity = None
    
    def step(self):
        """Perform a single optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad. copy()
            
            # L2 regularization
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            # Momentum
            if self.velocity is not None: 
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = self.velocity[i]
            
            # Update parameters
            param.data -= self.lr * grad
    
    def zero_grad(self):
        """Reset all gradients to None."""
        for param in self.parameters:
            param.grad = None


class Adam:
    """
    Adam Optimizer (Adaptive Moment Estimation)
    
    The most popular optimizer for training neural networks!
    
    Adam combines: 
    - Momentum (using exponential moving average of gradients)
    - RMSprop (using exponential moving average of squared gradients)
    
    Update rule: 
        m = β₁ * m + (1 - β₁) * g          # First moment (mean)
        v = β₂ * v + (1 - β₂) * g²         # Second moment (variance)
        m̂ = m / (1 - β₁^t)                  # Bias correction
        v̂ = v / (1 - β₂^t)                  # Bias correction
        θ = θ - lr * m̂ / (√v̂ + ε)          # Update
    
    This is exactly how PyTorch's Adam works!
    """
    
    def __init__(
        self,
        parameters:  List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            parameters: List of model parameters
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default:  1e-8)
            weight_decay: L2 regularization factor (default: 0)
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.m = [np.zeros_like(p. data) for p in parameters]  # First moment
        self. v = [np. zeros_like(p.data) for p in parameters]  # Second moment
        
        # Time step (for bias correction)
        self.t = 0
    
    def step(self):
        """
        Perform a single optimization step.
        
        This updates all parameters based on their gradients.
        """
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param. grad is None: 
                continue
            
            grad = param. grad.copy()
            
            # L2 regularization (weight decay)
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self. beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self. t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self. v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param. data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        """Reset all gradients to None."""
        for param in self.parameters:
            param.grad = None
    
    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v':  [v.copy() for v in self. v],
            'lr': self.lr,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from checkpoint."""
        self.t = state_dict['t']
        self.m = [m.copy() for m in state_dict['m']]
        self.v = [v.copy() for v in state_dict['v']]
        self. lr = state_dict['lr']


class AdamW(Adam):
    """
    AdamW Optimizer (Adam with decoupled weight decay)
    
    This is the preferred optimizer for modern LLMs like GPT-3, BERT, etc. 
    
    The key difference from Adam: 
    - Weight decay is applied directly to parameters, not added to gradients
    - This is mathematically different and often works better
    
    Update rule: 
        θ = θ - lr * (m̂ / (√v̂ + ε) + weight_decay * θ)
    """
    
    def step(self):
        """Perform a single optimization step with decoupled weight decay."""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad. copy()
            
            # Update biased first moment estimate
            self.m[i] = self. beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self. beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self. beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Compute update step
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            
            # Decoupled weight decay (applied directly to parameters)
            if self.weight_decay > 0:
                update += self. weight_decay * param.data
            
            # Update parameters
            param.data -= self.lr * update


class LearningRateScheduler:
    """
    Learning Rate Scheduler
    
    Implements warmup + cosine decay, which is standard for training LLMs.
    
    Schedule:
    1. Warmup: Linear increase from 0 to max_lr over warmup_steps
    2. Decay: Cosine decay from max_lr to min_lr
    """
    
    def __init__(
        self,
        optimizer: Adam,
        max_lr: float,
        min_lr: float = 0.0,
        warmup_steps: int = 100,
        total_steps: int = 1000
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            max_lr: Maximum learning rate (after warmup)
            min_lr:  Minimum learning rate (at end of training)
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self. warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np. pi * progress))
        
        self. optimizer.lr = lr
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr


# ==================== GRADIENT CLIPPING ====================

def clip_grad_norm(parameters: List[Tensor], max_norm:  float) -> float:
    """
    Clip gradient norm. 
    
    This prevents exploding gradients by scaling down gradients
    if their total norm exceeds max_norm. 
    
    Args:
        parameters:  List of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns: 
        The total gradient norm (before clipping)
    """
    # Compute total gradient norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += np.sum(param.grad ** 2)
    total_norm = np. sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad *= clip_coef
    
    return total_norm


# ==================== TESTS ====================

def test_adam():
    """Test Adam optimizer."""
    print("\n" + "=" * 60)
    print("TESTING ADAM OPTIMIZER")
    print("=" * 60)
    
    # Simple optimization problem:  minimize (x - 3)^2
    x = Tensor(np.array([0.0]), requires_grad=True)
    optimizer = Adam([x], lr=0.1)
    
    print(f"\nOptimizing (x - 3)²")
    print(f"Starting x = {x.data[0]:. 4f}")
    print(f"Target x = 3.0")
    
    for step in range(50):
        # Forward:  loss = (x - 3)^2
        loss = (x - 3.0) ** 2
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Update
        optimizer.step()
        
        if step % 10 == 0:
            print(f"  Step {step: 3d}: x = {x.data[0]:.4f}, loss = {loss.data[0]:.6f}")
    
    print(f"\nFinal x = {x. data[0]:. 4f} (should be ~3.0)")
    assert abs(x.data[0] - 3.0) < 0.1, "Adam optimization failed!"
    
    print("\n✅ Adam optimizer tests passed!")


def test_lr_scheduler():
    """Test learning rate scheduler."""
    print("\n" + "=" * 60)
    print("TESTING LEARNING RATE SCHEDULER")
    print("=" * 60)
    
    x = Tensor(np.array([0.0]), requires_grad=True)
    optimizer = Adam([x], lr=0.001)
    
    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        max_lr=0.001,
        min_lr=0.0001,
        warmup_steps=10,
        total_steps=50
    )
    
    print("\nLearning rate schedule:")
    print("-" * 40)
    
    for step in range(50):
        lr = scheduler.step()
        if step % 5 == 0:
            phase = "warmup" if step < 10 else "decay"
            bar_len = int(lr / 0.001 * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  Step {step: 3d} ({phase: 6s}): lr = {lr:.6f} |{bar}|")
    
    print("\n✅ Learning rate scheduler tests passed!")


if __name__ == "__main__":
    test_adam()
    test_lr_scheduler()