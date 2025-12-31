"""
Tensor Class with Automatic Differentiation (Autograd)

This is the foundation of our neural network. 
We implement forward pass AND backward pass (gradient computation) from scratch. 

Similar to PyTorch's Tensor, but built from scratch to understand every detail.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Set
from enum import Enum


class Tensor:
    """
    A tensor that tracks operations for automatic differentiation.
    
    Key concepts:
    - data: The actual numpy array with values
    - grad: Gradient of loss with respect to this tensor (same shape as data)
    - _backward: Function to compute gradients during backpropagation
    - _prev:  Set of tensors that were used to create this tensor
    - requires_grad: Whether to track gradients for this tensor
    """
    
    def __init__(
        self, 
        data:  Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        _children:  Tuple['Tensor', ...] = (),
        _op:  str = ''
    ):
        """
        Initialize a tensor.
        
        Args:
            data: The numerical data (will be converted to numpy array)
            requires_grad: If True, gradients will be computed for this tensor
            _children: Tensors used to create this one (for autograd graph)
            _op: The operation that created this tensor (for debugging)
        """
        # Convert to numpy array if needed
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        elif isinstance(data, (list, float, int, np.floating, np.integer)):
            self.data = np.array(data, dtype=np.float32)
        else:
            raise TypeError(f"Cannot create tensor from {type(data)}")
        
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        
        # Autograd graph tracking
        self._backward = lambda: None  # Function to compute gradients
        self._prev: Set[Tensor] = set(_children)
        self._op = _op  # Operation name (for debugging)
    
    @property
    def shape(self) -> Tuple[int, ... ]:
        """Return the shape of the tensor."""
        return self.data.shape
    
    @property
    def dtype(self):
        """Return the data type."""
        return self.data.dtype
    
    def __repr__(self) -> str:
        """String representation."""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_str})"
    
    def __getitem__(self, idx):
        """Enable indexing like tensor[0] or tensor[1:3]."""
        return Tensor(self.data[idx], requires_grad=self.requires_grad)
    
    # ==================== BASIC OPERATIONS ====================
    
    def __add__(self, other:  Union['Tensor', float, int]) -> 'Tensor':
        """
        Addition:  self + other
        
        Forward: out = a + b
        Backward: ∂L/∂a = ∂L/∂out, ∂L/∂b = ∂L/∂out
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='+'
        )
        
        def _backward():
            if self.requires_grad:
                # Gradient flows back unchanged for addition
                # Handle broadcasting by summing over broadcasted dimensions
                grad = out.grad
                
                # Sum over dimensions that were broadcasted
                if self.data.shape != out.data.shape:
                    # Find axes that were broadcasted
                    ndim_diff = len(out.data.shape) - len(self.data.shape)
                    axes_to_sum = list(range(ndim_diff))
                    
                    for i, (s1, s2) in enumerate(zip(self.data.shape, out.data.shape[ndim_diff:])):
                        if s1 == 1 and s2 > 1:
                            axes_to_sum.append(i + ndim_diff)
                    
                    if axes_to_sum:
                        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
                        grad = grad.reshape(self.data.shape)
                
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad
            
            if other.requires_grad:
                grad = out.grad
                
                if other.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(other.data.shape)
                    axes_to_sum = list(range(ndim_diff))
                    
                    for i, (s1, s2) in enumerate(zip(other.data.shape, out.data.shape[ndim_diff:])):
                        if s1 == 1 and s2 > 1:
                            axes_to_sum.append(i + ndim_diff)
                    
                    if axes_to_sum:
                        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
                        grad = grad.reshape(other.data.shape)
                
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        """Right addition:  other + self"""
        return self.__add__(other)
    
    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Element-wise multiplication: self * other
        
        Forward: out = a * b
        Backward: ∂L/∂a = ∂L/∂out * b, ∂L/∂b = ∂L/∂out * a
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='*'
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                
                # Handle broadcasting
                if self.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(self.data.shape)
                    axes_to_sum = list(range(ndim_diff))
                    for i, (s1, s2) in enumerate(zip(self.data.shape, out.data.shape[ndim_diff:])):
                        if s1 == 1 and s2 > 1:
                            axes_to_sum.append(i + ndim_diff)
                    if axes_to_sum:
                        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
                        grad = grad.reshape(self.data.shape)
                
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad
            
            if other.requires_grad:
                grad = out.grad * self.data
                
                if other.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(other.data.shape)
                    axes_to_sum = list(range(ndim_diff))
                    for i, (s1, s2) in enumerate(zip(other.data.shape, out.data.shape[ndim_diff:])):
                        if s1 == 1 and s2 > 1:
                            axes_to_sum.append(i + ndim_diff)
                    if axes_to_sum:
                        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
                        grad = grad.reshape(other.data.shape)
                
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other: Union[float, int]) -> 'Tensor': 
        """Right multiplication: other * self"""
        return self.__mul__(other)
    
    def __neg__(self) -> 'Tensor': 
        """Negation: -self"""
        return self * -1
    
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor': 
        """Subtraction: self - other"""
        return self + (-other if isinstance(other, Tensor) else -other)
    
    def __rsub__(self, other:  Union[float, int]) -> 'Tensor':
        """Right subtraction: other - self"""
        return other + (-self)
    
    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Division: self / other"""
        return self * (other ** -1 if isinstance(other, Tensor) else 1/other)
    
    def __pow__(self, power:  Union[int, float]) -> 'Tensor':
        """
        Power: self ** power
        
        Forward:  out = a^n
        Backward:  ∂L/∂a = ∂L/∂out * n * a^(n-1)
        """
        assert isinstance(power, (int, float)), "Power must be int or float"
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'**{power}'
        )
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * power * (self.data ** (power - 1))
        
        out._backward = _backward
        return out
    
    # ==================== MATRIX OPERATIONS ====================
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication: self @ other
        
        For 2D tensors: 
        Forward: out = A @ B  where A is (m, n) and B is (n, p)
        Backward: ∂L/∂A = ∂L/∂out @ B. T
                  ∂L/∂B = A.T @ ∂L/∂out
        """
        out = Tensor(
            np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='@'
        )
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # ∂L/∂A = ∂L/∂out @ B.T
                self.grad += np.matmul(out.grad, other.data.swapaxes(-1, -2))
            
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # ∂L/∂B = A.T @ ∂L/∂out
                other.grad += np.matmul(self.data.swapaxes(-1, -2), out.grad)
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Enable @ operator for matrix multiplication."""
        return self.matmul(other)
    
    # ==================== SHAPE OPERATIONS ====================
    
    def sum(self, axis:  Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor': 
        """
        Sum elements along axis.
        
        Forward: out = sum(x)
        Backward:  ∂L/∂x = ∂L/∂out (broadcasted to x's shape)
        """
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum'
        )
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                
                # Gradient is broadcasted back to original shape
                grad = out.grad
                if not keepdims and axis is not None: 
                    # Need to add back the reduced dimensions for broadcasting
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                
                self.grad += np.broadcast_to(grad, self.data.shape)
        
        out._backward = _backward
        return out
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Mean of elements along axis. 
        
        Forward:  out = mean(x) = sum(x) / n
        Backward: ∂L/∂x = ∂L/∂out / n
        """
        if axis is None: 
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = np.prod([self.data.shape[ax] for ax in axis])
        
        return self.sum(axis=axis, keepdims=keepdims) / n
    
    def reshape(self, *shape) -> 'Tensor':
        """
        Reshape tensor. 
        
        Forward: out = reshape(x, new_shape)
        Backward:  ∂L/∂x = reshape(∂L/∂out, original_shape)
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='reshape'
        )
        
        original_shape = self.data.shape
        
        def _backward():
            if self.requires_grad:
                if self.grad is None: 
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad.reshape(original_shape)
        
        out._backward = _backward
        return out
    
    def transpose(self, *axes) -> 'Tensor':
        """Transpose the tensor."""
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        
        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='transpose'
        )
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if axes is None:
                    self.grad += np.transpose(out.grad)
                else:
                    # Inverse permutation
                    inv_axes = [0] * len(axes)
                    for i, ax in enumerate(axes):
                        inv_axes[ax] = i
                    self.grad += np.transpose(out.grad, inv_axes)
        
        out._backward = _backward
        return out
    
    @property
    def T(self) -> 'Tensor': 
        """Transpose property."""
        return self.transpose()
    
    # ==================== ACTIVATION FUNCTIONS ====================
    
    def relu(self) -> 'Tensor':
        """
        ReLU activation: max(0, x)
        
        Forward: out = max(0, x)
        Backward: ∂L/∂x = ∂L/∂out if x > 0, else 0
        """
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='relu'
        )
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * (self.data > 0)
        
        out._backward = _backward
        return out
    
    def gelu(self) -> 'Tensor':
        """
        GELU activation (Gaussian Error Linear Unit).
        Used in GPT-2, BERT, and most modern transformers.
        
        Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        """
        # Constants
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        
        # Forward
        x = self.data
        cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
        out_data = x * cdf
        
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='gelu'
        )
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                
                # GELU gradient (using approximation)
                x = self.data
                tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x**3)
                tanh_val = np.tanh(tanh_arg)
                sech2 = 1 - tanh_val**2
                
                # d(gelu)/dx = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * sqrt(2/π) * (1 + 3*0.044715*x^2)
                grad = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)
                
                self.grad += out.grad * grad
        
        out._backward = _backward
        return out
    
    def softmax(self, axis: int = -1) -> 'Tensor': 
        """
        Softmax activation. 
        
        Forward: out = exp(x - max(x)) / sum(exp(x - max(x)))
        """
        # Subtract max for numerical stability
        x_max = np.max(self.data, axis=axis, keepdims=True)
        exp_x = np.exp(self.data - x_max)
        out_data = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='softmax'
        )
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                
                # Softmax backward pass
                # ∂L/∂x_i = s_i * (∂L/∂s_i - sum_j(∂L/∂s_j * s_j))
                s = out.data
                grad_sum = np.sum(out.grad * s, axis=axis, keepdims=True)
                self.grad += s * (out.grad - grad_sum)
        
        out._backward = _backward
        return out
    
    def log(self) -> 'Tensor':
        """
        Natural logarithm. 
        
        Forward: out = log(x)
        Backward: ∂L/∂x = ∂L/∂out / x
        """
        out = Tensor(
            np.log(self.data + 1e-8),  # Add small epsilon for numerical stability
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='log'
        )
        
        def _backward():
            if self.requires_grad: 
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad / (self.data + 1e-8)
        
        out._backward = _backward
        return out
    
    def exp(self) -> 'Tensor': 
        """
        Exponential.
        
        Forward: out = exp(x)
        Backward:  ∂L/∂x = ∂L/∂out * exp(x)
        """
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='exp'
        )
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * out.data
        
        out._backward = _backward
        return out
    
    # ==================== BACKPROPAGATION ====================
    
    def backward(self):
        """
        Compute gradients for all tensors that contributed to this tensor. 
        
        Uses reverse-mode automatic differentiation (backpropagation).
        This is the same algorithm used by PyTorch! 
        """
        # Build topological order of all tensors in the computation graph
        topo_order = []
        visited = set()
        
        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_topo(child)
                topo_order.append(tensor)
        
        build_topo(self)
        
        # Initialize gradient of output to 1 (∂L/∂L = 1)
        self.grad = np.ones_like(self.data)
        
        # Propagate gradients backwards through the graph
        for tensor in reversed(topo_order):
            tensor._backward()
    
    def zero_grad(self):
        """Reset gradient to None."""
        self.grad = None
    
    # ==================== UTILITY METHODS ====================
    
    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array."""
        return self.data
    
    def item(self) -> float:
        """Return a scalar value (only works for single-element tensors)."""
        return self.data.item()
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of zeros."""
        return Tensor(np.zeros(shape), requires_grad=requires_grad)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of ones."""
        return Tensor(np.ones(shape), requires_grad=requires_grad)
    
    @staticmethod
    def randn(*shape, requires_grad:  bool = False) -> 'Tensor': 
        """Create a tensor with random normal values."""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)
    
    @staticmethod
    def uniform(low: float, high: float, shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor with uniform random values."""
        return Tensor(np.random.uniform(low, high, shape), requires_grad=requires_grad)


# ==================== TEST AUTOGRAD ====================

def test_autograd():
    """Test that our autograd implementation is correct."""
    
    print("=" * 60)
    print("TESTING AUTOGRAD IMPLEMENTATION")
    print("=" * 60)
    
    # Test 1: Simple addition
    print("\n--- Test 1: Addition ---")
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a + b
    loss = c.sum()
    loss.backward()
    
    print(f"a = {a.data}, b = {b.data}")
    print(f"c = a + b = {c.data}")
    print(f"loss = sum(c) = {loss.data}")
    print(f"∂loss/∂a = {a.grad}  (expected: [1, 1])")
    print(f"∂loss/∂b = {b.grad}  (expected: [1, 1])")
    assert np.allclose(a.grad, [1, 1]), "Addition gradient failed!"
    assert np.allclose(b.grad, [1, 1]), "Addition gradient failed!"
    print("✅ Passed!")
    
    # Test 2: Multiplication
    print("\n--- Test 2: Multiplication ---")
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    loss = c.sum()
    loss.backward()
    
    print(f"a = {a.data}, b = {b.data}")
    print(f"c = a * b = {c.data}")
    print(f"loss = sum(c) = {loss.data}")
    print(f"∂loss/∂a = {a.grad}  (expected: [4, 5] = b)")
    print(f"∂loss/∂b = {b.grad}  (expected: [2, 3] = a)")
    assert np.allclose(a.grad, [4, 5]), "Multiplication gradient failed!"
    assert np.allclose(b.grad, [2, 3]), "Multiplication gradient failed!"
    print("✅ Passed!")
    
    # Test 3: Matrix multiplication
    print("\n--- Test 3: Matrix Multiplication ---")
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)  # 2x2
    b = Tensor([[5, 6], [7, 8]], requires_grad=True)  # 2x2
    c = a @ b  # 2x2
    loss = c.sum()
    loss.backward()
    
    print(f"a = \n{a.data}")
    print(f"b = \n{b.data}")
    print(f"c = a @ b = \n{c.data}")
    print(f"∂loss/∂a = \n{a.grad}")
    print(f"∂loss/∂b = \n{b.grad}")
    # Gradient of sum(A @ B) w.r.t. A is ones @ B.T = [[11, 15], [11, 15]]
    expected_a_grad = np.array([[11, 15], [11, 15]])
    # Gradient of sum(A @ B) w.r.t. B is A.T @ ones = [[4, 4], [6, 6]]
    expected_b_grad = np.array([[4, 4], [6, 6]])
    assert np.allclose(a.grad, expected_a_grad), f"Matmul gradient for a failed! Got {a.grad}"
    assert np.allclose(b.grad, expected_b_grad), f"Matmul gradient for b failed! Got {b.grad}"
    print("✅ Passed!")
    
    # Test 4: Chain of operations
    print("\n--- Test 4: Chain of Operations ---")
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2 + 1  # y = 2x + 1
    z = y ** 2     # z = (2x + 1)^2
    loss = z.sum()
    loss.backward()
    
    # ∂loss/∂x = ∂(sum((2x+1)^2))/∂x = 2 * (2x+1) * 2 = 4 * (2x+1)
    # For x = [1, 2, 3]:  4 * [3, 5, 7] = [12, 20, 28]
    expected = 4 * (2 * x.data + 1)
    print(f"x = {x.data}")
    print(f"y = 2x + 1 = {y.data}")
    print(f"z = y^2 = {z.data}")
    print(f"loss = sum(z) = {loss.data}")
    print(f"∂loss/∂x = {x.grad}  (expected: {expected})")
    assert np.allclose(x.grad, expected), "Chain rule gradient failed!"
    print("✅ Passed!")
    
    # Test 5: ReLU
    print("\n--- Test 5: ReLU ---")
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = x.relu()
    loss = y.sum()
    loss.backward()
    
    print(f"x = {x.data}")
    print(f"y = relu(x) = {y.data}")
    print(f"∂loss/∂x = {x.grad}  (expected: [0, 0, 0, 1, 1])")
    assert np.allclose(x.grad, [0, 0, 0, 1, 1]), "ReLU gradient failed!"
    print("✅ Passed!")
    
    # Test 6: Mean
    print("\n--- Test 6: Mean ---")
    x = Tensor([2.0, 4.0, 6.0, 8.0], requires_grad=True)
    y = x.mean()
    y.backward()
    
    print(f"x = {x.data}")
    print(f"y = mean(x) = {y.data}")
    print(f"∂y/∂x = {x.grad}  (expected: [0.25, 0.25, 0.25, 0.25])")
    assert np.allclose(x.grad, [0.25, 0.25, 0.25, 0.25]), "Mean gradient failed!"
    print("✅ Passed!")
    
    print("\n" + "=" * 60)
    print("ALL AUTOGRAD TESTS PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_autograd()